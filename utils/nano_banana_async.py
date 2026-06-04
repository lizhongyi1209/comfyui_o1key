import asyncio
import base64
import json
import time
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import aiohttp
from PIL import Image

from .http_error import (
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    RETRYABLE_STATUS_CODES,
    _compute_delay,
    extract_structured_error_message,
    get_friendly_message,
)
from ..clients.gemini_client import GeminiAPIClient


_MAX_BODY_BYTES = 20_000_000
_BODY_TARGET_BYTES = int(_MAX_BODY_BYTES * 0.8)
_SUBMIT_ENDPOINT = "/async/v1/generateImage"
_TASK_ENDPOINT = "/async/v1/tasks/{task_id}"
_POLL_SCHEDULE = [5.0, 20.0]
_POLL_INTERVAL = 3.0
_MAX_WAIT_SECONDS = 900.0
_INTERRUPT_STEP = 0.2
_RUNNING_PROGRESS_MAX = 0.99
_POLL_LOG_ENABLED = False

_SUCCESS_STATUSES = {"success", "succeed", "succeeded", "completed", "done", "finished"}
_FAILURE_STATUSES = {
    "failure",
    "fail",
    "failed",
    "error",
    "expired",
    "timeout",
    "timed_out",
    "cancel",
    "canceled",
    "cancelled",
    "rejected",
}
_RUNNING_STATUSES = {
    "submitted",
    "queued",
    "pending",
    "running",
    "processing",
    "in_progress",
    "in-progress",
    "created",
}


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _json_dumps(body: Dict[str, Any]) -> str:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


def _json_size(body: Dict[str, Any]) -> int:
    return len(_json_dumps(body).encode("utf-8"))


def _scale_images(images: List[Image.Image], scale: float) -> List[Image.Image]:
    if scale >= 1.0:
        return images
    scaled = []
    for img in images:
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        scaled.append(img.resize((new_w, new_h), Image.Resampling.LANCZOS))
    return scaled


def _encode_image_data_url(
    image: Image.Image,
    image_format: str,
    quality: Optional[int] = None,
) -> str:
    buffered = BytesIO()
    working = image
    fmt = image_format.upper()
    save_kwargs = {"format": fmt}

    if fmt == "JPEG":
        if working.mode != "RGB":
            working = working.convert("RGB")
        save_kwargs.update({"quality": quality or 90, "optimize": True, "subsampling": 2})
        mime_type = "image/jpeg"
    else:
        if working.mode == "RGBA":
            working = working.convert("RGB")
        mime_type = "image/png"

    working.save(buffered, **save_kwargs)
    encoded = base64.b64encode(buffered.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _encode_image_data_urls(
    images: Sequence[Image.Image],
    image_format: str,
    quality: Optional[int] = None,
) -> List[str]:
    return [_encode_image_data_url(img, image_format, quality) for img in images]


def _fit_image_data_urls_to_body_limit(
    images: Sequence[Image.Image],
    build_body: Callable[[List[str]], Dict[str, Any]],
) -> Tuple[List[str], List[Image.Image], str, int]:
    working_images = list(images)
    image_urls = _encode_image_data_urls(working_images, "PNG")
    body_size = _json_size(build_body(image_urls))
    if body_size <= _BODY_TARGET_BYTES:
        return image_urls, working_images, "PNG", body_size

    for _ in range(10):
        if body_size <= _BODY_TARGET_BYTES:
            break
        ratio = _BODY_TARGET_BYTES / max(body_size, 1)
        scale = min(0.98, ratio ** 0.5)
        working_images = _scale_images(working_images, scale)
        image_urls = _encode_image_data_urls(working_images, "PNG")
        body_size = _json_size(build_body(image_urls))

    return image_urls, working_images, "PNG", body_size


def _shorten_base64_for_log(value: Any, max_len: int = 160) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in ("data", "b64_json", "base64", "image_base64") and isinstance(item, str) and len(item) > max_len:
                result[key] = f"<base64 data, {len(item)} chars>"
            else:
                result[key] = _shorten_base64_for_log(item, max_len)
        return result
    if isinstance(value, list):
        return [_shorten_base64_for_log(item, max_len) for item in value]
    if isinstance(value, str) and value.startswith("data:image") and len(value) > max_len:
        return f"<data image url, {len(value)} chars>"
    return value


def _log_body(label: str, text_or_body: Any) -> None:
    if isinstance(text_or_body, str):
        try:
            text_or_body = json.loads(text_or_body)
        except Exception:
            print(f"{label}\n{text_or_body}")
            return
    print(
        f"{label}\n"
        f"{json.dumps(_shorten_base64_for_log(text_or_body), ensure_ascii=False, indent=2)}"
    )


def build_nano_banana_submit_body(
    model: str,
    prompt: str,
    resolution: str,
    aspect_ratio: str,
    images: Optional[List[Image.Image]] = None,
    enable_grounding: bool = False,
    thinking_level: Optional[str] = None,
    request_log_enabled: bool = False,
    node_label: str = "Nano Banana",
) -> Dict[str, Any]:
    def _make_body(image_urls: List[str]) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": resolution,
        }
        if aspect_ratio and aspect_ratio != "智能":
            body["aspect_ratio"] = aspect_ratio
        if image_urls:
            body["images"] = image_urls
        if enable_grounding:
            body["google_search"] = True
        if thinking_level:
            body["thinking_level"] = thinking_level
        return body

    working_images = list(images or [])
    image_urls: List[str] = []

    if working_images:
        image_urls, working_images, _, _ = _fit_image_data_urls_to_body_limit(
            working_images,
            _make_body,
        )

    body = _make_body(image_urls)
    body_size = _json_size(body)

    if working_images and body_size > _MAX_BODY_BYTES:
        raise ValueError(
            f"Request body exceeds the 20MB limit after compression "
            f"({body_size / 1_000_000:.2f}MB). Reduce reference image count, "
            "image complexity, or prompt length."
        )
    if not working_images and body_size > _MAX_BODY_BYTES:
        raise ValueError(
            f"Request body exceeds the 20MB limit ({body_size / 1_000_000:.2f}MB). "
            "Shorten the prompt or system instructions."
        )

    if request_log_enabled:
        print(
            f"[{node_label} 异步请求体] {body_size / 1024:.1f}KB\n"
            f"{json.dumps(_shorten_base64_for_log(body), ensure_ascii=False, indent=2)}"
        )

    return body


async def _interruptible_sleep(
    seconds: float,
    check_interrupt: Optional[Callable[[], None]] = None,
) -> None:
    elapsed = 0.0
    while elapsed < seconds:
        if check_interrupt:
            check_interrupt()
        delay = min(_INTERRUPT_STEP, seconds - elapsed)
        await asyncio.sleep(delay)
        elapsed += delay
    if check_interrupt:
        check_interrupt()


def _payload_sources(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    queue = [payload]
    seen = set()
    while queue:
        current = queue.pop(0)
        if not isinstance(current, dict):
            continue
        obj_id = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        yield current
        for key in ("data", "result", "response", "output", "task_result", "content"):
            value = current.get(key)
            if isinstance(value, dict):
                queue.append(value)


def _extract_task_id(payload: Dict[str, Any]) -> str:
    for source in _payload_sources(payload):
        for key in ("task_id", "taskId", "id"):
            value = source.get(key)
            if value:
                return str(value)
    raise RuntimeError(f"提交响应中未找到 task_id: {payload}")


def _extract_status(payload: Dict[str, Any]) -> str:
    statuses = []
    for source in _payload_sources(payload):
        for key in ("status", "task_status", "state", "task_state"):
            value = source.get(key)
            if value is not None and str(value).strip():
                statuses.append(str(value).strip())

    for status in statuses:
        normalized = status.lower()
        if normalized in _FAILURE_STATUSES or any(
            token in normalized for token in ("fail", "error", "reject", "timeout", "cancel")
        ):
            return status
    for status in statuses:
        if status.lower() in _RUNNING_STATUSES:
            return status
    for status in statuses:
        if status.lower() in _SUCCESS_STATUSES:
            return status
    return statuses[0] if statuses else ""


def _coerce_progress_fraction(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        progress = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        has_percent_suffix = text.endswith("%")
        if has_percent_suffix:
            text = text[:-1].strip()
        try:
            progress = float(text)
        except ValueError:
            return None
        if has_percent_suffix:
            progress /= 100.0
    else:
        return None

    if progress > 1.0:
        progress /= 100.0
    return max(0.0, min(progress, 1.0))


def _extract_progress(payload: Dict[str, Any]) -> Optional[float]:
    for source in _payload_sources(payload):
        for key in ("progress", "percentage", "percent"):
            progress = _coerce_progress_fraction(source.get(key))
            if progress is not None:
                return progress

        for key in ("progressInfo", "progress_info"):
            info = source.get(key)
            if not isinstance(info, dict):
                continue
            for field in ("progress", "percentage", "percent"):
                progress = _coerce_progress_fraction(info.get(field))
                if progress is not None:
                    return progress

    return None


def _is_failure_status(normalized_status: str) -> bool:
    return normalized_status in _FAILURE_STATUSES or any(
        token in normalized_status for token in ("fail", "error", "reject", "timeout", "cancel")
    )


def _extract_error_message(payload: Dict[str, Any]) -> str:
    for source in _payload_sources(payload):
        error = source.get("error")
        if isinstance(error, dict):
            for key in ("message", "msg", "detail", "reason", "code"):
                value = error.get(key)
                if value:
                    return str(value)
        elif error:
            message = extract_structured_error_message(str(error))
            return message or str(error)

        for key in (
            "fail_reason",
            "failure_reason",
            "task_status_msg",
            "status_msg",
            "error_message",
            "message",
            "msg",
            "reason",
            "detail",
        ):
            value = source.get(key)
            if value:
                message = extract_structured_error_message(str(value))
                return message or str(value)
    return "未知错误"


async def _submit_task(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    body: Dict[str, Any],
    node_label: str,
    log_body_enabled: bool = False,
) -> str:
    url = f"{base_url}{_SUBMIT_ENDPOINT}"
    timeout = aiohttp.ClientTimeout(total=120, connect=30, sock_read=120)
    last_status = None
    last_text = ""

    for attempt in range(DEFAULT_MAX_RETRIES + 1):
        try:
            async with session.post(
                url,
                headers=_headers(api_key),
                data=_json_dumps(body).encode("utf-8"),
                timeout=timeout,
            ) as resp:
                text = await resp.text()
                if log_body_enabled:
                    _log_body(f"[{node_label} 异步提交响应] HTTP {resp.status}", text)
                if resp.status not in (200, 201, 202):
                    last_status = resp.status
                    last_text = text
                    if resp.status in RETRYABLE_STATUS_CODES and attempt < DEFAULT_MAX_RETRIES:
                        friendly = get_friendly_message(resp.status)
                        delay = _compute_delay(
                            attempt,
                            DEFAULT_BASE_DELAY,
                            DEFAULT_MAX_DELAY,
                            DEFAULT_BACKOFF_FACTOR,
                        )
                        print(f"{node_label}: {friendly} {delay:.1f}s 后重试提交 ({attempt + 1}/{DEFAULT_MAX_RETRIES})...")
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(get_friendly_message(resp.status, text))

                try:
                    data = json.loads(text)
                except Exception:
                    raise RuntimeError(f"提交响应 JSON 解析失败: {text[:500]}") from None
                task_id = _extract_task_id(data)
                status = _extract_status(data) or "SUBMITTED"
                print(f"{node_label}: 异步任务已提交 | task_id={task_id} | status={status}")
                return task_id
        except (
            aiohttp.ClientConnectorError,
            aiohttp.ClientOSError,
            aiohttp.ServerDisconnectedError,
            asyncio.TimeoutError,
        ) as e:
            last_text = str(e)
            if attempt < DEFAULT_MAX_RETRIES:
                delay = _compute_delay(
                    attempt,
                    DEFAULT_BASE_DELAY,
                    DEFAULT_MAX_DELAY,
                    DEFAULT_BACKOFF_FACTOR,
                )
                print(f"{node_label}: 网络连接失败，{delay:.1f}s 后重试提交 ({attempt + 1}/{DEFAULT_MAX_RETRIES})...")
                await asyncio.sleep(delay)
                continue
            raise RuntimeError(
                f"网络连接失败，无法连接 {url}: {str(e)}。请切换节点里的网络线路，或检查 VPN/代理/防火墙。"
            ) from None

    raise RuntimeError(get_friendly_message(last_status or 0, last_text))


async def _poll_task(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    task_id: str,
    node_label: str,
    check_interrupt: Optional[Callable[[], None]] = None,
    log_body_enabled: bool = False,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    url = f"{base_url}{_TASK_ENDPOINT.format(task_id=task_id)}"
    start_time = time.time()
    last_poll_at = start_time
    poll_count = 0

    while True:
        if check_interrupt:
            check_interrupt()

        if poll_count < len(_POLL_SCHEDULE):
            next_poll_at = start_time + _POLL_SCHEDULE[poll_count]
        else:
            next_poll_at = last_poll_at + _POLL_INTERVAL

        sleep_time = next_poll_at - time.time()
        if sleep_time > 0:
            await _interruptible_sleep(sleep_time, check_interrupt=check_interrupt)

        last_poll_at = time.time()
        elapsed = last_poll_at - start_time
        if elapsed > _MAX_WAIT_SECONDS:
            raise RuntimeError(f"任务 {task_id} 超时（>{int(_MAX_WAIT_SECONDS)}秒），请稍后用 task_id 查询结果")

        poll_count += 1
        async with session.get(url, headers=_headers(api_key)) as resp:
            text = await resp.text()
            if log_body_enabled:
                _log_body(f"[{node_label} 任务查询响应 #{poll_count}] HTTP {resp.status}", text)
            if resp.status != 200:
                raise RuntimeError(get_friendly_message(resp.status, text))
            try:
                payload = json.loads(text)
            except Exception:
                raise RuntimeError(f"任务查询响应 JSON 解析失败: {text[:500]}") from None

        status = _extract_status(payload) or "UNKNOWN"
        normalized = status.lower()
        progress = _extract_progress(payload)
        is_failure = _is_failure_status(normalized)
        if _POLL_LOG_ENABLED:
            progress_text = ""
            if progress is not None and not is_failure:
                displayed_progress = 1.0 if normalized in _SUCCESS_STATUSES else min(progress, _RUNNING_PROGRESS_MAX)
                progress_text = f" | progress={displayed_progress * 100:.0f}%"
            print(f"{node_label}: 查询任务 #{poll_count} | task_id={task_id} | status={status}{progress_text}")

        if normalized in _SUCCESS_STATUSES:
            if progress_callback:
                progress_callback(1.0)
            return payload
        if is_failure:
            raise RuntimeError(f"任务失败: {_extract_error_message(payload)}")
        if normalized not in _RUNNING_STATUSES:
            raise RuntimeError(f"未知任务状态 {status}: {payload}")
        if progress_callback and progress is not None:
            progress_callback(min(progress, _RUNNING_PROGRESS_MAX))


async def _image_from_url_or_data(
    value: str,
    session: aiohttp.ClientSession,
) -> Optional[Image.Image]:
    if not value:
        return None
    if value.startswith("data:image"):
        try:
            _, b64_data = value.split(",", 1)
            return Image.open(BytesIO(base64.b64decode(b64_data))).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"data URL 图片解码失败: {e}") from None
    if value.startswith("http"):
        async with session.get(value, allow_redirects=True) as resp:
            if resp.status != 200:
                raise RuntimeError(f"图片下载失败 ({resp.status}): {value}")
            img_bytes = await resp.read()
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    return None


async def _parse_direct_images(
    payload: Dict[str, Any],
    session: aiohttp.ClientSession,
) -> List[Image.Image]:
    images: List[Image.Image] = []

    async def _try_item(item: Any) -> None:
        if isinstance(item, str):
            img = await _image_from_url_or_data(item, session)
            if img:
                images.append(img)
            return
        if not isinstance(item, dict):
            return

        for key in ("url", "image_url", "result_url", "download_url"):
            img = await _image_from_url_or_data(str(item.get(key) or ""), session)
            if img:
                images.append(img)
                return

        b64_data = item.get("b64_json") or item.get("base64") or item.get("image_base64")
        if b64_data:
            images.append(Image.open(BytesIO(base64.b64decode(str(b64_data)))).convert("RGB"))
            return

        for inline_key in ("inline_data", "inlineData"):
            inline = item.get(inline_key)
            if isinstance(inline, dict) and inline.get("data"):
                images.append(Image.open(BytesIO(base64.b64decode(str(inline["data"])))).convert("RGB"))
                return

    for source in _payload_sources(payload):
        for key in ("image_url", "result_url", "url", "download_url"):
            img = await _image_from_url_or_data(str(source.get(key) or ""), session)
            if img:
                images.append(img)

        for key in ("images", "output_images", "outputs"):
            value = source.get(key)
            if isinstance(value, list):
                for item in value:
                    await _try_item(item)
            elif value:
                await _try_item(value)

    return images


async def _parse_task_images(
    task_payload: Dict[str, Any],
    session: aiohttp.ClientSession,
    api_key: str,
) -> List[Image.Image]:
    direct_images = await _parse_direct_images(task_payload, session)
    if direct_images:
        return direct_images

    client = GeminiAPIClient(api_key=api_key)
    last_error = None
    for source in _payload_sources(task_payload):
        if "candidates" not in source:
            continue
        try:
            images, _ = await client.parse_response_async(source, session=session)
            if images:
                return [img.convert("RGB") for img in images]
        except Exception as e:
            last_error = e

    if last_error is not None:
        raise RuntimeError(str(last_error)) from None
    raise RuntimeError(f"任务成功但未找到图片结果: {task_payload}")


async def generate_nano_banana_async(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    prompt: str,
    model: str,
    resolution: str,
    aspect_ratio: str,
    images: Optional[List[Image.Image]] = None,
    enable_grounding: bool = False,
    thinking_level: Optional[str] = None,
    node_label: str = "Nano Banana",
    request_log_enabled: bool = False,
    check_interrupt: Optional[Callable[[], None]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    if check_interrupt:
        check_interrupt()

    body = build_nano_banana_submit_body(
        model=model,
        prompt=prompt,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        images=images,
        enable_grounding=enable_grounding,
        thinking_level=thinking_level,
        request_log_enabled=request_log_enabled,
        node_label=node_label,
    )

    task_start = time.time()
    task_id = await _submit_task(
        session,
        base_url,
        api_key,
        body,
        node_label,
        log_body_enabled=request_log_enabled,
    )
    task_payload = await _poll_task(
        session,
        base_url,
        api_key,
        task_id,
        node_label,
        check_interrupt=check_interrupt,
        log_body_enabled=request_log_enabled,
        progress_callback=progress_callback,
    )
    task_done = time.time()

    parse_start = time.time()
    images_list = await _parse_task_images(task_payload, session, api_key)
    parse_done = time.time()

    return images_list, {
        "task_id": task_id,
        "task_ms": (task_done - task_start) * 1000,
        "parse_ms": (parse_done - parse_start) * 1000,
        "request_bytes": _json_size(body),
    }
