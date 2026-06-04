"""
new-api Veo 3.1 video client.

Implements the OpenAI-compatible /v1/videos task flow:
submit, poll, and stream-download video content.
"""

import asyncio
import json
import os
import re
import time
from typing import Any, Callable, Dict, Optional

import aiohttp

from .base_client import BaseAPIClient
from ..utils.config import get_api_base_url, get_api_key_or_raise


class NewAPIVeoClient(BaseAPIClient):
    CREATE_ENDPOINT = "/v1/videos"
    STATUS_ENDPOINT = "/v1/videos/{task_id}"
    CONTENT_ENDPOINT = "/v1/videos/{task_id}/content"

    RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
    COMPLETED_STATUSES = {"completed", "succeeded", "success", "done"}
    FAILED_STATUSES = {"failed", "error", "cancelled", "canceled"}

    def __init__(
        self,
        base_url: Optional[str] = None,
    ):
        api_key = get_api_key_or_raise("O1KEY_API_KEY")
        resolved_base_url = (base_url or "").strip() or get_api_base_url()
        super().__init__(base_url=resolved_base_url.rstrip("/"), api_key=api_key)

    def get_endpoint(self, **kwargs) -> str:
        return self.CREATE_ENDPOINT

    def build_request_body(self, **kwargs) -> Dict[str, Any]:
        return self._build_video_body(**kwargs)

    def parse_response(self, response: Dict[str, Any]) -> Any:
        return response

    @staticmethod
    def _build_video_body(
        prompt: str,
        model: str,
        duration: int,
        aspect_ratio: str,
        resolution: str,
        negative_prompt: str = "",
        generate_audio: bool = True,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "aspectRatio": aspect_ratio,
            "resolution": resolution,
            "generateAudio": bool(generate_audio),
        }

        negative_prompt = (negative_prompt or "").strip()
        if negative_prompt:
            metadata["negativePrompt"] = negative_prompt

        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "duration": int(duration),
            "metadata": metadata,
        }

        return body

    @staticmethod
    def _print_request_body(body: Dict[str, Any], image_bytes: Optional[bytes] = None) -> None:
        log_body = dict(body)
        if image_bytes is not None:
            log_body["input_reference"] = f"<PNG bytes: {len(image_bytes)}>"
        print(
            "NewAPI Veo request body:\n"
            f"{json.dumps(log_body, ensure_ascii=False, indent=2)}"
        )

    @staticmethod
    def _safe_task_filename(task_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("._")
        return safe or "newapi_veo"

    @staticmethod
    def _extract_task_id(data: Dict[str, Any]) -> Optional[str]:
        for key in ("id", "task_id", "video_id"):
            value = data.get(key)
            if value:
                return str(value)

        nested = data.get("data")
        if isinstance(nested, dict):
            for key in ("id", "task_id", "video_id"):
                value = nested.get(key)
                if value:
                    return str(value)
        return None

    @staticmethod
    def _extract_status(data: Dict[str, Any]) -> str:
        for key in ("status", "state", "task_status"):
            value = data.get(key)
            if value:
                return str(value).lower()

        nested = data.get("data")
        if isinstance(nested, dict):
            for key in ("status", "state", "task_status"):
                value = nested.get(key)
                if value:
                    return str(value).lower()
        return "unknown"

    @staticmethod
    def _extract_progress(data: Dict[str, Any]) -> int:
        progress = data.get("progress")
        if progress is None and isinstance(data.get("data"), dict):
            progress = data["data"].get("progress")

        if isinstance(progress, str):
            progress = progress.rstrip("%").strip()
            try:
                return int(float(progress))
            except ValueError:
                return 0
        if isinstance(progress, (int, float)):
            return int(progress)
        return 0

    @classmethod
    def _format_http_error(
        cls,
        endpoint: str,
        status: int,
        error_text: str,
        task_id: Optional[str] = None,
    ) -> str:
        code = ""
        message = error_text
        try:
            payload = json.loads(error_text)
            error = payload.get("error", payload)
            if isinstance(error, dict):
                code = str(error.get("code") or error.get("type") or "")
                message = str(error.get("message") or payload.get("message") or error_text)
            elif error is not None:
                message = str(error)
        except Exception:
            pass

        message = (message or "").strip()
        if len(message) > 1200:
            message = message[:1200] + "...(truncated)"

        if status in (401, 403):
            hint = "凭证或分组权限问题，请检查 new-api token、模型分组或渠道权限。"
        elif status == 429:
            hint = "频率或额度限制，请稍后重试或检查 new-api 额度。"
        elif status in (502, 503, 504):
            hint = "上游服务暂时不可用或超时，请稍后用 task_id 继续查询。"
        elif status == 400:
            hint = "请求参数错误，请检查 model、duration、metadata 和图片输入。"
        else:
            hint = "new-api 视频请求失败。"

        parts = [
            hint,
            f"endpoint: {endpoint}",
            f"http_status: {status}",
        ]
        if task_id:
            parts.append(f"task_id: {task_id}")
        if code:
            parts.append(f"error_code: {code}")
        if message:
            parts.append(f"message: {message}")
        return "\n".join(parts)

    @classmethod
    def _format_task_failure(cls, task_id: str, data: Dict[str, Any]) -> str:
        error = data.get("error")
        if error is None and isinstance(data.get("data"), dict):
            error = data["data"].get("error")

        if isinstance(error, dict):
            code = error.get("code") or error.get("type") or ""
            message = error.get("message") or json.dumps(error, ensure_ascii=False)
        else:
            code = ""
            message = str(error or "未知错误")

        return "\n".join(
            [
                "Veo 视频任务失败。",
                f"endpoint: {cls.STATUS_ENDPOINT.format(task_id=task_id)}",
                f"task_id: {task_id}",
                f"error_code: {code}",
                f"message: {message}",
            ]
        )

    async def create_video_async(
        self,
        prompt: str,
        model: str,
        duration: int,
        aspect_ratio: str,
        resolution: str,
        negative_prompt: str = "",
        generate_audio: bool = True,
        image_bytes: Optional[bytes] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Dict[str, Any]:
        body = self._build_video_body(
            prompt=prompt,
            model=model,
            duration=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            negative_prompt=negative_prompt,
            generate_audio=generate_audio,
        )

        url = f"{self.base_url}{self.CREATE_ENDPOINT}"
        close_session = False
        if session is None:
            session = self._make_session()
            close_session = True

        try:
            timeout = aiohttp.ClientTimeout(total=120, connect=30, sock_read=120)
            headers = {"Authorization": f"Bearer {self.api_key}"}

            if image_bytes is not None:
                if len(image_bytes) > self.max_request_size:
                    raise ValueError(
                        f"输入图片过大，超过 {self.max_request_size / 1024 / 1024:.0f}MB 限制"
                    )

                self._print_request_body(body, image_bytes=image_bytes)
                form = aiohttp.FormData()
                form.add_field("model", body["model"])
                form.add_field("prompt", body["prompt"])
                form.add_field("duration", str(body["duration"]))
                form.add_field("metadata", json.dumps(body["metadata"], ensure_ascii=False))
                form.add_field(
                    "input_reference",
                    image_bytes,
                    filename="input_reference.png",
                    content_type="image/png",
                )
                request_kwargs = {"data": form, "headers": headers}
                print(
                    "NewAPI Veo: POST /v1/videos multipart "
                    f"| model={model} | duration={duration}s | {resolution} {aspect_ratio}"
                )
            else:
                self._print_request_body(body)
                headers["Content-Type"] = "application/json"
                request_kwargs = {"json": body, "headers": headers}
                print(
                    "NewAPI Veo: POST /v1/videos json "
                    f"| model={model} | duration={duration}s | {resolution} {aspect_ratio}"
                )

            async with session.post(url, timeout=timeout, **request_kwargs) as response:
                if response.status >= 300:
                    error_text = await response.text()
                    raise RuntimeError(
                        self._format_http_error(
                            self.CREATE_ENDPOINT,
                            response.status,
                            error_text,
                        )
                    )
                return await response.json()
        finally:
            if close_session:
                await session.close()

    async def _get_json_with_retry(
        self,
        endpoint: str,
        session: aiohttp.ClientSession,
        task_id: Optional[str] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        headers = self.get_headers(use_bearer_token=True)
        timeout = aiohttp.ClientTimeout(total=60, connect=30, sock_read=60)

        last_error = ""
        last_status = 0
        for attempt in range(max_retries + 1):
            async with session.get(url, headers=headers, timeout=timeout) as response:
                if response.status < 300:
                    return await response.json()

                last_status = response.status
                last_error = await response.text()
                if response.status not in self.RETRYABLE_STATUS_CODES or attempt >= max_retries:
                    break

            await asyncio.sleep(min(2 ** attempt, 8))

        raise RuntimeError(
            self._format_http_error(endpoint, last_status, last_error, task_id=task_id)
        )

    async def poll_video_status_async(
        self,
        task_id: str,
        poll_interval: int = 5,
        timeout: int = 900,
        progress_callback: Optional[Callable[[int, str, float], None]] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Dict[str, Any]:
        endpoint = self.STATUS_ENDPOINT.format(task_id=task_id)
        close_session = False
        if session is None:
            session = self._make_session()
            close_session = True

        start = time.time()
        try:
            poll_interval = max(1, int(poll_interval))
            await asyncio.sleep(poll_interval)

            while True:
                data = await self._get_json_with_retry(endpoint, session, task_id=task_id)
                status = self._extract_status(data)
                elapsed = time.time() - start
                progress = self._extract_progress(data)

                if status == "unknown":
                    print(
                        "NewAPI Veo status response did not include a recognized status field:\n"
                        f"{json.dumps(data, ensure_ascii=False, indent=2)[:1200]}"
                    )

                if progress_callback:
                    progress_callback(progress, status, elapsed)

                if status in self.COMPLETED_STATUSES:
                    return data

                if status in self.FAILED_STATUSES:
                    raise RuntimeError(self._format_task_failure(task_id, data))

                if elapsed >= timeout:
                    raise TimeoutError(
                        "Veo 视频任务轮询超时；任务未被标记为失败，可用 task_id 继续查询。\n"
                        f"endpoint: {endpoint}\n"
                        f"task_id: {task_id}\n"
                        f"status: {status}\n"
                        f"timeout: {timeout}s"
                    )

                remaining = max(0.0, timeout - elapsed)
                await asyncio.sleep(min(poll_interval, remaining))
        finally:
            if close_session:
                await session.close()

    async def _download_url_to_file(
        self,
        url: str,
        save_path: str,
        session: aiohttp.ClientSession,
        max_retries: int = 3,
    ) -> None:
        timeout = aiohttp.ClientTimeout(total=900, connect=30, sock_read=900)
        last_status = 0
        last_error = ""

        for attempt in range(max_retries + 1):
            async with session.get(url, timeout=timeout, allow_redirects=True) as response:
                if response.status < 300:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    return

                last_status = response.status
                last_error = await response.text()
                if response.status not in self.RETRYABLE_STATUS_CODES or attempt >= max_retries:
                    break

            await asyncio.sleep(min(2 ** attempt, 8))

        raise RuntimeError(
            self._format_http_error("download_url", last_status, last_error)
        )

    async def download_video_async(
        self,
        task_id: str,
        save_path: str,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        endpoint = self.CONTENT_ENDPOINT.format(task_id=task_id)
        url = f"{self.base_url}{endpoint}"
        headers = self.get_headers(use_bearer_token=True)
        timeout = aiohttp.ClientTimeout(total=900, connect=30, sock_read=900)

        close_session = False
        if session is None:
            session = self._make_session()
            close_session = True

        try:
            last_status = 0
            last_error = ""
            for attempt in range(4):
                async with session.get(url, headers=headers, timeout=timeout, allow_redirects=True) as response:
                    if response.status < 300:
                        content_type = response.headers.get("Content-Type", "")
                        if "application/json" in content_type.lower():
                            data = await response.json()
                            nested = data.get("data") if isinstance(data.get("data"), dict) else {}
                            download_url = (
                                data.get("url")
                                or data.get("download_url")
                                or nested.get("url")
                                or nested.get("download_url")
                            )
                            if not download_url:
                                raise RuntimeError(
                                    "视频下载失败: content 响应为 JSON，但未包含 url/download_url。\n"
                                    f"endpoint: {endpoint}\n"
                                    f"task_id: {task_id}"
                                )
                            await self._download_url_to_file(download_url, save_path, session)
                        else:
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            with open(save_path, "wb") as f:
                                async for chunk in response.content.iter_chunked(1024 * 1024):
                                    if chunk:
                                        f.write(chunk)

                        if not os.path.isfile(save_path) or os.path.getsize(save_path) <= 0:
                            raise RuntimeError(
                                "视频下载失败: 保存后的文件为空。\n"
                                f"endpoint: {endpoint}\n"
                                f"task_id: {task_id}"
                            )
                        return save_path

                    last_status = response.status
                    last_error = await response.text()
                    if response.status not in self.RETRYABLE_STATUS_CODES or attempt >= 3:
                        break

                await asyncio.sleep(min(2 ** attempt, 8))

            raise RuntimeError(
                self._format_http_error(endpoint, last_status, last_error, task_id=task_id)
            )
        finally:
            if close_session:
                await session.close()

    def generate_video_sync(
        self,
        prompt: str,
        model: str,
        duration: int,
        aspect_ratio: str,
        resolution: str,
        output_dir: str,
        negative_prompt: str = "",
        generate_audio: bool = True,
        image_bytes: Optional[bytes] = None,
        poll_interval: int = 5,
        timeout: int = 900,
        reuse_task_id: str = "",
        progress_callback: Optional[Callable[[int, str, float], None]] = None,
    ) -> Dict[str, Any]:
        async def _run():
            async with self._make_session() as session:
                create_response: Dict[str, Any] = {}
                task_id = (reuse_task_id or "").strip()
                if task_id:
                    print(f"NewAPI Veo: reuse task_id={task_id}")
                else:
                    create_response = await self.create_video_async(
                        prompt=prompt,
                        model=model,
                        duration=duration,
                        aspect_ratio=aspect_ratio,
                        resolution=resolution,
                        negative_prompt=negative_prompt,
                        generate_audio=generate_audio,
                        image_bytes=image_bytes,
                        session=session,
                    )
                    task_id = self._extract_task_id(create_response) or ""
                    if not task_id:
                        raise RuntimeError(
                            "new-api 未返回视频任务 ID。\n"
                            f"endpoint: {self.CREATE_ENDPOINT}\n"
                            f"response: {json.dumps(create_response, ensure_ascii=False)[:1200]}"
                        )

                status_response = await self.poll_video_status_async(
                    task_id=task_id,
                    poll_interval=poll_interval,
                    timeout=timeout,
                    progress_callback=progress_callback,
                    session=session,
                )
                status = self._extract_status(status_response)

                os.makedirs(output_dir, exist_ok=True)
                filename = f"{self._safe_task_filename(task_id)}.mp4"
                save_path = os.path.join(output_dir, filename)
                video_path = await self.download_video_async(
                    task_id=task_id,
                    save_path=save_path,
                    session=session,
                )

                return {
                    "task_id": task_id,
                    "status": status,
                    "video_path": video_path,
                    "raw_json": {
                        "create": create_response,
                        "status": status_response,
                    },
                }

        return self.run_async_in_thread(_run())
