"""
Grok Video API client.

Flow:
  1. POST /v1/videos
  2. GET  /v1/videos/{task_id}
  3. GET  /v1/videos/{task_id}/content, or download a URL from the status body
"""

import asyncio
import base64
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from .base_client import BaseAPIClient
from ..utils.config import get_api_base_url, get_api_key_or_raise
from ..utils.http_error import RETRYABLE_STATUS_CODES, get_friendly_message
from ..utils.video_task import (
    check_interrupt,
    extract_error_message,
    extract_progress,
    extract_status,
    extract_video_url,
    interruptible_sleep,
    is_failure_status,
    is_success_status,
    run_with_interrupt,
)


class GrokVideoClient(BaseAPIClient):
    CREATE_ENDPOINT = "/v1/videos"
    STATUS_ENDPOINT = "/v1/videos/{task_id}"
    CONTENT_ENDPOINT = "/v1/videos/{task_id}/content"

    MODEL_OPTIONS = ["grok-imagine-video-1.5-preview", "grok-imagine-1.0-video"]
    ASPECT_RATIO_OPTIONS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
    QUALITY_OPTIONS = ["720p"]
    MODEL_SECONDS_OPTIONS = {
        "grok-imagine-1.0-video": [6, 10, 12, 16, 20],
    }
    QUALITY_API_MAP = {
        "720p": "high",
        "high": "high",
    }

    SUCCESS_STATUSES = {"complete", "completed", "succeed", "succeeded", "success", "done", "finished"}
    FAILURE_STATUSES = {"fail", "failed", "failure", "error", "expired", "timeout", "cancelled", "canceled"}

    def __init__(self, base_url: Optional[str] = None):
        api_key = get_api_key_or_raise("O1KEY_API_KEY")
        resolved_base_url = (base_url or "").strip() or get_api_base_url()
        super().__init__(base_url=resolved_base_url.rstrip("/"), api_key=api_key)

    def get_endpoint(self, **kwargs) -> str:
        return self.CREATE_ENDPOINT

    def build_request_body(self, **kwargs) -> Dict[str, Any]:
        return self.build_video_body(**kwargs)

    def parse_response(self, response: Dict[str, Any]) -> Any:
        return response

    @classmethod
    def build_video_body(
        cls,
        prompt: str,
        model: str,
        aspect_ratio: str,
        seconds: int,
        quality: str = "720p",
        images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("提示词不能为空。")

        if model not in cls.MODEL_OPTIONS:
            raise ValueError(f"模型仅支持: {', '.join(cls.MODEL_OPTIONS)}。")

        if aspect_ratio not in cls.ASPECT_RATIO_OPTIONS:
            raise ValueError(f"宽高比仅支持: {', '.join(cls.ASPECT_RATIO_OPTIONS)}。")

        try:
            seconds_value = int(seconds)
        except (TypeError, ValueError):
            raise ValueError("秒数必须是整数。") from None

        allowed_seconds = cls.MODEL_SECONDS_OPTIONS.get(model)
        if allowed_seconds is not None:
            if seconds_value not in allowed_seconds:
                raise ValueError(
                    f"模型 {model} 仅支持秒数: "
                    f"{', '.join(str(s) for s in allowed_seconds)}。"
                    "请修改为正确的秒数后再发起请求。"
                )
        elif seconds_value < 5 or seconds_value > 15:
            raise ValueError("秒数仅支持 5 到 15。")

        api_quality = cls.QUALITY_API_MAP.get(str(quality), str(quality))
        if api_quality != "high":
            raise ValueError("画质仅支持 720p。")

        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "seconds": str(seconds_value),
            "quality": api_quality,
        }

        image_list = [img for img in (images or []) if img]
        if image_list:
            body["images"] = image_list[:3]

        return body

    @staticmethod
    def _safe_task_filename(task_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("._")
        return safe or "grok_video"

    @staticmethod
    def _mask_body_for_log(body: Dict[str, Any]) -> Dict[str, Any]:
        log_body = dict(body)
        images = log_body.get("images")
        if isinstance(images, list):
            log_body["images"] = [f"<data-url chars={len(item)}>" for item in images]
        return log_body

    @staticmethod
    def _extract_task_id(payload: Dict[str, Any]) -> Optional[str]:
        sources = [payload]
        data = payload.get("data")
        if isinstance(data, dict):
            sources.append(data)

        for source in sources:
            for key in ("id", "task_id", "video_id"):
                value = source.get(key)
                if value:
                    return str(value)
        return None

    @staticmethod
    def _format_http_error(endpoint: str, status: int, error_text: str, task_id: Optional[str] = None) -> str:
        message = get_friendly_message(status, error_text)
        parts = [
            "Grok Video 请求失败。",
            f"endpoint: {endpoint}",
            f"http_status: {status}",
        ]
        if task_id:
            parts.append(f"task_id: {task_id}")
        if message:
            parts.append(f"message: {message}")
        return "\n".join(parts)

    @classmethod
    def _format_task_failure(cls, task_id: str, payload: Dict[str, Any]) -> str:
        return "\n".join(
            [
                "Grok Video 任务失败。",
                f"endpoint: {cls.STATUS_ENDPOINT.format(task_id=task_id)}",
                f"task_id: {task_id}",
                f"message: {extract_error_message(payload)}",
            ]
        )

    async def _request_json_with_retry(
        self,
        method: str,
        endpoint: str,
        session: aiohttp.ClientSession,
        task_id: Optional[str] = None,
        json_body: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        timeout_seconds: int = 120,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        headers = self.get_headers(use_bearer_token=True)
        timeout = aiohttp.ClientTimeout(total=timeout_seconds, connect=30, sock_read=timeout_seconds)

        last_status = 0
        last_text = ""

        for attempt in range(max_retries + 1):
            check_interrupt()
            response = None
            try:
                response = await run_with_interrupt(
                    session.request(method, url, json=json_body, headers=headers, timeout=timeout)
                )
                text = await run_with_interrupt(response.text())
                last_status = response.status
                last_text = text

                if 200 <= response.status < 300:
                    if not text.strip():
                        return {}
                    try:
                        return json.loads(text)
                    except Exception:
                        raise RuntimeError(f"Grok Video 响应 JSON 解析失败，原始内容：{text[:500]}") from None

                if response.status in RETRYABLE_STATUS_CODES and attempt < max_retries:
                    delay = min(2 ** attempt, 8)
                    print(
                        f"Grok Video：{get_friendly_message(response.status)} "
                        f"{delay}s 后重试 ({attempt + 1}/{max_retries})..."
                    )
                    await interruptible_sleep(delay)
                    continue

                break

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_retries:
                    delay = min(2 ** attempt, 8)
                    print(f"Grok Video：网络错误，{delay}s 后重试 ({attempt + 1}/{max_retries})...")
                    await interruptible_sleep(delay)
                    continue
                raise RuntimeError(f"Grok Video 网络错误: {e}") from None

            finally:
                if response is not None:
                    response.release()

        raise RuntimeError(self._format_http_error(endpoint, last_status, last_text, task_id=task_id))

    async def create_video_async(
        self,
        body: Dict[str, Any],
        session: aiohttp.ClientSession,
    ) -> Dict[str, Any]:
        print("Grok Video：正在提交任务...")
        return await self._request_json_with_retry(
            "POST",
            self.CREATE_ENDPOINT,
            session=session,
            json_body=body,
            timeout_seconds=180,
        )

    async def poll_video_status_async(
        self,
        task_id: str,
        session: aiohttp.ClientSession,
        poll_interval: int = 5,
        timeout: int = 900,
        progress_callback: Optional[Callable[[int, str, float], None]] = None,
    ) -> Dict[str, Any]:
        endpoint = self.STATUS_ENDPOINT.format(task_id=task_id)
        start = time.time()
        interval = max(1, int(poll_interval))

        await interruptible_sleep(interval)

        while True:
            data = await self._request_json_with_retry(
                "GET",
                endpoint,
                session=session,
                task_id=task_id,
                timeout_seconds=60,
            )

            status = extract_status(data)
            progress = extract_progress(data)
            elapsed = time.time() - start

            if progress_callback:
                progress_callback(progress, status, elapsed)

            if status in self.SUCCESS_STATUSES or is_success_status(status):
                return data

            if status in self.FAILURE_STATUSES or is_failure_status(status, data):
                raise RuntimeError(self._format_task_failure(task_id, data))

            if elapsed >= timeout:
                raise TimeoutError(
                    "Grok Video 任务轮询超时；任务未被标记为失败，可用 task_id 继续查询。\n"
                    f"endpoint: {endpoint}\n"
                    f"task_id: {task_id}\n"
                    f"status: {status or 'unknown'}\n"
                    f"timeout: {timeout}s"
                )

            await interruptible_sleep(min(interval, max(0.0, timeout - elapsed)))

    async def _download_url_to_file(
        self,
        url: str,
        save_path: str,
        session: aiohttp.ClientSession,
        max_retries: int = 3,
    ) -> str:
        timeout = aiohttp.ClientTimeout(total=900, connect=30, sock_read=900)
        last_status = 0
        last_text = ""
        headers = None
        resolved_url = url

        if url.startswith("data:"):
            if "," not in url:
                raise RuntimeError("Grok Video 下载失败：data URL 格式无效。")
            _, b64_data = url.split(",", 1)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(base64.b64decode(b64_data))
            if not os.path.isfile(save_path) or os.path.getsize(save_path) <= 0:
                raise RuntimeError("Grok Video 下载失败：保存后的文件为空。")
            return save_path

        if url.startswith("/"):
            resolved_url = f"{self.base_url}{url}"
            headers = self.get_headers(use_bearer_token=True)

        for attempt in range(max_retries + 1):
            check_interrupt()
            async with session.get(
                resolved_url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
            ) as response:
                if 200 <= response.status < 300:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(1024 * 1024):
                            check_interrupt()
                            if chunk:
                                f.write(chunk)

                    if not os.path.isfile(save_path) or os.path.getsize(save_path) <= 0:
                        raise RuntimeError("Grok Video 下载失败：保存后的文件为空。")
                    return save_path

                last_status = response.status
                last_text = await response.text()
                if response.status not in RETRYABLE_STATUS_CODES or attempt >= max_retries:
                    break

            delay = min(2 ** attempt, 8)
            print(f"Grok Video：下载重试 {attempt + 1}/{max_retries}，{delay}s 后继续...")
            await interruptible_sleep(delay)

        raise RuntimeError(self._format_http_error("download_url", last_status, last_text))

    async def download_video_async(
        self,
        task_id: str,
        save_path: str,
        session: aiohttp.ClientSession,
    ) -> str:
        endpoint = self.CONTENT_ENDPOINT.format(task_id=task_id)
        url = f"{self.base_url}{endpoint}"
        headers = self.get_headers(use_bearer_token=True)
        timeout = aiohttp.ClientTimeout(total=900, connect=30, sock_read=900)

        last_status = 0
        last_text = ""

        for attempt in range(4):
            check_interrupt()
            async with session.get(url, headers=headers, timeout=timeout, allow_redirects=True) as response:
                if 200 <= response.status < 300:
                    content_type = response.headers.get("Content-Type", "").lower()
                    if "application/json" in content_type:
                        data = await response.json(content_type=None)
                        download_url = extract_video_url(data)
                        if not download_url:
                            raise RuntimeError(
                                "Grok Video 下载失败：content 响应为 JSON，但未包含视频 URL。\n"
                                f"endpoint: {endpoint}\n"
                                f"task_id: {task_id}"
                            )
                        return await self._download_url_to_file(download_url, save_path, session)

                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(1024 * 1024):
                            check_interrupt()
                            if chunk:
                                f.write(chunk)

                    if not os.path.isfile(save_path) or os.path.getsize(save_path) <= 0:
                        raise RuntimeError(
                            "Grok Video 下载失败：保存后的文件为空。\n"
                            f"endpoint: {endpoint}\n"
                            f"task_id: {task_id}"
                        )
                    return save_path

                last_status = response.status
                last_text = await response.text()
                if response.status not in RETRYABLE_STATUS_CODES or attempt >= 3:
                    break

            delay = min(2 ** attempt, 8)
            print(f"Grok Video：content 下载重试 {attempt + 1}/3，{delay}s 后继续...")
            await interruptible_sleep(delay)

        raise RuntimeError(self._format_http_error(endpoint, last_status, last_text, task_id=task_id))

    def generate_video_sync(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str,
        seconds: int,
        quality: str,
        images: Optional[List[str]],
        output_dir: Optional[str] = None,
        save_path: Optional[str] = None,
        poll_interval: int = 5,
        timeout: int = 900,
        progress_callback: Optional[Callable[[int, str, float], None]] = None,
    ) -> Dict[str, Any]:
        async def _run():
            async with self._make_session() as session:
                body = self.build_video_body(
                    prompt=prompt,
                    model=model,
                    aspect_ratio=aspect_ratio,
                    seconds=seconds,
                    quality=quality,
                    images=images,
                )

                create_response = await self.create_video_async(body, session)
                task_id = self._extract_task_id(create_response) or ""
                if not task_id:
                    raise RuntimeError(
                        "Grok Video 未返回任务 ID。\n"
                        f"endpoint: {self.CREATE_ENDPOINT}\n"
                        f"response: {json.dumps(create_response, ensure_ascii=False)[:1200]}"
                    )

                print(f"Grok Video：任务已提交，任务ID：{task_id}")
                print("Grok Video：视频生成中...")
                status_response = await self.poll_video_status_async(
                    task_id=task_id,
                    session=session,
                    poll_interval=poll_interval,
                    timeout=timeout,
                    progress_callback=progress_callback,
                )

                video_url = extract_video_url(status_response)
                print("Grok Video：视频生成完成，正在下载...")
                if save_path is None:
                    resolved_output_dir = output_dir or os.getcwd()
                    os.makedirs(resolved_output_dir, exist_ok=True)
                    target_path = os.path.join(
                        resolved_output_dir,
                        f"{self._safe_task_filename(task_id)}.mp4",
                    )
                else:
                    target_path = save_path

                if video_url:
                    video_path = await self._download_url_to_file(video_url, target_path, session)
                else:
                    video_path = await self.download_video_async(task_id, target_path, session)

                return {
                    "task_id": task_id,
                    "status": extract_status(status_response),
                    "video_path": video_path,
                    "raw_json": {
                        "create": create_response,
                        "status": status_response,
                    },
                }

        return self.run_async_in_thread(_run())
