"""
Kling 视频生成 API 客户端
提供图生视频创建、状态轮询、视频下载功能
"""

import asyncio
import base64
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from .base_client import BaseAPIClient
from ..utils.config import get_api_key_or_raise, get_api_base_url


class KlingClient(BaseAPIClient):
    """
    Kling 图生视频生成客户端

    工作流程：
    1. create_video    → POST /kling/v1/videos/image2video (提交生成任务)
    2. poll_status    → GET  /kling/v1/videos/{id}  (轮询直到完成/失败)
    3. download_video → GET  /kling/v1/videos/{id}/content (下载视频文件)
    """

    CREATE_ENDPOINT = "/kling/v1/videos/image2video"
    STATUS_ENDPOINT = "/kling/v1/videos/{video_id}"
    CONTENT_ENDPOINT = "/kling/v1/videos/{video_id}/content"

    POLL_INITIAL_INTERVAL = 3
    POLL_MAX_INTERVAL = 15

    def __init__(self):
        api_key = get_api_key_or_raise()
        base_url = get_api_base_url()
        super().__init__(base_url=base_url, api_key=api_key)

    # ------------------------------------------------------------------
    # BaseAPIClient 抽象方法实现
    # ------------------------------------------------------------------

    def get_endpoint(self, **kwargs) -> str:
        return self.CREATE_ENDPOINT

    def build_request_body(self, **kwargs) -> Dict[str, Any]:
        return {}

    def parse_response(self, response: Dict[str, Any]) -> Any:
        return response

    # ------------------------------------------------------------------
    # 核心异步方法
    # ------------------------------------------------------------------

    async def create_video_async(
        self,
        image: str,
        prompt: str = "",
        negative_prompt: str = "",
        model_name: str = "kling-v2-6",
        duration: str = "5",
        mode: str = "pro",
        image_tail: Optional[str] = None,
        sound: str = "off",
        callback_url: str = "",
        external_task_id: str = "",
        cfg_scale: Optional[float] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Dict[str, Any]:
        """
        提交图生视频生成任务

        Args:
            image: 参考图片，支持 Base64 编码或图片 URL
            prompt: 正向文本提示
            negative_prompt: 负向文本提示
            model_name: 模型名称 (kling-v1-5, kling-v1-6, kling-v2-6)
            duration: 视频时长，枚举值 "5" 或 "10"
            mode: 生成模式，"std"(高性能) 或 "pro"(高表现)
            image_tail: 参考图片 - 尾帧控制（可选）
            sound: 声音开关，"off" 或其他
            callback_url: 回调通知地址（可选）
            external_task_id: 外部任务ID（可选）
            cfg_scale: 生成视频的自由度，0-1 之间
            session: aiohttp 会话

        Returns:
            API 响应 JSON，包含 task id 和初始状态
        """
        url = f"{self.base_url}{self.CREATE_ENDPOINT}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # 构建请求体
        body: Dict[str, Any] = {
            "model_name": model_name,
            "image": image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "duration": duration,
            "mode": mode,
            "sound": sound,
            "callback_url": callback_url,
            "external_task_id": external_task_id,
        }

        # 添加可选参数
        if image_tail:
            body["image_tail"] = image_tail
        if cfg_scale is not None:
            body["cfg_scale"] = cfg_scale

        # 过滤空值
        body = {k: v for k, v in body.items() if v}

        # 检查图片大小（Base64 情况）
        if len(image) > self.max_request_size:
            raise ValueError(
                f"参考图片约 {len(image) / 1024 / 1024:.1f}MB，"
                f"超过 {self.max_request_size / 1024 / 1024:.0f}MB 限制，请使用较小的图片"
            )

        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            async with session.post(url, json=body, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    error_message = self._extract_error_message(error_text, response.status)
                    raise RuntimeError(error_message)

                resp_json = await response.json()
                return resp_json

        finally:
            if close_session:
                await session.close()

    async def poll_video_status_async(
        self,
        video_id: str,
        progress_callback: Optional[Callable[[int, float], None]] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Dict[str, Any]:
        """
        轮询视频生成状态，直到完成或失败

        Args:
            video_id: 视频任务 ID
            progress_callback: 进度回调 (progress_percent, elapsed_seconds)
            session: aiohttp 会话

        Returns:
            最终状态的 API 响应

        Raises:
            RuntimeError: 生成失败
        """
        url = f"{self.base_url}{self.STATUS_ENDPOINT.format(video_id=video_id)}"
        headers = self.get_headers(use_bearer_token=True)

        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        interval = self.POLL_INITIAL_INTERVAL

        try:
            while True:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        error_message = self._extract_error_message(error_text, response.status)
                        raise RuntimeError(error_message)

                    data = await response.json()

                # status 兼容大小写
                status = data.get("status", "").lower()

                # progress 兼容
                progress_raw = data.get("progress", 0)
                if isinstance(progress_raw, str):
                    try:
                        progress = int(progress_raw.rstrip("%").strip())
                    except ValueError:
                        progress = 0
                else:
                    progress = int(progress_raw) if progress_raw else 0

                if progress_callback:
                    progress_callback(progress)

                if status == "completed":
                    return data

                if status == "failed":
                    error_info = data.get("error", {})
                    error_msg = error_info.get("message", "未知错误") if isinstance(error_info, dict) else str(error_info)
                    raise RuntimeError(f"视频生成失败: {error_msg}")

                await asyncio.sleep(interval)
                interval = min(interval * 1.5, self.POLL_MAX_INTERVAL)

        finally:
            if close_session:
                await session.close()

    async def download_video_async(
        self,
        video_id: str,
        save_path: str,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """
        下载生成的视频文件

        Args:
            video_id: 视频任务 ID
            save_path: 保存路径
            session: aiohttp 会话

        Returns:
            保存的文件路径
        """
        url = f"{self.base_url}{self.CONTENT_ENDPOINT.format(video_id=video_id)}"
        headers = self.get_headers(use_bearer_token=True)

        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            async with session.get(url, headers=headers, allow_redirects=True) as response:
                if response.status != 200:
                    error_text = await response.text()
                    error_message = self._extract_error_message(error_text, response.status)
                    raise RuntimeError(f"视频下载失败: {error_message}")

                content_type = response.headers.get("Content-Type", "")

                if "application/json" in content_type:
                    data = await response.json()
                    download_url = data.get("url") or data.get("download_url")
                    if not download_url:
                        raise RuntimeError("视频下载失败: 响应中未找到下载链接")
                    await self._download_from_url(download_url, save_path, session)
                else:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)

            return save_path

        finally:
            if close_session:
                await session.close()

    # ------------------------------------------------------------------
    # 同步包装
    # ------------------------------------------------------------------

    def generate_video_sync(
        self,
        image: str,
        prompt: str = "",
        negative_prompt: str = "",
        model_name: str = "kling-v2-6",
        duration: str = "5",
        mode: str = "pro",
        image_tail: Optional[str] = None,
        sound: str = "off",
        callback_url: str = "",
        external_task_id: str = "",
        cfg_scale: Optional[float] = None,
        save_path: str = "",
        progress_callback: Optional[Callable[[int, float], None]] = None,
        on_stage: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        同步执行完整的视频生成流程（创建 → 轮询 → 下载）

        Returns:
            保存的视频文件路径
        """

        async def _run():
            connector = aiohttp.TCPConnector(limit=0)
            async with aiohttp.ClientSession(connector=connector) as session:
                # 1. 提交任务
                if on_stage:
                    on_stage("submitting")
                result = await self.create_video_async(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    model_name=model_name,
                    duration=duration,
                    mode=mode,
                    image_tail=image_tail,
                    sound=sound,
                    callback_url=callback_url,
                    external_task_id=external_task_id,
                    cfg_scale=cfg_scale,
                    session=session,
                )
                task_id = result.get("task_id") or result.get("id") or result.get("task_id")
                if not task_id:
                    raise RuntimeError("API 未返回视频任务 ID")

                if on_stage:
                    on_stage(f"submitted:{task_id}")

                # 2. 轮询状态
                if on_stage:
                    on_stage("polling")
                await self.poll_video_status_async(
                    video_id=task_id,
                    progress_callback=progress_callback,
                    session=session,
                )

                # 3. 下载视频
                if on_stage:
                    on_stage("downloading")
                path = await self.download_video_async(
                    video_id=task_id,
                    save_path=save_path,
                    session=session,
                )

                if on_stage:
                    on_stage("done")
                return path

        return self.run_async_in_thread(_run())

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    async def _download_from_url(
        self,
        url: str,
        save_path: str,
        session: aiohttp.ClientSession,
    ) -> None:
        """从给定 URL 下载文件到本地路径"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        async with session.get(url) as response:
            if response.status != 200:
                raise RuntimeError(f"从下载链接获取视频失败 (状态码: {response.status})")
            with open(save_path, "wb") as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)

    @staticmethod
    def _extract_error_message(error_text: str, status_code: int) -> str:
        """从错误响应中提取可读的错误信息"""
        error_message = error_text
        try:
            error_json = json.loads(error_text)
            if "error" in error_json:
                if isinstance(error_json["error"], dict):
                    error_message = error_json["error"].get("message", error_text)
                else:
                    error_message = str(error_json["error"])
            elif "message" in error_json:
                error_message = error_json["message"]
        except (json.JSONDecodeError, KeyError):
            pass

        status_hints = {
            400: "请求参数错误 (400)",
            401: "认证失败 (401)，请检查 API 密钥",
            403: "权限不足 (403)，请检查账户权限或余额",
            429: "请求频率超限 (429)，请稍后重试",
            503: "服务暂时不可用 (503)，请稍后重试",
            504: "请求超时 (504)，请稍后重试",
        }
        hint = status_hints.get(status_code, f"API 请求失败 (状态码: {status_code})")
        return f"{hint}\nAPI 返回: {error_message}"
