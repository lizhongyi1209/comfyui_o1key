"""
Sora 视频生成 API 客户端
提供视频创建、状态轮询、视频下载功能
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
from ..utils.image_utils import encode_image_to_base64


class SoraClient(BaseAPIClient):
    """
    Sora 视频生成客户端

    工作流程：
    1. create_video  → POST /v1/videos  (提交生成任务)
    2. poll_status   → GET  /v1/videos/{id}  (轮询直到完成/失败)
    3. download_video→ GET  /v1/videos/{id}/content (下载视频文件)
    """

    CREATE_ENDPOINT = "/v1/videos"
    STATUS_ENDPOINT = "/v1/videos/{video_id}"
    CONTENT_ENDPOINT = "/v1/videos/{video_id}/content"

    POLL_INITIAL_INTERVAL = 3
    POLL_MAX_INTERVAL = 15
    POLL_TIMEOUT = 600  # 10 分钟

    def __init__(self):
        api_key = get_api_key_or_raise()
        base_url = get_api_base_url()
        super().__init__(base_url=base_url, api_key=api_key)

    # ------------------------------------------------------------------
    # BaseAPIClient 抽象方法实现（本客户端主要使用自定义方法）
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
        prompt: str,
        model: str,
        seconds: int = 4,
        size: str = "720x1280",
        input_reference_bytes: Optional[bytes] = None,
        seed: Optional[int] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Dict[str, Any]:
        """
        提交视频生成任务

        格式策略（根据抓包确认）：
        - 无参考图片：application/json
        - 有参考图片：multipart/form-data，input_reference 以 PNG 文件上传

        注意：seed 不被上游 API 接受，仅在 ComfyUI 节点侧用于缓存刷新

        Returns:
            API 响应 JSON，包含 video id 和初始状态
        """
        url = f"{self.base_url}{self.CREATE_ENDPOINT}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # ============================================================
        # ⚠️  已验证可用的标准请求方案，请勿随意修改！（2026-02-28）
        # ============================================================
        # 经多轮调试确认：
        #   - 有图片：必须使用 multipart/form-data，input_reference 以 PNG 文件上传
        #     · filename="reference.png", content_type="image/png"（与抓包一致）
        #     · 不可改为 application/json + base64 → 400 "expected a file, got a string"
        #     · 不可改为 application/json + data URI → 500 upstream error
        #     · 不可改为 multipart + image/jpeg     → 400 "Inpaint image must match..."（尺寸校验失败）
        #   - 无图片：使用 application/json，已验证成功
        # ============================================================
        if input_reference_bytes:
            if len(input_reference_bytes) > self.max_request_size:
                raise ValueError(
                    f"参考图片约 {len(input_reference_bytes) / 1024 / 1024:.1f}MB，"
                    f"超过 {self.max_request_size / 1024 / 1024:.0f}MB 限制，请使用较小的图片"
                )
            # ⚠️ 有图片：multipart/form-data + PNG 文件上传（唯一验证成功的方案）
            form = aiohttp.FormData()
            form.add_field("prompt", prompt)
            form.add_field("model", model)
            form.add_field("seconds", str(seconds))
            form.add_field("size", size)
            form.add_field(
                "input_reference",
                input_reference_bytes,
                filename="reference.png",   # ⚠️ 不可改文件名/扩展名
                content_type="image/png",   # ⚠️ 不可改为 image/jpeg
            )
            send_kwargs: Dict[str, Any] = {"data": form, "headers": headers}
        else:
            # ⚠️ 无图片：application/json（已验证成功）
            body: Dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "seconds": str(seconds),
                "size": size,
            }
            send_kwargs = {"json": body, "headers": headers}

        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            async with session.post(url, **send_kwargs) as response:
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
            RuntimeError: 生成失败或超时
        """
        url = f"{self.base_url}{self.STATUS_ENDPOINT.format(video_id=video_id)}"
        headers = self.get_headers(use_bearer_token=True)

        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        interval = self.POLL_INITIAL_INTERVAL
        start_time = time.time()

        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed > self.POLL_TIMEOUT:
                    raise RuntimeError(
                        f"视频生成超时（已等待 {int(elapsed)}s，上限 {self.POLL_TIMEOUT}s）"
                    )

                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        error_message = self._extract_error_message(error_text, response.status)
                        raise RuntimeError(error_message)

                    data = await response.json()

                # status 兼容大小写：queued / in_progress / IN_PROGRESS / completed / COMPLETED
                status = data.get("status", "").lower()

                # progress 兼容整数 (30) 和字符串 ("30%") 两种格式
                progress_raw = data.get("progress", 0)
                if isinstance(progress_raw, str):
                    try:
                        progress = int(progress_raw.rstrip("%").strip())
                    except ValueError:
                        progress = 0
                else:
                    progress = int(progress_raw) if progress_raw else 0

                if progress_callback:
                    progress_callback(progress, elapsed)

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

        处理两种情况：
        1. 响应为重定向或 JSON 含下载 URL → 跟随下载
        2. 响应为二进制视频流 → 直接保存

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
        prompt: str,
        model: str,
        seconds: int,
        size: str,
        save_path: str,
        input_reference_bytes: Optional[bytes] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, float], None]] = None,
        on_stage: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        同步执行完整的视频生成流程（创建 → 轮询 → 下载）

        Args:
            on_stage: 阶段回调，用于打印状态切换信息

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
                    prompt=prompt,
                    model=model,
                    seconds=seconds,
                    size=size,
                    input_reference_bytes=input_reference_bytes,
                    seed=seed,
                    session=session,
                )
                video_id = result.get("id")
                if not video_id:
                    raise RuntimeError("API 未返回视频任务 ID")

                if on_stage:
                    on_stage(f"submitted:{video_id}")

                # 2. 轮询状态
                if on_stage:
                    on_stage("polling")
                await self.poll_video_status_async(
                    video_id=video_id,
                    progress_callback=progress_callback,
                    session=session,
                )

                # 3. 下载视频
                if on_stage:
                    on_stage("downloading")
                path = await self.download_video_async(
                    video_id=video_id,
                    save_path=save_path,
                    session=session,
                )

                if on_stage:
                    on_stage("done")
                return path

        return self.run_async_in_thread(_run())

    async def _generate_one_video_async(
        self,
        prompt: str,
        model: str,
        seconds: int,
        size: str,
        save_path: str,
        input_reference_bytes: Optional[bytes] = None,
        seed: Optional[int] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> str:
        """
        异步生成单个视频（创建 → 轮询 → 下载）

        Returns:
            保存的视频文件路径
        """
        result = await self.create_video_async(
            prompt=prompt,
            model=model,
            seconds=seconds,
            size=size,
            input_reference_bytes=input_reference_bytes,
            seed=seed,
            session=session,
        )
        video_id = result.get("id")
        if not video_id:
            raise RuntimeError("API 未返回视频任务 ID")

        await self.poll_video_status_async(video_id=video_id, session=session)
        path = await self.download_video_async(
            video_id=video_id, save_path=save_path, session=session
        )
        return path

    async def generate_batch_videos_async(
        self,
        prompt: str,
        model: str,
        seconds: int,
        size: str,
        save_paths: List[str],
        input_reference_bytes: Optional[bytes] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, bool, Optional[str]], None]] = None,
    ) -> List[str]:
        """
        并发生成多个视频

        Args:
            prompt: 提示词
            model: 模型名称
            seconds: 视频时长（秒）
            size: 分辨率
            save_paths: 各视频的保存路径列表，长度决定并发数量
            input_reference_bytes: 参考图片字节（可选）
            seed: 随机种子（仅节点侧使用）
            progress_callback: 进度回调 (current, total, success, error_msg)

        Returns:
            成功生成的视频路径列表
        """
        batch_size = len(save_paths)
        connector = aiohttp.TCPConnector(limit=0)

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._generate_one_video_async(
                    prompt=prompt,
                    model=model,
                    seconds=seconds,
                    size=size,
                    save_path=save_paths[i],
                    input_reference_bytes=input_reference_bytes,
                    seed=seed,
                    session=session,
                )
                for i in range(batch_size)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        completed = 0
        paths: List[str] = []
        first_error = None
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = str(result).split("\n")[0]
                print(f"Sora: 第 {i + 1} 个视频生成失败: {error_msg}")
                if first_error is None:
                    first_error = result
                if progress_callback:
                    progress_callback(i + 1, batch_size, False, error_msg)
            else:
                completed += 1
                paths.append(result)
                if progress_callback:
                    progress_callback(completed, batch_size, True, None)

        if not paths:
            if first_error:
                raise first_error
            raise RuntimeError(f"批量视频生成失败，{batch_size} 个任务全部失败")

        return paths

    def generate_batch_videos_sync(
        self,
        prompt: str,
        model: str,
        seconds: int,
        size: str,
        save_paths: List[str],
        input_reference_bytes: Optional[bytes] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, bool, Optional[str]], None]] = None,
    ) -> List[str]:
        """
        同步并发生成多个视频（用于 ComfyUI 节点）

        Args:
            save_paths: 各视频的保存路径列表，长度决定并发数量

        Returns:
            成功生成的视频路径列表
        """
        coro = self.generate_batch_videos_async(
            prompt=prompt,
            model=model,
            seconds=seconds,
            size=size,
            save_paths=save_paths,
            input_reference_bytes=input_reference_bytes,
            seed=seed,
            progress_callback=progress_callback,
        )
        return self.run_async_in_thread(coro)

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
