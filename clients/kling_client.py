"""
Kling 视频生成 API 客户端
"""

import asyncio
import json
import os
from typing import Any, Callable, Dict, Optional

import aiohttp

from ..utils.config import get_api_key_or_raise, get_api_base_url


class KlingClient:
    """Kling 视频生成客户端"""

    ENDPOINTS = {
        "image2video":    "/kling/v1/videos/image2video",
        "text2video":     "/kling/v1/videos/text2video",
        "motion_control": "/kling/v1/videos/motion-control",
    }

    # new API 三段式端点（动作控制走这里）
    NEW_API_CREATE   = "/v1/videos"
    NEW_API_STATUS   = "/v1/videos/{video_id}"
    NEW_API_CONTENT  = "/v1/videos/{video_id}/content"

    POLL_INITIAL_INTERVAL = 3
    POLL_MAX_INTERVAL = 15

    def __init__(self):
        self.api_key = get_api_key_or_raise()
        self.base_url = get_api_base_url()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ── 提交任务 ──────────────────────────────────────────────────────

    async def create_video_async(
        self,
        endpoint_type: str,
        body: Dict[str, Any],
        session: aiohttp.ClientSession,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{self.ENDPOINTS[endpoint_type]}"

        async with session.post(url, json=body, headers=self._headers()) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"提交失败 ({resp.status}): {text}")
            return json.loads(text)

    # ── 轮询状态 ──────────────────────────────────────────────────────

    async def poll_status_async(
        self,
        task_id: str,
        endpoint_type: str,
        session: aiohttp.ClientSession,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{self.ENDPOINTS[endpoint_type]}/{task_id}"
        interval = self.POLL_INITIAL_INTERVAL

        while True:
            async with session.get(url, headers=self._headers()) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"状态查询失败 ({resp.status}): {text}")
                result = json.loads(text)

            data = result.get("data", {})
            inner_data = data.get("data", {}) if isinstance(data, dict) else {}
            status = (
                data.get("status") or
                inner_data.get("task_status") or
                result.get("status") or
                ""
            )
            status = status.lower() if status else ""

            progress_str = data.get("progress", "0%")
            try:
                progress_pct = int(str(progress_str).replace("%", "").strip())
            except (ValueError, AttributeError):
                progress_pct = 0

            print(f"[视频生成] 生成中 {progress_pct}%")

            if on_progress:
                on_progress(progress_pct)

            if status in ("success", "completed", "done", "finished", "succeed"):
                return result
            elif status in ("failed", "fail"):
                error_info = result.get("error", {})
                if isinstance(error_info, dict):
                    error_msg = error_info.get("message", "未知错误")
                else:
                    error_msg = str(error_info)
                raise RuntimeError(f"生成失败：{error_msg}")

            await asyncio.sleep(interval)
            interval = min(interval * 1.5, self.POLL_MAX_INTERVAL)

    # ── 下载视频 ──────────────────────────────────────────────────────

    async def download_video_async(
        self,
        video_url: str,
        save_path: str,
        session: aiohttp.ClientSession,
    ) -> str:
        print("[视频生成] 下载视频...")
        async with session.get(video_url, allow_redirects=True) as resp:
            if resp.status != 200:
                raise RuntimeError(f"视频下载失败 ({resp.status})")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(8192):
                    f.write(chunk)
        return save_path

    # ── 异步入口（供节点调用）────────────────────────────────────────

    async def generate_async(
        self,
        endpoint_type: str,
        body: Dict[str, Any],
        save_path: str,
        on_stage: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> str:
        """提交 → 轮询 → 下载，返回本地文件路径"""
        connector = aiohttp.TCPConnector(ssl=False, force_close=True)
        async with aiohttp.ClientSession(connector=connector) as session:
            if on_stage:
                on_stage("submitting")

            result = await self.create_video_async(endpoint_type, body, session)
            # 提交响应结构：result.data.task_id
            task_id = result.get("task_id") or result.get("data", {}).get("task_id")
            if not task_id:
                raise RuntimeError(f"API 未返回任务 ID，响应：{result}")
            if on_stage:
                on_stage(f"submitted:{task_id}")

            if on_stage:
                on_stage("polling")
            final = await self.poll_status_async(
                task_id, endpoint_type, session, on_progress=on_progress
            )

            # 兼容多种URL路径
            # 响应结构：result.data.result_url 或 result.data.data.task_result.videos[0].url
            data = final.get("data", {})
            inner_data = data.get("data", {}) if isinstance(data, dict) else {}
            video_url = (
                data.get("result_url") or
                final.get("url") or
                final.get("video_url") or
                (inner_data.get("task_result", {}).get("videos", [{}])[0].get("url")
                 if inner_data.get("task_result", {}).get("videos") else None)
            )
            if not video_url:
                raise RuntimeError(f"API 未返回视频 URL，响应：{final}")

            if on_stage:
                on_stage("downloading")
            path = await self.download_video_async(video_url, save_path, session)

            if on_stage:
                on_stage("done")
            return path

    # ── 动作控制：走 new API 三段式流程 ──────────────────────────────

    async def motion_control_async(
        self,
        body: Dict[str, Any],
        save_path: str,
        on_stage: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> str:
        """
        动作控制专用入口：
          POST /v1/videos  →  GET /v1/videos/{id}  →  GET /v1/videos/{id}/content
        body 字段与 Kling 官方动作控制接口一致（image_url/video_url/prompt/...）。
        """
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        interval = self.POLL_INITIAL_INTERVAL

        connector = aiohttp.TCPConnector(ssl=False, force_close=True)
        async with aiohttp.ClientSession(connector=connector) as session:

            # 1. 提交
            if on_stage:
                on_stage("submitting")
            create_url = f"{self.base_url}{self.NEW_API_CREATE}"
            async with session.post(create_url, json=body, headers=headers) as resp:
                text = await resp.text()
                if resp.status != 200:
                    # 尝试提取友好错误信息
                    try:
                        err = json.loads(text)
                        msg = err.get("error", {}).get("message") or err.get("message") or text
                    except Exception:
                        msg = text
                    raise RuntimeError(f"动作控制提交失败 ({resp.status}): {msg}")
                create_resp = json.loads(text)

            video_id = create_resp.get("id")
            if not video_id:
                raise RuntimeError(f"API 未返回视频 ID，响应：{create_resp}")
            if on_stage:
                on_stage(f"submitted:{video_id}")

            # 2. 轮询
            status_url = f"{self.base_url}{self.NEW_API_STATUS.format(video_id=video_id)}"
            while True:
                async with session.get(status_url, headers=headers) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        try:
                            err = json.loads(text)
                            msg = err.get("error", {}).get("message") or err.get("message") or text
                        except Exception:
                            msg = text
                        raise RuntimeError(f"状态查询失败 ({resp.status}): {msg}")
                    status_resp = json.loads(text)

                status = status_resp.get("status", "").lower()
                progress_raw = status_resp.get("progress", 0)
                try:
                    progress_pct = int(str(progress_raw).rstrip("%").strip())
                except (ValueError, AttributeError):
                    progress_pct = 0

                print(f"[动作控制] 生成中 {progress_pct}%")
                if on_progress:
                    on_progress(progress_pct)

                if status == "completed":
                    break
                if status == "failed":
                    error_info = status_resp.get("error", {})
                    error_msg = (error_info.get("message", "未知错误")
                                 if isinstance(error_info, dict) else str(error_info))
                    raise RuntimeError(f"动作控制生成失败：{error_msg}")

                await asyncio.sleep(interval)
                interval = min(interval * 1.5, self.POLL_MAX_INTERVAL)

            # 3. 下载
            if on_stage:
                on_stage("downloading")
            content_url = f"{self.base_url}{self.NEW_API_CONTENT.format(video_id=video_id)}"
            async with session.get(content_url, headers=headers,
                                   allow_redirects=True) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"视频下载失败 ({resp.status})")
                content_type = resp.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    data = await resp.json()
                    download_url = data.get("url") or data.get("download_url")
                    if not download_url:
                        raise RuntimeError("视频下载失败：响应中未找到下载链接")
                    async with session.get(download_url) as dl_resp:
                        if dl_resp.status != 200:
                            raise RuntimeError(f"从下载链接获取视频失败 ({dl_resp.status})")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        with open(save_path, "wb") as f:
                            async for chunk in dl_resp.content.iter_chunked(8192):
                                f.write(chunk)
                else:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            f.write(chunk)

            if on_stage:
                on_stage("done")
            return save_path

