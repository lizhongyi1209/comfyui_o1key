"""
Seedance 视频生成客户端
使用 new-api 原生格式：POST /v1/video/generations → GET /v1/video/generations/{task_id}
"""

import asyncio
import json
import os
from typing import Any, Callable, Dict, Optional

import aiohttp

from ..utils.config import get_api_key_or_raise, get_api_base_url


class SeedanceClient:
    """Seedance 视频生成客户端（new-api 原生三段式）"""

    # 提交任务
    CREATE_ENDPOINT = "/v1/video/generations"
    # 查询任务状态：{task_id} 占位
    STATUS_ENDPOINT = "/v1/video/generations/{task_id}"

    POLL_INITIAL_INTERVAL = 4   # 首次轮询等待秒数
    POLL_MAX_INTERVAL = 15      # 最大轮询间隔秒数

    # new-api 返回的成功状态值
    SUCCESS_STATUSES = {"succeeded", "success", "completed", "done", "finished"}
    FAILURE_STATUSES = {"failed", "fail", "error", "expired"}

    def __init__(self):
        self.api_key = get_api_key_or_raise()
        self.base_url = get_api_base_url()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ── 1. 提交任务 ────────────────────────────────────────────────────

    async def submit_async(
        self,
        body: Dict[str, Any],
        session: aiohttp.ClientSession,
    ) -> str:
        """提交视频生成任务，返回 task_id"""
        url = f"{self.base_url}{self.CREATE_ENDPOINT}"
        async with session.post(url, json=body, headers=self._headers()) as resp:
            text = await resp.text()
            if resp.status != 200:
                try:
                    err = json.loads(text)
                    msg = (err.get("error", {}).get("message")
                           or err.get("message")
                           or text)
                except Exception:
                    msg = text
                raise RuntimeError(f"提交失败 ({resp.status}): {msg}")
            data = json.loads(text)

        # new-api 返回字段：id / task_id
        task_id = data.get("id") or data.get("task_id")
        if not task_id:
            raise RuntimeError(f"API 未返回任务 ID，响应：{data}")
        return task_id

    # ── 2. 轮询状态 ────────────────────────────────────────────────────

    async def poll_async(
        self,
        task_id: str,
        session: aiohttp.ClientSession,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> str:
        """轮询任务状态，成功后返回视频 URL"""
        url = f"{self.base_url}{self.STATUS_ENDPOINT.format(task_id=task_id)}"
        interval = self.POLL_INITIAL_INTERVAL

        while True:
            async with session.get(url, headers=self._headers()) as resp:
                text = await resp.text()
                if resp.status != 200:
                    try:
                        err = json.loads(text)
                        msg = (err.get("error", {}).get("message")
                               or err.get("message")
                               or text)
                    except Exception:
                        msg = text
                    raise RuntimeError(f"状态查询失败 ({resp.status}): {msg}")
                result = json.loads(text)

            status = (result.get("status") or "").lower()

            # 调试：打印原始响应（排查状态字段问题后可删除）
            print(f"[Seedance][DEBUG] 原始响应: {result}")

            # 解析进度
            progress_raw = result.get("progress", "0")
            try:
                progress_pct = int(str(progress_raw).rstrip("%").strip())
            except (ValueError, AttributeError):
                progress_pct = 0

            print(f"[Seedance] 生成中 {progress_pct}%  (status={status})")
            if on_progress:
                on_progress(progress_pct)

            if status in self.SUCCESS_STATUSES:
                # 取视频 URL：url / metadata.url / output.video_url
                video_url = (
                    result.get("url")
                    or (result.get("output") or {}).get("video_url")
                    or (result.get("metadata") or {}).get("url")
                )
                if not video_url:
                    raise RuntimeError(f"任务成功但未找到视频 URL，响应：{result}")
                return video_url

            if status in self.FAILURE_STATUSES:
                reason = (
                    result.get("fail_reason")
                    or (result.get("error") or {}).get("message")
                    or "未知错误"
                )
                raise RuntimeError(f"视频生成失败：{reason}")

            await asyncio.sleep(interval)
            interval = min(interval * 1.5, self.POLL_MAX_INTERVAL)

    # ── 3. 下载视频 ────────────────────────────────────────────────────

    async def download_async(
        self,
        video_url: str,
        save_path: str,
        session: aiohttp.ClientSession,
    ) -> str:
        """下载视频到本地，返回本地路径"""
        print(f"[Seedance] 下载视频...")
        async with session.get(video_url, allow_redirects=True) as resp:
            if resp.status != 200:
                raise RuntimeError(f"视频下载失败 ({resp.status})")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(8192):
                    f.write(chunk)
        return save_path

    # ── 全流程入口（供节点调用）────────────────────────────────────────

    async def generate_async(
        self,
        body: Dict[str, Any],
        save_path: str,
        on_stage: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> str:
        """提交 → 轮询 → 下载，返回本地文件路径"""
        connector = aiohttp.TCPConnector(force_close=True)
        async with aiohttp.ClientSession(connector=connector) as session:

            # 提交
            if on_stage:
                on_stage("submitting")
            task_id = await self.submit_async(body, session)
            print(f"[Seedance] 任务已提交 → {task_id}")
            if on_stage:
                on_stage(f"submitted:{task_id}")

            # 轮询
            video_url = await self.poll_async(task_id, session, on_progress=on_progress)

            # 下载
            if on_stage:
                on_stage("downloading")
            path = await self.download_async(video_url, save_path, session)

            if on_stage:
                on_stage("done")
            return path
