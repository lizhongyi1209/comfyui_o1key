"""
Flux 图像编辑 API 客户端
通过 vip.o1key.com 调用 Flux2 图像编辑 + SeedVR2 超分辨率服务

工作流程：
1. submit_task  → POST /v1/images/edits (multipart/form-data 提交主图+参考图+提示词)
2. poll_result  → GET  /v1/images/edits/{task_id} (直连容器轮询)
"""

import base64
import time
from typing import Optional

import requests

from ..utils.config import get_api_key_or_raise, get_api_base_url


# 显示名 → 实际请求值的映射
SIZE_DISPLAY_MAP = {
    "2K": "2048",
    "4K": "4096",
}

# 轮询直连容器地址，绕过代理层
POLL_BASE_URL = "https://xrrh7tn08tfgwa8w-8188.container.x-gpu.com"


class FluxEditClient:
    """
    Flux 图像编辑客户端

    对接 vip.o1key.com 上的 /v1/images/edits 接口，
    将图像编辑+超分辨率任务提交到远程服务器执行。
    """

    SUBMIT_ENDPOINT = "/v1/images/edits"
    STATUS_ENDPOINT = "/v1/images/edits/{task_id}"

    DEFAULT_POLL_INTERVAL = 15  # 秒

    def __init__(self):
        self.api_key = get_api_key_or_raise()
        self.base_url = get_api_base_url()

    # ------------------------------------------------------------------
    # 同步方法（供 ComfyUI 节点调用）
    # ------------------------------------------------------------------

    def submit_and_wait(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str,
        size: str = "4K",
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        progress_callback=None,
    ) -> bytes:
        """
        提交任务并同步等待结果（阻塞直到完成）

        Args:
            image_bytes: 主图二进制数据
            mask_bytes: 参考图二进制数据
            prompt: 编辑提示词
            size: 分辨率显示名 ("2K" 或 "4K")
            poll_interval: 轮询间隔（秒）
            progress_callback: 进度回调 fn(status_str)

        Returns:
            结果图像的二进制数据

        Raises:
            RuntimeError: 任务失败
        """
        size_value = SIZE_DISPLAY_MAP.get(size, size)

        # 1. 提交任务（走代理）
        task_id = self._submit_task_sync(image_bytes, mask_bytes, prompt, size_value)
        if progress_callback:
            progress_callback(f"任务已提交: {task_id[:8]}...")

        # 2. 轮询等待（直连容器）
        return self._poll_result_sync(
            task_id, poll_interval, progress_callback
        )

    def _submit_task_sync(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str,
        size: str,
    ) -> str:
        """同步提交任务，返回 task_id"""
        url = f"{self.base_url}{self.SUBMIT_ENDPOINT}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        files = {
            "image": ("image.jpg", image_bytes, "image/jpeg"),
            "mask": ("mask.jpg", mask_bytes, "image/jpeg"),
        }
        data = {
            "prompt": prompt,
            "size": size,
            "model": "flux2-fp8-dualr",
        }

        try:
            resp = requests.post(url, files=files, data=data, headers=headers, timeout=60)
        except requests.exceptions.Timeout:
            raise RuntimeError("提交任务超时，请检查网络连接")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("无法连接到服务器，请检查网络或服务器地址")

        if resp.status_code != 200:
            raise RuntimeError(
                f"提交任务失败 (HTTP {resp.status_code})\n"
                f"响应: {resp.text[:500]}"
            )

        result = resp.json()
        task_id = result.get("id")
        if not task_id:
            raise RuntimeError(f"服务器返回异常: 未获取到任务ID\n{result}")

        return task_id

    def _poll_result_sync(
        self,
        task_id: str,
        poll_interval: int,
        progress_callback=None,
    ) -> bytes:
        """同步轮询任务状态（直连容器），返回结果图像二进制"""
        url = f"{POLL_BASE_URL}{self.STATUS_ENDPOINT.format(task_id=task_id)}"

        start_time = time.time()
        last_status = None

        while True:
            elapsed = time.time() - start_time

            try:
                resp = requests.get(url, timeout=30)
            except requests.exceptions.ConnectionError:
                raise RuntimeError("轮询时无法连接到服务器，请检查网络")

            if resp.status_code != 200:
                raise RuntimeError(
                    f"查询任务状态失败 (HTTP {resp.status_code})\n"
                    f"响应: {resp.text[:500]}"
                )

            result = resp.json()
            status = result.get("status", "unknown")

            # 状态变化时打印日志
            if status != last_status:
                elapsed_str = f"{elapsed:.0f}s"
                print(f"Flux Edit: [{elapsed_str}] 任务 {task_id[:8]}... → {status}")
                last_status = status

            if progress_callback:
                elapsed_str = f"{elapsed:.0f}s"
                status_desc = {
                    "pending": "排队中",
                    "processing": "处理中",
                    "generating": "生图中，请耐心等待，预计耗时140s左右",
                }.get(status, status)
                progress_callback(f"{status_desc} (当前进度：{elapsed_str})")

            if status == "completed":
                # 解码 base64 图像
                b64_data = result.get("result")
                if not b64_data:
                    raise RuntimeError("任务完成但未返回图像数据")
                return base64.b64decode(b64_data)

            elif status == "failed":
                error_msg = result.get("error", "未知错误")
                raise RuntimeError(
                    f"图像编辑任务失败\n"
                    f"错误: {error_msg}"
                )

            elif status in ("not_found",):
                raise RuntimeError(
                    f"任务未找到: {task_id}\n"
                    f"可能已被清理或 ID 无效"
                )

            # 继续等待
            time.sleep(poll_interval)

    def query_balance_sync(self) -> dict:
        """查询余额（兼容现有节点的 finally 块调用）"""
        return {"name": "flux-edit", "total_available": 0}
