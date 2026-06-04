"""
K26 图生视频节点
支持 720p 和 1080p 模式
"""

import asyncio
import json
import math
import os
import tempfile

import aiohttp

from ..utils.config import get_api_key_or_raise, get_api_base_url, NETWORK_ROUTE_OPTIONS, get_base_url_by_route
from ..utils.image_utils import tensor_to_pil, encode_image_to_base64
from ..utils.http_error import async_request_with_retry
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

try:
    from comfy_api.latest import InputImpl
    import folder_paths
    _FOLDER_PATHS_OK = True
except ImportError:
    _FOLDER_PATHS_OK = False

# 模型基础名，运行时动态拼接完整名称
_MODEL_BASE = "kling-v2-6"

# API 端点
_ENDPOINT_CREATE = "/v1/video/generations"
_ENDPOINT_STATUS = "/v1/video/generations/{task_id}"

_POLL_INIT     = 3
_POLL_MAX      = 15


def _image_to_base64(tensor, scale=1.0) -> str:
    from PIL import Image
    pil = tensor_to_pil(tensor)
    img = pil[0]
    if scale < 1.0:
        w, h = img.size
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return encode_image_to_base64(img, format="PNG")


class KVideoImage2Video:
    """K26 图生视频节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "起始帧":   ("IMAGE",),
                "提示词":   ("STRING", {"multiline": True, "default": ""}),
                "模式":     (["720p", "1080p"], {"default": "720p"}),
                "时长":     ([5, 10], {"default": 5}),
                "生成音频": (["关闭", "打开"], {"default": "关闭"}),
                "网络线路": (NETWORK_ROUTE_OPTIONS, {"default": "全球加速"}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2147483647,
                    "tooltip": "seed 仅控制节点是否重新运行，结果本身不可复现。",
                }),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频",)
    FUNCTION = "generate"
    CATEGORY = "comfyui_o1key/KVideo"

    async def generate(self, 起始帧, 提示词, 模式, 时长, 生成音频="关闭", 网络线路="全球加速", seed=0):
        if 模式 == "720p" and 生成音频 == "打开":
            raise RuntimeError("K26 仅1080p支持音频，请将模式切换为1080p或关闭生成音频。")

        api_key  = get_api_key_or_raise()
        base_url = get_base_url_by_route(网络线路)
        headers  = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }

        # ── 动态拼接模型名 ────────────────────────────────────────────
        mode_api   = "std" if 模式 == "720p" else "pro"
        voice      = "voice" if 生成音频 == "打开" else "novoice"
        model_name = f"{_MODEL_BASE}-{mode_api}-{时长}s-{voice}"

        # ── 构建请求体（超过 10MB 自动缩放图片）────────────────────────
        MAX_BODY  = 10 * 1024 * 1024
        scale     = 1.0

        print(f"[K26 图生视频] 请求体大小限制: 10MB，超出将自动缩放图片")

        while True:
            body = {
                "model":    model_name,
                "prompt":   提示词.strip(),
                "image":    _image_to_base64(起始帧, scale),
                "mode":     mode_api,
                "duration": 时长,
            }
            if 生成音频 == "打开":
                body["metadata"] = {"sound": "on"}

            body_str  = json.dumps(body, ensure_ascii=False)
            body_size = len(body_str.encode("utf-8"))

            if body_size <= MAX_BODY:
                print(f"[K26 图生视频] 请求体大小: {body_size / 1024 / 1024:.2f}MB"
                      + (f"（已缩放至 {scale:.1%}）" if scale < 1.0 else ""))
                break

            # 等比缩放：图片像素面积与 base64 长度近似线性
            target_ratio = MAX_BODY / body_size
            scale = scale * math.sqrt(target_ratio) * 0.95  # 5% 安全余量

            if scale < 0.01:
                raise RuntimeError("图片缩放后仍超过10MB限制，请使用更小的参考图")

            w, h = tensor_to_pil(起始帧)[0].size
            print(f"[K26 图生视频] 请求体 {body_size / 1024 / 1024:.2f}MB 超限，"
                  f"自动缩放至 {scale:.1%}（{int(w * scale)}x{int(h * scale)}）")

        # ── 进度条 ────────────────────────────────────────────────────
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(100)
        except Exception:
            pbar = None

        def _stage(s: str):
            if s == "submitting":
                print("[K26 图生视频] 提交中...")
                if pbar: pbar.update_absolute(0, 100)
            elif s.startswith("submitted:"):
                print(f"[K26 图生视频] 任务已提交 → {s.split(':', 1)[1]}")
                if pbar: pbar.update_absolute(5, 100)
            elif s == "downloading":
                print("[K26 图生视频] 下载视频...")
                if pbar: pbar.update_absolute(99, 100)
            elif s == "done":
                print("[K26 图生视频] 完成")
                if pbar: pbar.update_absolute(100, 100)

        def _progress(pct: int):
            if pbar: pbar.update_absolute(5 + int(pct * 0.94), 100)

        # ── 保存路径（临时文件，避免与下游保存节点重复落盘）──────────────────
        tmp_fd, save_path = tempfile.mkstemp(suffix=".mp4", prefix="k26_")

        connector = aiohttp.TCPConnector(ssl=False, force_close=True)
        async with aiohttp.ClientSession(connector=connector) as session:

            # 1. 提交
            check_interrupt()
            _stage("submitting")
            create_url = f"{base_url}{_ENDPOINT_CREATE}"
            resp = await run_with_interrupt(async_request_with_retry(
                session, "POST", create_url, json=body, headers=headers, prefix="K26 图生视频提交: "
            ))
            check_interrupt()
            sr = await resp.json()

            task_id = sr.get("task_id") or sr.get("id")
            if not task_id:
                raise RuntimeError(f"API 未返回 task_id，响应：{sr}")

            _stage(f"submitted:{task_id}")

            # 2. 轮询
            status_url = f"{base_url}{_ENDPOINT_STATUS.format(task_id=task_id)}"
            interval   = _POLL_INIT
            video_url  = None

            while True:
                await interruptible_sleep(interval)

                check_interrupt()
                async with session.get(status_url, headers=headers) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        raise RuntimeError(f"查询失败 ({resp.status}): {err_text}")
                    sr = await resp.json()

                data   = sr.get("data", {}) or {}
                status = extract_status(sr)

                pct = extract_progress(sr)
                print(f"[K26 图生视频] 生成中 {pct}%")
                _progress(pct)

                if is_success_status(status):
                    # 提取视频 URL
                    video_url = extract_video_url(sr)
                    break
                if is_failure_status(status, sr):
                    err_msg = extract_error_message(sr)
                    raise RuntimeError(f"K26 生成失败：{err_msg}")

                interval = min(interval * 1.5, _POLL_MAX)

            if not video_url:
                raise RuntimeError(f"API 未返回视频 URL，响应：{sr}")

            # 3. 下载
            check_interrupt()
            _stage("downloading")
            async with session.get(video_url, allow_redirects=True) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"视频下载失败 ({resp.status})")
                os.close(tmp_fd)
                with open(save_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        check_interrupt()
                        f.write(chunk)

        _stage("done")

        if _FOLDER_PATHS_OK:
            return (InputImpl.VideoFromFile(save_path),)
        return (save_path,)


NODE_CLASS_MAPPINGS = {
    "KVideoImage2Video": KVideoImage2Video,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KVideoImage2Video": "K26 图生视频",
}
