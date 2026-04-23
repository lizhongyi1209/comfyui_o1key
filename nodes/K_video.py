"""
K26 图生视频节点
"""

import asyncio
import json
import os
import re

import aiohttp

from ..utils.config import get_api_key_or_raise, get_api_base_url
from ..utils.image_utils import tensor_to_pil, encode_image_to_base64

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


def _get_video_dir() -> str:
    if _FOLDER_PATHS_OK:
        base = folder_paths.get_output_directory()
    else:
        plugin = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base = os.path.join(os.path.dirname(os.path.dirname(plugin)), "output")
    d = os.path.join(base, "video")
    os.makedirs(d, exist_ok=True)
    return d


def _next_counter(directory: str, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)")
    max_n = 0
    if os.path.exists(directory):
        for f in os.listdir(directory):
            m = pattern.match(f)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _image_to_base64(tensor) -> str:
    pil = tensor_to_pil(tensor)
    return encode_image_to_base64(pil[0], format="PNG")


class KVideo:
    """K26 图生视频节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "起始帧":   ("IMAGE",),
                "提示词":   ("STRING", {"multiline": True, "default": ""}),
                "模式":     (["pro"],),
                "时长":     ([5, 10],),
                "生成音频": (["关闭", "打开"], {"default": "关闭"}),
            },
            "optional": {
                "尾帧":     ("IMAGE",),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频",)
    FUNCTION = "generate"
    CATEGORY = "comfyui_o1key/KVideo"

    async def generate(self, 起始帧, 提示词, 模式, 时长, 生成音频="关闭", 尾帧=None):
        api_key  = get_api_key_or_raise()
        base_url = get_api_base_url()
        headers  = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }

        # ── 动态拼接模型名 ────────────────────────────────────────────
        voice      = "voice" if 生成音频 == "打开" else "novoice"
        model_name = f"{_MODEL_BASE}-{模式}-{时长}s-{voice}"

        # ── 构建请求体 ────────────────────────────────────────────────
        body = {
            "model":    model_name,
            "prompt":   提示词.strip(),
            "image":    _image_to_base64(起始帧),
            "mode":     模式,
            "duration": 时长,
        }
        if 生成音频 == "打开":
            body["generate_audio"] = True
        if 尾帧 is not None:
            body["metadata"] = {"image_tail": _image_to_base64(尾帧)}

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

        # ── 保存路径 ──────────────────────────────────────────────────
        video_dir = _get_video_dir()
        counter   = _next_counter(video_dir, "k26")
        save_path = os.path.join(video_dir, f"k26_{counter:05d}.mp4")

        connector = aiohttp.TCPConnector(ssl=False, force_close=True)
        async with aiohttp.ClientSession(connector=connector) as session:

            # 1. 提交
            _stage("submitting")
            create_url = f"{base_url}{_ENDPOINT_CREATE}"
            async with session.post(create_url, json=body, headers=headers) as resp:
                text = await resp.text()
                if resp.status != 200:
                    try:
                        err = json.loads(text)
                        msg = err.get("error", {}).get("message") or err.get("message") or text
                    except Exception:
                        msg = text
                    raise RuntimeError(f"K26 提交失败 ({resp.status}): {msg}")
                create_resp = json.loads(text)

            task_id = (
                create_resp.get("task_id")
                or create_resp.get("id")
                or create_resp.get("data", {}).get("task_id")
            )
            if not task_id:
                raise RuntimeError(f"API 未返回任务 ID，响应：{create_resp}")
            _stage(f"submitted:{task_id}")

            # 2. 轮询
            status_url = f"{base_url}{_ENDPOINT_STATUS.format(task_id=task_id)}"
            interval   = _POLL_INIT
            video_url  = None

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
                    sr = json.loads(text)

                data   = sr.get("data", sr)
                status = (data.get("status") or sr.get("status") or "").lower()

                pct_raw = data.get("progress", 0)
                try:
                    pct = int(str(pct_raw).rstrip("%").strip())
                except (ValueError, AttributeError):
                    pct = 0
                print(f"[K26 图生视频] 生成中 {pct}%")
                _progress(pct)

                if status in ("success", "completed", "done", "finished", "succeed"):
                    # 提取视频 URL
                    video_url = (
                        data.get("video_url")
                        or data.get("result_url")
                        or data.get("url")
                        or (data.get("result", {}) or {}).get("url")
                        or sr.get("video_url")
                        or sr.get("url")
                    )
                    break
                if status in ("failed", "fail"):
                    err_info = data.get("error") or sr.get("error") or {}
                    err_msg  = (err_info.get("message", "未知错误")
                                if isinstance(err_info, dict) else str(err_info))
                    raise RuntimeError(f"K26 生成失败：{err_msg}")

                await asyncio.sleep(interval)
                interval = min(interval * 1.5, _POLL_MAX)

            if not video_url:
                raise RuntimeError(f"API 未返回视频 URL，响应：{sr}")

            # 3. 下载
            _stage("downloading")
            async with session.get(video_url, allow_redirects=True) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"视频下载失败 ({resp.status})")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        f.write(chunk)

        _stage("done")

        if _FOLDER_PATHS_OK:
            return (InputImpl.VideoFromFile(save_path),)
        return (save_path,)


NODE_CLASS_MAPPINGS = {
    "KVideo": KVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KVideo": "K26 图生视频",
}
