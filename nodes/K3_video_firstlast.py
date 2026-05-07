"""
首尾帧 K3 自研节点
基于 K3 图生视频 自研，去掉分镜功能，新增尾帧可选输入。
"""

import asyncio
import json
import os
import tempfile

import aiohttp

from ..utils.config import get_api_key_or_raise, get_async_api_base_url
from ..utils.image_utils import tensor_to_pil, encode_image_to_base64

try:
    from comfy_api.latest import InputImpl
    import folder_paths
    _FOLDER_PATHS_OK = True
except Exception:
    _FOLDER_PATHS_OK = False


# ── 常量 ──────────────────────────────────────────────────────────────────────

_MODEL_BASE = "kling-v3"
_MODES      = ["720p", "1080p", "4K"]
_MODE_MAP   = {"720p": "std", "1080p": "pro", "4K": "4k"}

_ENDPOINT_CREATE = "/v1/video/generations"
_ENDPOINT_STATUS = "/v1/video/generations/{task_id}"

_POLL_INIT = 3
_POLL_MAX  = 15


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _prepare_image_base64(tensor) -> str:
    """转换并校验图片，不符合约束时自动等比缩放后返回 base64。"""
    import io
    import base64

    pil_list = tensor_to_pil(tensor)
    img = pil_list[0].convert("RGB")
    w, h = img.size

    # 1. 宽高比校验
    ratio = w / h
    if ratio < 1 / 2.5 or ratio > 2.5:
        raise RuntimeError(
            f"图片宽高比 {w}:{h}（{ratio:.2f}）超出允许范围 1:2.5 ~ 2.5:1，请裁剪后重试。"
        )

    # 2. 最小尺寸：任意边 < 300px 时等比放大
    if w < 300 or h < 300:
        scale = max(300 / w, 300 / h)
        img = img.resize((int(w * scale), int(h * scale)), resample=1)

    # 3. 文件大小：循环等比缩小直到 ≤ 10MB
    MAX_BYTES = 10 * 1024 * 1024
    for _ in range(20):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        if buf.tell() <= MAX_BYTES:
            break
        scale = (MAX_BYTES / buf.tell()) ** 0.5 * 0.95
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        if new_w < 300 or new_h < 300:
            raise RuntimeError(
                f"图片压缩至 10MB 以内后尺寸（{new_w}x{new_h}）低于最小限制 300px，无法同时满足两项约束。"
            )
        img = img.resize((new_w, new_h), resample=1)
    else:
        raise RuntimeError("图片经过 20 次缩放仍超过 10MB，请检查原始图片。")

    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── 节点 ──────────────────────────────────────────────────────────────────────

class K3VideoFirstLast:
    """首尾帧 K3 自研"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "起始帧":    ("IMAGE",),
                "提示词":    ("STRING", {"multiline": True, "default": ""}),
                "负向提示词": ("STRING", {"multiline": True, "default": ""}),
                "时长":      ([5, 10, 15], {"default": 5}),
                "生成音频":  (["关闭", "打开"], {"default": "关闭"}),
                "模式":      (_MODES, {"default": "720p"}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2147483647,
                    "tooltip": "seed 仅控制节点是否重新运行，结果本身不可复现。",
                }),
            },
            "optional": {
                "尾帧": ("IMAGE", {"tooltip": "可选。传入后将作为视频尾帧参考。"}),
            },
        }

    RETURN_TYPES  = ("VIDEO",)
    RETURN_NAMES  = ("视频",)
    FUNCTION      = "generate"
    CATEGORY      = "comfyui_o1key/KVideo"

    async def generate(self, 起始帧, 提示词, 负向提示词, 时长, 生成音频, 模式, seed, 尾帧=None):
        api_key  = get_api_key_or_raise()
        base_url = get_async_api_base_url()
        headers  = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }

        voice      = "voice" if 生成音频 == "打开" else "novoice"
        mode_api   = _MODE_MAP[模式]
        if mode_api == "4k":
            model_name = f"{_MODEL_BASE}-4k-{时长}s"
        else:
            model_name = f"{_MODEL_BASE}-{mode_api}-{时长}s-{voice}"

        if not 提示词.strip():
            raise RuntimeError("提示词不能为空。")

        # ── 构建请求体 ────────────────────────────────────────────────
        body: dict = {
            "model":    model_name,
            "prompt":   提示词.strip(),
            "mode":     mode_api,
            "duration": 时长,
            "image":    _prepare_image_base64(起始帧),
        }

        if 负向提示词.strip():
            body["negative_prompt"] = 负向提示词.strip()

        # metadata：尾帧 + 音频
        metadata: dict = {}
        if 尾帧 is not None:
            metadata["image_tail"] = _prepare_image_base64(尾帧)
        if 生成音频 == "打开":
            metadata["sound"] = "on"
        if metadata:
            body["metadata"] = metadata

        # generate_audio 字段（非 metadata 路径）
        if 生成音频 == "打开" and not metadata.get("sound"):
            body["generate_audio"] = True

        # ── 进度条 ────────────────────────────────────────────────────
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(100)
        except Exception:
            pbar = None

        def _stage(s: str):
            if s == "submitting":
                print("[K3 首尾帧] 提交中...")
                if pbar: pbar.update_absolute(0, 100)
            elif s.startswith("submitted:"):
                print(f"[K3 首尾帧] 任务已提交 → {s.split(':', 1)[1]}")
                if pbar: pbar.update_absolute(5, 100)
            elif s == "downloading":
                print("[K3 首尾帧] 下载视频...")
                if pbar: pbar.update_absolute(99, 100)
            elif s == "done":
                print("[K3 首尾帧] 完成")
                if pbar: pbar.update_absolute(100, 100)

        def _progress(pct: int):
            if pbar: pbar.update_absolute(5 + int(pct * 0.94), 100)

        # ── 保存路径（临时文件，避免与下游保存节点重复落盘）──────────────────
        tmp_fd, save_path = tempfile.mkstemp(suffix=".mp4", prefix="k3fl_")

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
                    raise RuntimeError(f"K3 首尾帧提交失败 ({resp.status}): {msg}")
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
                print(f"[K3 首尾帧] 生成中 {pct}%")
                _progress(pct)

                if status in ("success", "completed", "done", "finished", "succeed"):
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
                    raise RuntimeError(f"K3 首尾帧生成失败：{err_msg}")

                await asyncio.sleep(interval)
                interval = min(interval * 1.5, _POLL_MAX)

            if not video_url:
                raise RuntimeError(f"API 未返回视频 URL，响应：{sr}")

            # 3. 下载
            _stage("downloading")
            async with session.get(video_url, allow_redirects=True) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"视频下载失败 ({resp.status})")
                os.close(tmp_fd)
                with open(save_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        f.write(chunk)

        _stage("done")

        if _FOLDER_PATHS_OK:
            return (InputImpl.VideoFromFile(save_path),)
        return (save_path,)


# ── 节点注册 ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "K3VideoFirstLast": K3VideoFirstLast,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "K3VideoFirstLast": "首尾帧 K3 自研",
}
