"""
K3 图生视频 自研节点（图生视频 / 多镜头）
模型名根据 模式/时长/音频 动态拼接，不暴露在前端。
起始帧为必填，仅作图生视频；多镜头功能待实现。
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
except Exception:
    _FOLDER_PATHS_OK = False


# ── 常量 ──────────────────────────────────────────────────────────────────────

_MODEL_BASE = "kling-v3"          # 动态拼接为 kling-v3-{模式}-{时长}s-{voice}
_MODES      = ["标准", "专家", "4K"]
_MODE_MAP   = {"标准": "std", "专家": "pro", "4K": "4k"}

_MULTI_SHOT_OPTIONS = [
    "禁用",
    "1个故事板",
    "2个故事板",
    "3个故事板",
    "4个故事板",
    "5个故事板",
    "6个故事板",
]

_ENDPOINT_CREATE = "/v1/video/generations"
_ENDPOINT_STATUS = "/v1/video/generations/{task_id}"

_POLL_INIT = 3
_POLL_MAX  = 15


# ── 工具函数 ───────────────────────────────────────────────────────────────────

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


def _prepare_image_base64(tensor) -> str:
    """转换并校验图片，不符合约束时自动等比缩放后返回 base64。"""
    import io
    import base64

    pil_list = tensor_to_pil(tensor)
    img = pil_list[0].convert("RGB")
    w, h = img.size

    # 1. 宽高比校验（无法通过等比缩放修复，直接报错）
    ratio = w / h
    if ratio < 1 / 2.5 or ratio > 2.5:
        raise RuntimeError(
            f"图片宽高比 {w}:{h}（{ratio:.2f}）超出允许范围 1:2.5 ~ 2.5:1，请裁剪后重试。"
        )

    # 2. 最小尺寸：任意边 < 300px 时等比放大
    if w < 300 or h < 300:
        scale = max(300 / w, 300 / h)
        img = img.resize((int(w * scale), int(h * scale)), resample=1)  # LANCZOS=1

    # 3. 文件大小：循环等比缩小直到 ≤ 10MB
    MAX_BYTES = 10 * 1024 * 1024
    for _ in range(20):                      # 最多迭代 20 次，防止死循环
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        if buf.tell() <= MAX_BYTES:
            break
        scale = (MAX_BYTES / buf.tell()) ** 0.5 * 0.95   # 留 5% 余量
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

class K3Video:
    """K3 图生视频 自研"""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "多镜头": (_MULTI_SHOT_OPTIONS, {
                "default": "禁用",
                "tooltip": "禁用：单段模式；N个故事板：启用 N 段分镜。",
            }),
            "起始帧":    ("IMAGE",),
            "提示词":    ("STRING", {"multiline": True, "default": ""}),
            "负向提示词": ("STRING", {"multiline": True, "default": ""}),
            "时长":      ([5, 10, 15], {"default": 5}),
            "生成音频":  (["关闭", "打开"], {"default": "关闭"}),
            "模式":      (_MODES, {"default": "标准"}),
            "seed": ("INT", {
                "default": 0, "min": 0, "max": 2147483647,
                "tooltip": "seed 仅控制节点是否重新运行，结果本身不可复现。",
            }),
        }

        for i in range(1, 7):
            required[f"分镜{i}_提示词"] = ("STRING", {
                "multiline": True, "default": "",
                "tooltip": f"第 {i} 段分镜提示词，最多 512 字符。",
            })
            required[f"分镜{i}_时长"] = ("INT", {
                "default": 4, "min": 1, "max": 15,
                "display": "slider",
                "tooltip": f"第 {i} 段分镜时长（秒）。",
            })

        return {"required": required}

    RETURN_TYPES  = ("VIDEO",)
    RETURN_NAMES  = ("视频",)
    FUNCTION      = "generate"
    CATEGORY      = "comfyui_o1key/KVideo"

    async def generate(self, 多镜头, 起始帧, 提示词, 负向提示词, 时长, 生成音频, 模式, seed, **kwargs):
        api_key  = get_api_key_or_raise()
        base_url = get_api_base_url()
        headers  = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }

        is_multi  = 多镜头 != "禁用"
        voice     = "voice" if 生成音频 == "打开" else "novoice"
        mode_api  = _MODE_MAP[模式]
        if mode_api == "4k":
            model_name = f"{_MODEL_BASE}-4k-{时长}s"
        else:
            model_name = f"{_MODEL_BASE}-{mode_api}-{时长}s-{voice}"

        # ── 多镜头模式 ────────────────────────────────────────────────
        if is_multi:
            shot_count = int(多镜头[0])  # "3个故事板" → 3

            # 收集分镜参数
            multi_prompt = []
            for i in range(1, shot_count + 1):
                p = kwargs.get(f"分镜{i}_提示词", "").strip()
                d = kwargs.get(f"分镜{i}_时长", 0)
                if not p:
                    raise RuntimeError(f"多镜头模式错误：第 {i} 段分镜提示词不能为空。")
                if d < 1:
                    raise RuntimeError(f"多镜头模式错误：第 {i} 段分镜时长不能小于 1 秒。")
                multi_prompt.append({"index": i, "prompt": p, "duration": str(d)})

            # 校验时长总和
            total = sum(int(s["duration"]) for s in multi_prompt)
            if total != 时长:
                raise RuntimeError(
                    f"多镜头模式错误：各分镜时长之和（{total}s）必须等于总时长（{时长}s）。"
                )

            metadata: dict = {
                "multi_shot": "true",
                "shot_type":  "customize",
                "multi_prompt": multi_prompt,
            }
            if 生成音频 == "打开":
                metadata["sound"] = "on"

            body: dict = {
                "model":    model_name,
                "prompt":   提示词.strip() or " ",
                "mode":     mode_api,
                "duration": 时长,
                "image":    _prepare_image_base64(起始帧),
                "metadata": metadata,
            }
            if 负向提示词.strip():
                body["negative_prompt"] = 负向提示词.strip()

        # ── 单段图生视频模式 ──────────────────────────────────────────
        else:
            if not 提示词.strip():
                raise RuntimeError("单段模式错误：提示词不能为空。")

            body = {
                "model":    model_name,
                "prompt":   提示词.strip(),
                "mode":     mode_api,
                "duration": 时长,
                "image":    _prepare_image_base64(起始帧),
            }
            if 负向提示词.strip():
                body["negative_prompt"] = 负向提示词.strip()
            if 生成音频 == "打开":
                body["generate_audio"] = True

        # ── 进度条 ────────────────────────────────────────────────────
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(100)
        except Exception:
            pbar = None

        tag = "多镜头" if is_multi else "图生视频"

        def _stage(s: str):
            if s == "submitting":
                print(f"[K3 {tag}] 提交中...")
                if pbar: pbar.update_absolute(0, 100)
            elif s.startswith("submitted:"):
                print(f"[K3 {tag}] 任务已提交 → {s.split(':', 1)[1]}")
                if pbar: pbar.update_absolute(5, 100)
            elif s == "downloading":
                print(f"[K3 {tag}] 下载视频...")
                if pbar: pbar.update_absolute(99, 100)
            elif s == "done":
                print(f"[K3 {tag}] 完成")
                if pbar: pbar.update_absolute(100, 100)

        def _progress(pct: int):
            if pbar: pbar.update_absolute(5 + int(pct * 0.94), 100)

        # ── 保存路径 ──────────────────────────────────────────────────
        video_dir = _get_video_dir()
        counter   = _next_counter(video_dir, "k3")
        save_path = os.path.join(video_dir, f"k3_{counter:05d}.mp4")

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
                    raise RuntimeError(f"K3 提交失败 ({resp.status}): {msg}")
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
                print(f"[K3 {tag}] 生成中 {pct}%")
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
                    raise RuntimeError(f"K3 生成失败：{err_msg}")

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


# ── 节点注册 ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "K3Video": K3Video,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "K3Video": "K3 图生视频 自研",
}
