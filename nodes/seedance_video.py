"""
Seedance 视频生成节点
节点列表:
  - Seedance: 文生视频 / 图生视频 / 首尾帧生视频（根据图片输入自动切换模式）
"""

import io
import os
import re

import aiohttp
import torch

from ..clients.seedance_client import SeedanceClient
from ..clients.gemini_client import GeminiAPIClient
from ..utils.image_utils import tensor_to_pil, encode_image_to_base64, pil_to_tensor

from comfy_api.latest import InputImpl

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False


# ── 模型列表 ──────────────────────────────────────────────────────────────────

_MODELS = [
    "doubao-seedance-2-0-260128",
    "doubao-seedance-2-0-fast-260128",
]

_RESOLUTIONS = ["720p", "480p"]


# ── 模型能力判断 ──────────────────────────────────────────────────────────────

def _supports_camera_fixed(model: str) -> bool:
    """2.0 系列不支持固定镜头"""
    return False  # 当前仅 2.0 模型，均不支持


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _get_video_output_dir() -> str:
    if FOLDER_PATHS_AVAILABLE:
        base = folder_paths.get_output_directory()
    else:
        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base = os.path.join(os.path.dirname(os.path.dirname(plugin_dir)), "output")
    video_dir = os.path.join(base, "video")
    os.makedirs(video_dir, exist_ok=True)
    return video_dir


def _get_next_counter(directory: str, prefix: str) -> int:
    if not os.path.exists(directory):
        return 1
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)")
    max_counter = 0
    for f in os.listdir(directory):
        m = pattern.match(f)
        if m:
            max_counter = max(max_counter, int(m.group(1)))
    return max_counter + 1


def _tensor_to_base64_url(tensor) -> str:
    """ComfyUI IMAGE tensor → data:image/png;base64,xxx"""
    pil_images = tensor_to_pil(tensor)
    b64 = encode_image_to_base64(pil_images[0], format="PNG")
    return f"data:image/png;base64,{b64}"


async def _url_to_tensor(url: str) -> torch.Tensor:
    """从 URL 下载图片并转为 ComfyUI IMAGE tensor，失败时返回 None"""
    try:
        from PIL import Image
        async with aiohttp.ClientSession() as session:
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return pil_to_tensor([img])
    except Exception as e:
        print(f"[Seedance] 末帧图片下载失败: {e}")
        return None


def _show_balance():
    """完成后打印余额（静默失败）"""
    try:
        client = GeminiAPIClient()
        data = client.query_balance_sync()
        print(f"Seedance: {client.format_balance_info(data)}")
    except Exception:
        pass


def _make_pbar():
    try:
        from comfy.utils import ProgressBar
        return ProgressBar(100)
    except Exception:
        return None


def _make_callbacks(tag: str, pbar):
    def on_stage(stage: str):
        if stage == "submitting":
            print(f"[{tag}] 提交中...")
            if pbar: pbar.update_absolute(0, 100)
        elif stage.startswith("submitted:"):
            print(f"[{tag}] 已提交 → {stage.split(':', 1)[1]}")
            if pbar: pbar.update_absolute(5, 100)
        elif stage == "downloading":
            print(f"[{tag}] 下载视频中...")
            if pbar: pbar.update_absolute(99, 100)
        elif stage == "done":
            print(f"[{tag}] 完成")
            if pbar: pbar.update_absolute(100, 100)

    def on_progress(pct: int):
        if pbar: pbar.update_absolute(5 + int(pct * 0.94), 100)

    return on_stage, on_progress


# ── 统一节点 ─────────────────────────────────────────────────────────────────
#
#   模式由图片输入自动判断：
#     首帧 = None              → T2V 文生视频   （联网搜索生效）
#     首帧 = 图片，尾帧 = None → I2V 图生视频   （固定镜头生效，当前 2.0 不支持故忽略）
#     首帧 = 图片，尾帧 = 图片 → FlipFlop 首尾帧（联网搜索/固定镜头均忽略）

class Seedance:
    """Seedance 视频生成（文生视频 / 图生视频 / 首尾帧，自动判断模式）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词":           ("STRING", {"multiline": True, "default": ""}),
                "模型":             (_MODELS, {"default": "doubao-seedance-2-0-260128"}),
                "分辨率":           (_RESOLUTIONS, {"default": "720p"}),
                "宽高比":           (["16:9", "adaptive", "9:16", "1:1", "4:3", "3:4", "21:9"],
                                     {"default": "16:9"}),
                "时长秒(-1=自动)":  ("INT", {"default": 5, "min": -1, "max": 30, "step": 1}),
                "生成音频":         (["关闭", "打开"], {"default": "关闭"}),
                "联网搜索":         (["关闭", "打开"], {"default": "关闭"}),
                "返回末帧图片":     (["关闭", "打开"], {"default": "关闭"}),
                "seed":             ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
            "optional": {
                "首帧图片":         ("IMAGE",),
                "尾帧图片":         ("IMAGE",),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE")
    RETURN_NAMES = ("视频", "末帧图片")
    FUNCTION = "generate"
    CATEGORY = "comfyui_o1key/Seedance"

    async def generate(self, **kwargs):
        prompt      = kwargs["提示词"].strip()
        model       = kwargs["模型"]
        resolution  = kwargs["分辨率"]
        ratio       = kwargs["宽高比"]
        duration    = kwargs["时长秒(-1=自动)"]
        gen_audio   = kwargs["生成音频"] == "打开"
        web_search  = kwargs["联网搜索"] == "打开"
        return_last = kwargs["返回末帧图片"] == "打开"
        seed        = kwargs.get("seed", 0)
        first_image = kwargs.get("首帧图片", None)
        last_image  = kwargs.get("尾帧图片", None)

        # 模式判断
        if first_image is None and last_image is not None:
            raise ValueError("请同时接入首帧图片，或仅接入首帧图片。")
        if first_image is None:
            mode = "t2v"
            tag  = "Seedance文生视频"
            file_prefix = "seedance_t2v"
        elif last_image is None:
            mode = "i2v"
            tag  = "Seedance图生视频"
            file_prefix = "seedance_i2v"
        else:
            mode = "flipflop"
            tag  = "Seedance首尾帧"
            file_prefix = "seedance_flip"

        if not prompt:
            raise ValueError("提示词不能为空。")
        if duration == -1 and mode == "t2v":
            pass  # 2.0 均支持自动时长
        elif duration == -1 and mode != "t2v":
            pass  # 2.0 均支持自动时长

        metadata: dict = {
            "resolution": resolution,
            "watermark":  False,
        }
        if ratio != "adaptive":
            metadata["ratio"] = ratio
        if duration != -1:
            metadata["duration"] = duration
        if gen_audio:
            metadata["generate_audio"] = True
        if return_last:
            metadata["return_last_frame"] = True
        if seed != 0:
            metadata["seed"] = seed

        # 模式专属参数
        if mode == "t2v":
            if web_search:
                metadata["tools"] = [{"type": "web_search"}]
            body = {
                "model":    model,
                "prompt":   prompt,
                "metadata": metadata,
            }

        elif mode == "i2v":
            first_url = _tensor_to_base64_url(first_image)
            metadata["content"] = [
                {
                    "type":      "image_url",
                    "image_url": {"url": first_url},
                    "role":      "first_frame",
                },
                {"type": "text", "text": prompt},
            ]
            body = {
                "model":    model,
                "prompt":   prompt,
                "images":   [first_url],
                "metadata": metadata,
            }

        else:  # flipflop
            first_url = _tensor_to_base64_url(first_image)
            last_url  = _tensor_to_base64_url(last_image)
            metadata["content"] = [
                {
                    "type":      "image_url",
                    "image_url": {"url": first_url},
                    "role":      "first_frame",
                },
                {
                    "type":      "image_url",
                    "image_url": {"url": last_url},
                    "role":      "last_frame",
                },
                {"type": "text", "text": prompt},
            ]
            body = {
                "model":    model,
                "prompt":   prompt,
                "images":   [first_url],
                "metadata": metadata,
            }

        video_dir = _get_video_output_dir()
        counter   = _get_next_counter(video_dir, file_prefix)
        save_path = os.path.join(video_dir, f"{file_prefix}_{counter:05d}.mp4")

        client            = SeedanceClient()
        pbar              = _make_pbar()
        on_stage, on_prog = _make_callbacks(tag, pbar)

        try:
            result_path, last_frame_url = await client.generate_async(
                body=body, save_path=save_path,
                on_stage=on_stage, on_progress=on_prog,
            )
            last_frame_tensor = None
            if return_last and last_frame_url:
                last_frame_tensor = await _url_to_tensor(last_frame_url)
            return (InputImpl.VideoFromFile(result_path), last_frame_tensor)
        finally:
            _show_balance()


# ── 节点注册 ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "Seedance": Seedance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Seedance": "Seedance 视频生成",
}
