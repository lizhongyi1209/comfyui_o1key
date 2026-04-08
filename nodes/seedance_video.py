"""
Seedance 视频生成节点
节点列表:
  - SeedanceT2V:       文生视频
  - SeedanceI2V:       图生视频（首帧驱动）
  - SeedanceFlipFlop:  首尾帧生视频
"""

import os
import re

from ..clients.seedance_client import SeedanceClient
from ..clients.gemini_client import GeminiAPIClient
from ..utils.image_utils import tensor_to_pil, encode_image_to_base64

from comfy_api.latest import InputImpl

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False


# ── 模型列表 ──────────────────────────────────────────────────────────────────

_T2V_MODELS = [
    "doubao-seedance-2-0-260128",
    "doubao-seedance-2-0-fast-260128",
    "doubao-seedance-1-5-pro-251215",
    "doubao-seedance-1-0-pro-250528",
    "doubao-seedance-1-0-lite-t2v",
]

_I2V_MODELS = [
    "doubao-seedance-2-0-260128",
    "doubao-seedance-2-0-fast-260128",
    "doubao-seedance-1-5-pro-251215",
    "doubao-seedance-1-0-pro-250528",
    "doubao-seedance-1-0-lite-i2v",
]

_FLIPFLOP_MODELS = [
    "doubao-seedance-2-0-260128",
    "doubao-seedance-2-0-fast-260128",
    "doubao-seedance-1-5-pro-251215",
    "doubao-seedance-1-0-pro-250528",
]


# ── 模型能力判断 ──────────────────────────────────────────────────────────────

def _is_v2(model: str) -> bool:
    return "seedance-2-0" in model

def _is_v15_pro(model: str) -> bool:
    return "seedance-1-5-pro" in model

def _supports_audio(model: str) -> bool:
    """2.0、2.0-fast、1.5-pro 支持生成音频"""
    return _is_v2(model) or _is_v15_pro(model)

def _supports_auto_duration(model: str) -> bool:
    """2.0 和 1.5-pro 支持自动时长（duration 不传或传 -1）"""
    return _is_v2(model) or _is_v15_pro(model)

def _supports_camera_fixed(model: str) -> bool:
    """仅非 2.0 模型支持固定镜头（2.0 已不支持）"""
    return not _is_v2(model)

def _supports_web_search(model: str) -> bool:
    """仅 2.0 系列支持联网搜索"""
    return _is_v2(model)


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
    """生成通用的 on_stage / on_progress 回调"""
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


# ── 节点 1：文生视频 ─────────────────────────────────────────────────────────

class SeedanceT2V:
    """Seedance 文生视频"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词":           ("STRING", {"multiline": True, "default": ""}),
                "模型":             (_T2V_MODELS, {"default": "doubao-seedance-2-0-260128"}),
                "分辨率":           (["720p", "1080p", "480p"], {"default": "720p"}),
                "宽高比":           (["16:9", "adaptive", "9:16", "1:1", "4:3", "3:4", "21:9"],
                                     {"default": "16:9"}),
                "时长秒(-1=自动)":  ("INT", {"default": 5, "min": -1, "max": 30, "step": 1}),
                "生成音频":         (["关闭", "打开"], {"default": "关闭"}),
                "水印":             (["关闭", "打开"], {"default": "关闭"}),
                "返回末帧图片":     (["关闭", "打开"], {"default": "关闭"}),
                "联网搜索":         (["关闭", "打开"], {"default": "关闭"}),
                "服务等级":         (["default", "flex"], {"default": "default"}),
                "seed":             ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频",)
    FUNCTION = "generate"
    CATEGORY = "comfyui_o1key/Seedance"

    async def generate(self, **kwargs):
        prompt       = kwargs["提示词"].strip()
        model        = kwargs["模型"]
        resolution   = kwargs["分辨率"]
        ratio        = kwargs["宽高比"]
        duration     = kwargs["时长秒(-1=自动)"]
        gen_audio    = kwargs["生成音频"] == "打开"
        watermark    = kwargs["水印"] == "打开"
        return_last  = kwargs["返回末帧图片"] == "打开"
        web_search   = kwargs["联网搜索"] == "打开"
        service_tier = kwargs["服务等级"]
        seed         = kwargs.get("seed", 0)

        if not prompt:
            raise ValueError("提示词不能为空。")
        if duration == -1 and not _supports_auto_duration(model):
            raise ValueError(f"模型 {model} 不支持自动时长（-1），请改用 2.0 或 1.5-pro 模型。")

        metadata: dict = {
            "resolution": resolution,
            "watermark":  watermark,
        }
        if ratio != "adaptive":
            metadata["ratio"] = ratio
        if duration != -1:
            metadata["duration"] = duration
        if gen_audio and _supports_audio(model):
            metadata["generate_audio"] = True
        if return_last:
            metadata["return_last_frame"] = True
        if web_search and _supports_web_search(model):
            metadata["tools"] = [{"type": "web_search"}]
        if seed != 0:
            metadata["seed"] = seed

        body = {
            "model":        model,
            "prompt":       prompt,
            "metadata":     metadata,
            "service_tier": service_tier,
        }

        video_dir = _get_video_output_dir()
        counter   = _get_next_counter(video_dir, "seedance_t2v")
        save_path = os.path.join(video_dir, f"seedance_t2v_{counter:05d}.mp4")

        client              = SeedanceClient()
        pbar                = _make_pbar()
        on_stage, on_prog   = _make_callbacks("Seedance文生视频", pbar)

        try:
            result_path = await client.generate_async(
                body=body, save_path=save_path,
                on_stage=on_stage, on_progress=on_prog,
            )
            return (InputImpl.VideoFromFile(result_path),)
        finally:
            _show_balance()


# ── 节点 2：图生视频（首帧驱动） ──────────────────────────────────────────────

class SeedanceI2V:
    """Seedance 图生视频（首帧驱动）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "首帧图片":         ("IMAGE",),
                "提示词":           ("STRING", {"multiline": True, "default": ""}),
                "模型":             (_I2V_MODELS, {"default": "doubao-seedance-2-0-260128"}),
                "分辨率":           (["720p", "1080p", "480p"], {"default": "720p"}),
                "宽高比":           (["16:9", "adaptive", "9:16", "1:1", "4:3", "3:4", "21:9"],
                                     {"default": "16:9"}),
                "时长秒(-1=自动)":  ("INT", {"default": 5, "min": -1, "max": 30, "step": 1}),
                "生成音频":         (["关闭", "打开"], {"default": "关闭"}),
                "水印":             (["关闭", "打开"], {"default": "关闭"}),
                "固定镜头":         (["关闭", "打开"], {"default": "关闭"}),
                "返回末帧图片":     (["关闭", "打开"], {"default": "关闭"}),
                "服务等级":         (["default", "flex"], {"default": "default"}),
                "seed":             ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频",)
    FUNCTION = "generate"
    CATEGORY = "comfyui_o1key/Seedance"

    async def generate(self, **kwargs):
        image        = kwargs["首帧图片"]
        prompt       = kwargs["提示词"].strip()
        model        = kwargs["模型"]
        resolution   = kwargs["分辨率"]
        ratio        = kwargs["宽高比"]
        duration     = kwargs["时长秒(-1=自动)"]
        gen_audio    = kwargs["生成音频"] == "打开"
        watermark    = kwargs["水印"] == "打开"
        cam_fixed    = kwargs["固定镜头"] == "打开"
        return_last  = kwargs["返回末帧图片"] == "打开"
        service_tier = kwargs["服务等级"]
        seed         = kwargs.get("seed", 0)

        if not prompt:
            raise ValueError("提示词不能为空。")
        if duration == -1 and not _supports_auto_duration(model):
            raise ValueError(f"模型 {model} 不支持自动时长（-1），请改用 2.0 或 1.5-pro 模型。")

        image_url = _tensor_to_base64_url(image)

        # 使用 metadata.content 携带带 role 的图片（会覆盖 new-api 从 images 字段构建的 content）
        content = [
            {
                "type":      "image_url",
                "image_url": {"url": image_url},
                "role":      "first_frame",
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]

        metadata: dict = {
            "resolution": resolution,
            "watermark":  watermark,
            "content":    content,
        }
        if ratio != "adaptive":
            metadata["ratio"] = ratio
        if duration != -1:
            metadata["duration"] = duration
        if gen_audio and _supports_audio(model):
            metadata["generate_audio"] = True
        if cam_fixed and _supports_camera_fixed(model):
            metadata["camera_fixed"] = True
        if return_last:
            metadata["return_last_frame"] = True
        if seed != 0:
            metadata["seed"] = seed

        body = {
            "model":        model,
            "prompt":       prompt,
            "images":       [image_url],   # 供 new-api HasImage() 识别，触发正确计费路径
            "metadata":     metadata,
            "service_tier": service_tier,
        }

        video_dir = _get_video_output_dir()
        counter   = _get_next_counter(video_dir, "seedance_i2v")
        save_path = os.path.join(video_dir, f"seedance_i2v_{counter:05d}.mp4")

        client              = SeedanceClient()
        pbar                = _make_pbar()
        on_stage, on_prog   = _make_callbacks("Seedance图生视频", pbar)

        try:
            result_path = await client.generate_async(
                body=body, save_path=save_path,
                on_stage=on_stage, on_progress=on_prog,
            )
            return (InputImpl.VideoFromFile(result_path),)
        finally:
            _show_balance()


# ── 节点 3：首尾帧生视频 ─────────────────────────────────────────────────────

class SeedanceFlipFlop:
    """Seedance 首尾帧生视频（同时指定起始帧与结束帧）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "首帧图片":         ("IMAGE",),
                "尾帧图片":         ("IMAGE",),
                "提示词":           ("STRING", {"multiline": True, "default": ""}),
                "模型":             (_FLIPFLOP_MODELS, {"default": "doubao-seedance-2-0-260128"}),
                "分辨率":           (["720p", "1080p", "480p"], {"default": "720p"}),
                "宽高比":           (["16:9", "adaptive", "9:16", "1:1", "4:3", "3:4", "21:9"],
                                     {"default": "16:9"}),
                "时长秒(-1=自动)":  ("INT", {"default": 5, "min": -1, "max": 30, "step": 1}),
                "生成音频":         (["关闭", "打开"], {"default": "关闭"}),
                "水印":             (["关闭", "打开"], {"default": "关闭"}),
                "返回末帧图片":     (["关闭", "打开"], {"default": "关闭"}),
                "服务等级":         (["default", "flex"], {"default": "default"}),
                "seed":             ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频",)
    FUNCTION = "generate"
    CATEGORY = "comfyui_o1key/Seedance"

    async def generate(self, **kwargs):
        first_image  = kwargs["首帧图片"]
        last_image   = kwargs["尾帧图片"]
        prompt       = kwargs["提示词"].strip()
        model        = kwargs["模型"]
        resolution   = kwargs["分辨率"]
        ratio        = kwargs["宽高比"]
        duration     = kwargs["时长秒(-1=自动)"]
        gen_audio    = kwargs["生成音频"] == "打开"
        watermark    = kwargs["水印"] == "打开"
        return_last  = kwargs["返回末帧图片"] == "打开"
        service_tier = kwargs["服务等级"]
        seed         = kwargs.get("seed", 0)

        if not prompt:
            raise ValueError("提示词不能为空。")
        if duration == -1 and not _supports_auto_duration(model):
            raise ValueError(f"模型 {model} 不支持自动时长（-1），请改用 2.0 或 1.5-pro 模型。")

        first_url = _tensor_to_base64_url(first_image)
        last_url  = _tensor_to_base64_url(last_image)

        content = [
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
            {
                "type": "text",
                "text": prompt,
            },
        ]

        metadata: dict = {
            "resolution": resolution,
            "watermark":  watermark,
            "content":    content,
        }
        if ratio != "adaptive":
            metadata["ratio"] = ratio
        if duration != -1:
            metadata["duration"] = duration
        if gen_audio and _supports_audio(model):
            metadata["generate_audio"] = True
        if return_last:
            metadata["return_last_frame"] = True
        if seed != 0:
            metadata["seed"] = seed

        body = {
            "model":        model,
            "prompt":       prompt,
            "images":       [first_url],   # 供 new-api HasImage() 识别
            "metadata":     metadata,
            "service_tier": service_tier,
        }

        video_dir = _get_video_output_dir()
        counter   = _get_next_counter(video_dir, "seedance_flip")
        save_path = os.path.join(video_dir, f"seedance_flip_{counter:05d}.mp4")

        client              = SeedanceClient()
        pbar                = _make_pbar()
        on_stage, on_prog   = _make_callbacks("Seedance首尾帧", pbar)

        try:
            result_path = await client.generate_async(
                body=body, save_path=save_path,
                on_stage=on_stage, on_progress=on_prog,
            )
            return (InputImpl.VideoFromFile(result_path),)
        finally:
            _show_balance()


# ── 节点注册 ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SeedanceT2V":      SeedanceT2V,
    "SeedanceI2V":      SeedanceI2V,
    "SeedanceFlipFlop": SeedanceFlipFlop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedanceT2V":      "Seedance 文生视频",
    "SeedanceI2V":      "Seedance 图生视频",
    "SeedanceFlipFlop": "Seedance 首尾帧生视频",
}
