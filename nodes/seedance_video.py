"""
Seedance 视频生成节点
节点列表:
  - Seedance: 文生视频 / 图生视频 / 首尾帧生视频（根据图片输入自动切换模式）
"""

import base64
import io
import json
import os
import tempfile

import aiohttp
import torch

from ..clients.seedance_client import SeedanceClient
from ..clients.gemini_client import GeminiAPIClient
from ..utils.image_utils import tensor_to_pil, pil_to_tensor
from ..utils.r2_uploader import upload_video, upload_audio
from ..utils.config import NETWORK_ROUTE_OPTIONS, get_base_url_by_route

from comfy_api.latest import InputImpl


# ── 模型列表 ──────────────────────────────────────────────────────────────────

_MODELS = [
    "doubao-seedance-2-0-260128",
]

_RESOLUTIONS = ["720p", "1080p", "480p"]

_MAX_IMAGE_BYTES = 30 * 1024 * 1024
_MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024


# ── 模型能力判断 ──────────────────────────────────────────────────────────────

def _supports_camera_fixed(model: str) -> bool:
    """2.0 系列不支持固定镜头"""
    return False  # 当前仅 2.0 模型，均不支持


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _format_mb(size_bytes: int) -> str:
    return f"{size_bytes / 1024 / 1024:.2f}MB"


def _tensor_to_base64_url(tensor, label: str = "图片") -> str:
    """ComfyUI IMAGE tensor → data:image/png;base64,xxx"""
    pil_images = tensor_to_pil(tensor)
    image = pil_images[0]
    if image.mode == "RGBA":
        image = image.convert("RGB")

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()
    image_size = len(image_bytes)

    if image_size > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"Seedance {label}大小 {_format_mb(image_size)} 超过单张图片 "
            f"{_format_mb(_MAX_IMAGE_BYTES)} 限制，请先压缩或缩小图片。"
        )

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _validate_request_body_size(body: dict, tag: str):
    body_size = len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
    if body_size > _MAX_REQUEST_BODY_BYTES:
        raise ValueError(
            f"{tag} 请求体大小 {_format_mb(body_size)} 超过 "
            f"{_format_mb(_MAX_REQUEST_BODY_BYTES)} 限制，请减少参考图片数量或降低图片尺寸。"
        )
    print(
        f"[{tag}] 请求体大小: {_format_mb(body_size)} "
        f"(限制 {_format_mb(_MAX_REQUEST_BODY_BYTES)})"
    )



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
                "网络线路":         (NETWORK_ROUTE_OPTIONS, {"default": "全球加速"}),
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
            first_url = _tensor_to_base64_url(first_image, "首帧图片")
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
            first_url = _tensor_to_base64_url(first_image, "首帧图片")
            last_url  = _tensor_to_base64_url(last_image, "尾帧图片")
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

        _validate_request_body_size(body, tag)

        # 保存路径（临时文件，避免与下游保存节点重复落盘）
        _, save_path = tempfile.mkstemp(suffix=".mp4", prefix=f"{file_prefix}_")

        client            = SeedanceClient()
        client.base_url   = get_base_url_by_route(kwargs.get("网络线路", "全球加速"))
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


# ── 多模态参考生视频节点 ──────────────────────────────────────────────────────

class SeedanceMultiModal:
    """Seedance 2.0 多模态参考生视频（参考图片 + 参考视频 + 参考音频 + 文本）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词":           ("STRING", {"multiline": True, "default": ""}),
                "网络线路":         (NETWORK_ROUTE_OPTIONS, {"default": "全球加速"}),
                "模型":             (_MODELS, {"default": "doubao-seedance-2-0-260128"}),
                "分辨率":           (_RESOLUTIONS, {"default": "720p"}),
                "宽高比":           (["adaptive", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
                                     {"default": "adaptive"}),
                "时长秒(-1=自动)":  ("INT", {"default": 5, "min": -1, "max": 15, "step": 1}),
                "生成音频":         (["关闭", "打开"], {"default": "关闭"}),
                "联网搜索":         (["关闭", "打开"], {"default": "关闭"}),
                "返回末帧图片":     (["关闭", "打开"], {"default": "关闭"}),
                "seed":             ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
            "optional": {
                "参考图片":         ("IMAGE",),
                "参考视频1":        ("VIDEO",),
                "参考视频2":        ("VIDEO",),
                "参考视频3":        ("VIDEO",),
                "参考音频1":        ("AUDIO",),
                "参考音频2":        ("AUDIO",),
                "参考音频3":        ("AUDIO",),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE")
    RETURN_NAMES = ("视频", "末帧图片")
    FUNCTION = "generate"
    CATEGORY = "comfyui_o1key/Seedance"
    INPUT_IS_LIST = True

    async def generate(self, **kwargs):
        # INPUT_IS_LIST=True 时所有参数都是列表，取第一个元素
        def _first(v, default=None):
            if isinstance(v, list):
                return v[0] if v else default
            return v if v is not None else default

        prompt      = _first(kwargs.get("提示词"), "").strip()
        model       = _first(kwargs.get("模型"))
        resolution  = _first(kwargs.get("分辨率"))
        ratio       = _first(kwargs.get("宽高比"))
        duration    = _first(kwargs.get("时长秒(-1=自动)"), 5)
        gen_audio   = _first(kwargs.get("生成音频"), "关闭") == "打开"
        web_search  = _first(kwargs.get("联网搜索"), "关闭") == "打开"
        return_last = _first(kwargs.get("返回末帧图片"), "关闭") == "打开"
        seed        = _first(kwargs.get("seed"), 0)
        network_route = _first(kwargs.get("网络线路"), "全球加速")

        # 参考图片：INPUT_IS_LIST 时是 [tensor, tensor, ...] 列表，直接保留
        raw_images = kwargs.get("参考图片", None)
        ref_images = [img for img in raw_images if img is not None] if raw_images else None

        ref_videos  = [_first(kwargs.get(f"参考视频{i}")) for i in range(1, 4)]
        ref_audios  = [_first(kwargs.get(f"参考音频{i}")) for i in range(1, 4)]

        ref_videos  = [v for v in ref_videos if v is not None]
        ref_audios  = [a for a in ref_audios if a is not None]

        # ── 校验 ──────────────────────────────────────────────────────────
        has_image = bool(ref_images)
        has_video = len(ref_videos) > 0
        has_audio = len(ref_audios) > 0

        if not has_image and not has_video and not has_audio and not prompt:
            raise ValueError("至少需要提供参考图片、参考视频或提示词之一。")
        if has_audio and not has_image and not has_video:
            raise ValueError("不可单独输入音频，请至少连接一张参考图片或一个参考视频。")

        # ── 构建 content 列表 ─────────────────────────────────────────────
        content = []

        # 参考图片（批次，最多9张）
        if has_image:
            imgs = ref_images[:9]
            if len(ref_images) > 9:
                print(f"[SeedanceMultiModal] 参考图片超过9张，仅取前9张（共{len(ref_images)}张）")
            for idx, img_tensor in enumerate(imgs, start=1):
                # 每个 tensor 可能是 [1,H,W,C] 或 [H,W,C]，统一确保有 batch 维
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                url = _tensor_to_base64_url(img_tensor, f"参考图片{idx}")
                content.append({
                    "type":      "image_url",
                    "image_url": {"url": url},
                    "role":      "reference_image",
                })

        # 参考视频（最多3个）
        for v in ref_videos:
            url = await upload_video(v)
            content.append({
                "type":      "video_url",
                "video_url": {"url": url},
                "role":      "reference_video",
            })

        # 参考音频（最多3段）
        for a in ref_audios:
            url = await upload_audio(a)
            content.append({
                "type":      "audio_url",
                "audio_url": {"url": url},
                "role":      "reference_audio",
            })

        # 文本提示词（放最后）
        if prompt:
            content.append({"type": "text", "text": prompt})

        if not content:
            raise ValueError("content 为空，请至少提供参考图片、参考视频或提示词。")

        # ── 构建请求体（new-api 兼容格式）──────────────────────────────────
        metadata: dict = {
            "resolution": resolution,
            "watermark":  False,
            "content":    content,
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
        if web_search:
            metadata["tools"] = [{"type": "web_search"}]

        # 顶层 image：取第一张参考图的 base64（new-api 单图字段）
        first_image_url = next(
            (item["image_url"]["url"] for item in content if item["type"] == "image_url"),
            None,
        )

        body = {
            "model":    model,
            "prompt":   prompt if prompt else " ",
            "metadata": metadata,
        }
        if first_image_url:
            body["image"] = first_image_url

        _validate_request_body_size(body, "Seedance多模态")

        # ── 保存路径（临时文件，避免与下游保存节点重复落盘）──────────────────
        _, save_path = tempfile.mkstemp(suffix=".mp4", prefix="seedance_mm_")

        client            = SeedanceClient()
        client.base_url   = get_base_url_by_route(network_route)
        pbar              = _make_pbar()
        on_stage, on_prog = _make_callbacks("Seedance多模态", pbar)

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
    "Seedance":          Seedance,
    "SeedanceMultiModal": SeedanceMultiModal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Seedance":          "Seedance 视频生成",
    "SeedanceMultiModal": "Seedance 多模态参考生视频",
}
