"""
Grok Video node.

Submits a /v1/videos task, polls until completion, downloads the mp4,
and returns ComfyUI's native VIDEO object.
"""

import json
import os
from typing import List, Optional

from ..clients.grok_video_client import GrokVideoClient
from ..utils.config import NETWORK_ROUTE_OPTIONS, get_base_url_by_route
from ..utils.image_utils import encode_images_for_request_body_limit, tensor_to_pil

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

try:
    from comfy.utils import ProgressBar
    PROGRESS_BAR_AVAILABLE = True
except ImportError:
    ProgressBar = None
    PROGRESS_BAR_AVAILABLE = False

try:
    from comfy_api.input_impl import VideoFromFile
except Exception:
    try:
        from comfy_api.latest import InputImpl
        VideoFromFile = InputImpl.VideoFromFile
    except Exception:
        VideoFromFile = None


MODEL_OPTIONS = ["grok-imagine-video-1.5-preview", "grok-imagine-1.0-video"]
ASPECT_RATIO_OPTIONS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
QUALITY_OPTIONS = ["720p"]
QUALITY_VALUE_MAP = {
    "720p": "high",
}
MODEL_SECONDS_OPTIONS = {
    "grok-imagine-1.0-video": [6, 10, 12, 16, 20],
}

MAX_REFERENCE_IMAGES = 3
MAX_REQUEST_BODY_BYTES = 20 * 1024 * 1024


def _get_output_dir() -> str:
    if FOLDER_PATHS_AVAILABLE:
        base = folder_paths.get_output_directory()
    else:
        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        comfy_root = os.path.dirname(os.path.dirname(plugin_dir))
        base = os.path.join(comfy_root, "output")

    output_dir = os.path.join(base, "grok_video")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _format_mb(size_bytes: int) -> str:
    return f"{size_bytes / 1024 / 1024:.2f}MB"


def _image_tensor_to_first_pil(image_tensor):
    if image_tensor is None:
        return None

    pil_images = tensor_to_pil(image_tensor)
    if not pil_images:
        return None

    image = pil_images[0]
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image


def _collect_reference_images(**kwargs) -> List[object]:
    images = []
    for i in range(1, MAX_REFERENCE_IMAGES + 1):
        image = _image_tensor_to_first_pil(kwargs.get(f"参考图{i}"))
        if image is not None:
            images.append(image)
    return images


def _to_data_urls(encoded_images) -> List[str]:
    return [f"data:{mime};base64,{b64}" for mime, b64 in encoded_images]


def _encode_image_data_urls(
    images: List[object],
    prompt: str,
    model: str,
    aspect_ratio: str,
    seconds: int,
    quality: str,
) -> Optional[List[str]]:
    if not images:
        return None

    def build_body(encoded_images):
        return GrokVideoClient.build_video_body(
            prompt=prompt,
            model=model,
            aspect_ratio=aspect_ratio,
            seconds=seconds,
            quality=quality,
            images=_to_data_urls(encoded_images),
        )

    encoded = encode_images_for_request_body_limit(
        images,
        build_body=build_body,
        max_body_bytes=MAX_REQUEST_BODY_BYTES,
    )
    data_urls = _to_data_urls(encoded)

    return data_urls


def _validate_request_body_size(body: dict) -> None:
    body_size = len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
    if body_size > MAX_REQUEST_BODY_BYTES:
        raise ValueError(
            f"Grok Video 请求体大小 {_format_mb(body_size)} 超过 "
            f"{_format_mb(MAX_REQUEST_BODY_BYTES)} 限制，请减少参考图片或降低图片尺寸。"
        )


class O1keyGrokVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "网络线路": (NETWORK_ROUTE_OPTIONS, {"default": NETWORK_ROUTE_OPTIONS[0]}),
                "模型": (MODEL_OPTIONS, {"default": MODEL_OPTIONS[0]}),
                "宽高比": (ASPECT_RATIO_OPTIONS, {"default": "16:9"}),
                "秒数（按模型限制）": (
                    "INT",
                    {
                        "default": 5,
                        "min": 5,
                        "max": 20,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "画质": (QUALITY_OPTIONS, {"default": "720p"}),
            },
            "optional": {
                "参考图1": ("IMAGE",),
                "参考图2": ("IMAGE",),
                "参考图3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频",)
    FUNCTION = "generate"
    CATEGORY = "comfyui_o1key/Video"

    DESCRIPTION = (
        "Grok Video /v1/videos task node. Supports prompt plus up to "
        "three image references, multiple aspect ratios, model-specific seconds, 720p output."
    )

    def generate(
        self,
        **kwargs,
    ):
        if VideoFromFile is None:
            raise RuntimeError("当前 ComfyUI 版本不支持原生 VIDEO 输入实现 VideoFromFile。")

        提示词 = kwargs.get("提示词", "")
        网络线路 = kwargs.get("网络线路", NETWORK_ROUTE_OPTIONS[0])
        模型 = kwargs.get("模型", MODEL_OPTIONS[0])
        宽高比 = kwargs.get("宽高比", "16:9")
        秒数 = kwargs.get("秒数（按模型限制）", kwargs.get("秒数（≤15s）", kwargs.get("秒数", 5)))
        画质 = kwargs.get("画质", "720p")

        prompt = (提示词 or "").strip()
        if not prompt:
            raise ValueError("提示词不能为空。")
        if 模型 not in MODEL_OPTIONS:
            raise ValueError(f"模型仅支持: {', '.join(MODEL_OPTIONS)}")
        if 宽高比 not in ASPECT_RATIO_OPTIONS:
            raise ValueError(f"宽高比仅支持: {', '.join(ASPECT_RATIO_OPTIONS)}。")
        seconds = int(秒数)
        allowed_seconds = MODEL_SECONDS_OPTIONS.get(模型)
        if allowed_seconds is not None:
            if seconds not in allowed_seconds:
                raise ValueError(
                    f"模型 {模型} 仅支持秒数: "
                    f"{', '.join(str(s) for s in allowed_seconds)}。"
                    "请修改为正确的秒数后再发起请求。"
                )
        elif seconds < 5 or seconds > 15:
            raise ValueError("秒数仅支持 5 到 15。")
        if 画质 not in QUALITY_OPTIONS:
            raise ValueError("画质仅支持 720p。")

        quality = QUALITY_VALUE_MAP[画质]
        reference_images = _collect_reference_images(**kwargs)
        image_data_urls = _encode_image_data_urls(
            reference_images,
            prompt=prompt,
            model=模型,
            aspect_ratio=宽高比,
            seconds=seconds,
            quality=quality,
        )

        request_body = GrokVideoClient.build_video_body(
            prompt=prompt,
            model=模型,
            aspect_ratio=宽高比,
            seconds=seconds,
            quality=quality,
            images=image_data_urls,
        )
        _validate_request_body_size(request_body)

        pbar = ProgressBar(100) if PROGRESS_BAR_AVAILABLE else None
        last_progress = [0]

        def progress_callback(progress: int, status: str, elapsed: float):
            progress_value = max(0, min(100, int(progress or 0)))
            if pbar is not None and progress_value > last_progress[0]:
                pbar.update(progress_value - last_progress[0])
                last_progress[0] = progress_value

        client = GrokVideoClient(base_url=get_base_url_by_route(网络线路))

        try:
            result = client.generate_video_sync(
                prompt=prompt,
                model=模型,
                aspect_ratio=宽高比,
                seconds=seconds,
                quality=quality,
                output_dir=_get_output_dir(),
                images=image_data_urls,
                poll_interval=5,
                timeout=1200,
                progress_callback=progress_callback,
            )

            if pbar is not None and last_progress[0] < 100:
                pbar.update(100 - last_progress[0])

            video_path = result["video_path"]
            print(f"Grok Video：下载完成：{video_path}")
            return (VideoFromFile(video_path),)
        finally:
            try:
                balance_data = client.query_balance_sync()
                balance_info = client.format_balance_info(balance_data)
                print(f"Grok Video：{balance_info}")
            except Exception:
                pass


NODE_CLASS_MAPPINGS = {
    "O1keyGrokVideo": O1keyGrokVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "O1keyGrokVideo": "Grok Video",
}
