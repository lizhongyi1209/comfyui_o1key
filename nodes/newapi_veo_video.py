"""
Single-node new-api Veo 3.1 generator.

The node submits a /v1/videos task, waits for completion, downloads the mp4,
and returns ComfyUI's native VIDEO object for the built-in Save Video node.
"""

import os
from io import BytesIO
from typing import Optional, Tuple

from ..clients.newapi_veo_client import NewAPIVeoClient
from ..utils.image_utils import tensor_to_pil
from ..utils.config import NETWORK_ROUTE_OPTIONS, get_base_url_by_route

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
    VideoFromFile = None


MODEL_OPTIONS = [
    "veo-3.1",
]

DURATION_OPTIONS = ["4", "6", "8"]
ASPECT_RATIO_OPTIONS = ["16:9", "9:16"]
RESOLUTION_OPTIONS = ["720p", "1080p"]

TARGET_SIZE_MAP = {
    ("720p", "16:9"): (1280, 720),
    ("720p", "9:16"): (720, 1280),
    ("1080p", "16:9"): (1920, 1080),
    ("1080p", "9:16"): (1080, 1920),
}


def _get_output_dir() -> str:
    if FOLDER_PATHS_AVAILABLE:
        return folder_paths.get_output_directory()

    plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    comfy_root = os.path.dirname(os.path.dirname(plugin_dir))
    return os.path.join(comfy_root, "output")


def _get_download_dir() -> str:
    output_dir = _get_output_dir()
    video_dir = os.path.join(output_dir, "newapi_veo")
    os.makedirs(video_dir, exist_ok=True)
    return video_dir


def _fit_image_to_target(image, target_size: Tuple[int, int]):
    from PIL import Image as PILImage

    target_w, target_h = target_size
    src_w, src_h = image.size
    src_ratio = src_w / src_h
    target_ratio = target_w / target_h

    if src_w == target_w and src_h == target_h:
        return image

    resample = PILImage.Resampling.LANCZOS if hasattr(PILImage, "Resampling") else PILImage.LANCZOS

    if src_ratio > target_ratio:
        scale = target_h / src_h
        new_w = round(src_w * scale)
        image = image.resize((new_w, target_h), resample=resample)
        left = max(0, (new_w - target_w) // 2)
        image = image.crop((left, 0, left + target_w, target_h))
    else:
        scale = target_w / src_w
        new_h = round(src_h * scale)
        image = image.resize((target_w, new_h), resample=resample)
        top = max(0, (new_h - target_h) // 2)
        image = image.crop((0, top, target_w, top + target_h))

    return image


def _image_to_png_bytes(image_tensor, resolution: str, aspect_ratio: str) -> Optional[bytes]:
    if image_tensor is None:
        return None

    pil_images = tensor_to_pil(image_tensor)
    if not pil_images:
        return None

    image = pil_images[0]
    if image.mode != "RGB":
        image = image.convert("RGB")

    target_size = TARGET_SIZE_MAP.get((resolution, aspect_ratio))
    if target_size is not None:
        original_size = image.size
        image = _fit_image_to_target(image, target_size)
        if image.size != original_size:
            print(
                "NewAPI Veo: input image fitted "
                f"{original_size[0]}x{original_size[1]} -> {image.size[0]}x{image.size[1]}"
            )

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    print(
        "NewAPI Veo: input_reference PNG "
        f"{len(image_bytes) / 1024:.0f} KB ({image.size[0]}x{image.size[1]})"
    )
    return image_bytes


class Google31Video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": (
                    "STRING",
                    {
                        "default": "A cinematic shot of a small robot walking through a rainy neon street.",
                        "multiline": True,
                    },
                ),
                "负向提示词": ("STRING", {"default": "", "multiline": True}),
                "网络线路": (NETWORK_ROUTE_OPTIONS, {"default": "全球加速"}),
                "模型": (MODEL_OPTIONS, {"default": MODEL_OPTIONS[0]}),
                "时长": (DURATION_OPTIONS, {"default": "8"}),
                "宽高比": (ASPECT_RATIO_OPTIONS, {"default": "16:9"}),
                "分辨率": (RESOLUTION_OPTIONS, {"default": "1080p"}),
                "生成音频": (["打开", "关闭"], {"default": "打开"}),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "参考图像": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频",)
    FUNCTION = "generate"
    CATEGORY = "comfyui_o1key/Video"

    DESCRIPTION = (
        "Submit a new-api /v1/videos Veo 3.1 task, poll until complete, "
        "download the mp4, and output native VIDEO for ComfyUI Save Video."
    )

    def generate(
        self,
        提示词: str,
        负向提示词: str,
        网络线路: str,
        模型: str,
        时长: str,
        宽高比: str,
        分辨率: str,
        生成音频: str,
        seed: int,
        参考图像=None,
    ):
        if VideoFromFile is None:
            raise RuntimeError("当前 ComfyUI 版本不支持原生 VIDEO 输入实现 VideoFromFile。")

        prompt = (提示词 or "").strip()
        if not prompt:
            raise ValueError("提示词不能为空。")

        duration_value = int(时长)
        if duration_value not in (4, 6, 8):
            raise ValueError("时长仅支持 4、6、8。")
        if 宽高比 not in ASPECT_RATIO_OPTIONS:
            raise ValueError("宽高比仅支持 16:9 或 9:16。")
        if 分辨率 not in RESOLUTION_OPTIONS:
            raise ValueError("分辨率仅支持 720p 或 1080p。")

        output_dir = _get_download_dir()
        image_bytes = _image_to_png_bytes(参考图像, 分辨率, 宽高比)

        pbar = ProgressBar(100) if PROGRESS_BAR_AVAILABLE else None
        last_progress = [0]
        last_status = [""]

        def progress_callback(progress: int, status: str, elapsed: float):
            if status != last_status[0]:
                print(
                    "NewAPI Veo: polling "
                    f"status={status} | elapsed={elapsed:.0f}s"
                )
                last_status[0] = status

            progress = max(0, min(100, int(progress or 0)))
            if pbar is not None and progress > last_progress[0]:
                pbar.update(progress - last_progress[0])
                last_progress[0] = progress

        client = NewAPIVeoClient(base_url=get_base_url_by_route(网络线路))

        result = client.generate_video_sync(
            prompt=prompt,
            model=模型,
            duration=duration_value,
            aspect_ratio=宽高比,
            resolution=分辨率,
            output_dir=output_dir,
            negative_prompt=负向提示词,
            generate_audio=(生成音频 == "打开"),
            image_bytes=image_bytes,
            poll_interval=10,
            timeout=900,
            progress_callback=progress_callback,
        )

        video_path = result["video_path"]
        video = VideoFromFile(video_path)

        print(
            "NewAPI Veo: completed "
            f"| task_id={result['task_id']} | video={video_path}"
        )

        return (video,)


NODE_CLASS_MAPPINGS = {
    "Google31Video": Google31Video,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Google31Video": "Google 3.1 Video",
}
