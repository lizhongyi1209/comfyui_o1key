"""
Google Veo 视频生成节点
ComfyUI 自定义节点，调用 Veo API 生成视频
"""

import os
import re
import time
from typing import Optional, Tuple

import torch

from ..utils.image_utils import tensor_to_pil
from ..clients.veo_client import VeoClient
from ..models_config import (
    get_enabled_veo_models,
    VEO_MODELS,
    VEO_RESOLUTION_MAP,
)

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

try:
    from comfy.utils import ProgressBar
    PROGRESS_BAR_AVAILABLE = True
except ImportError:
    PROGRESS_BAR_AVAILABLE = False
    print("⚠️ GoogleVeo: comfy.utils.ProgressBar 不可用，将只使用终端进度显示")


def _get_video_output_dir() -> str:
    """获取视频输出目录: ComfyUI/output/video"""
    if FOLDER_PATHS_AVAILABLE:
        base = folder_paths.get_output_directory()
    else:
        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base = os.path.join(os.path.dirname(os.path.dirname(plugin_dir)), "output")
    video_dir = os.path.join(base, "video")
    os.makedirs(video_dir, exist_ok=True)
    return video_dir


def _get_next_counter(directory: str, prefix: str) -> int:
    """扫描目录，获取下一个可用的文件计数器"""
    if not os.path.exists(directory):
        return 1
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)")
    max_counter = 0
    for f in os.listdir(directory):
        m = pattern.match(f)
        if m:
            max_counter = max(max_counter, int(m.group(1)))
    return max_counter + 1


def _fit_image_to_target(image, target_size: str):
    """
    将参考图片按 "等比缩放覆盖 + 居中裁剪" 策略适配到目标分辨率。
    """
    from PIL import Image as PILImage

    parts = target_size.lower().split("x")
    target_w, target_h = int(parts[0]), int(parts[1])

    src_w, src_h = image.size
    src_ratio = src_w / src_h
    target_ratio = target_w / target_h

    if abs(src_ratio - target_ratio) < 0.01 and src_w <= target_w and src_h <= target_h:
        return image

    print(f"Veo: 参考图片 {src_w}x{src_h} (比例 {src_ratio:.2f}) → 目标 {target_w}x{target_h} (比例 {target_ratio:.2f})")

    resample = PILImage.Resampling.LANCZOS if hasattr(PILImage, "Resampling") else PILImage.LANCZOS

    if src_ratio > target_ratio:
        scale = target_h / src_h
        new_w = round(src_w * scale)
        new_h = target_h
        image = image.resize((new_w, new_h), resample=resample)
        left = (new_w - target_w) // 2
        image = image.crop((left, 0, left + target_w, target_h))
    else:
        scale = target_w / src_w
        new_w = target_w
        new_h = round(src_h * scale)
        image = image.resize((new_w, new_h), resample=resample)
        top = (new_h - target_h) // 2
        image = image.crop((0, top, target_w, top + target_h))

    print(f"Veo: 参考图片已适配为 {image.size[0]}x{image.size[1]}")
    return image


def _compress_image_to_bytes(image, target_size: Optional[str] = None) -> bytes:
    """
    将 PIL Image 适配目标分辨率并编码为 PNG 字节
    """
    from io import BytesIO

    if image.mode != "RGB":
        image = image.convert("RGB")

    if target_size:
        image = _fit_image_to_target(image, target_size)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    size_kb = buffered.tell() / 1024
    print(f"Veo: 参考图片编码为 PNG，{size_kb:.0f} KB ({image.size[0]}x{image.size[1]})")
    return buffered.getvalue()


class GoogleVeo:
    """
    Google Veo 视频生成节点

    功能：
    - 文生视频：基于提示词生成视频
    - 图生视频：基于首帧/尾帧/参考图生成视频
    - 异步轮询：自动等待生成完成并下载
    """

    def __init__(self):
        self.client = None

    @classmethod
    def INPUT_TYPES(cls):
        enabled_models = get_enabled_veo_models()
        if not enabled_models:
            enabled_models = ["请在 models_config.py 中启用 Veo 模型"]

        # 分辨率选项
        resolution_options = ["720p", "1080p", "4K"]

        # 宽高比选项
        aspect_ratio_options = ["16:9", "9:16"]

        # 视频秒数选项
        seconds_options = ["4", "6", "8"]

        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "A calico cat playing a piano on stage",
                    "multiline": True,
                }),
                "模型": (enabled_models, {
                    "default": enabled_models[0] if enabled_models else "Veo3.1",
                }),
                "分辨率": (resolution_options, {
                    "default": "720p",
                }),
                "宽高比": (aspect_ratio_options, {
                    "default": "9:16",
                }),
                "视频时长": (seconds_options, {
                    "default": "8",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "生成数量": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                }),
            },
            "optional": {
                "首帧": ("IMAGE",),
                "尾帧": ("IMAGE",),
                "参考图": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("预览视频",)
    FUNCTION = "generate_video"
    CATEGORY = "video/generation"

    DESCRIPTION = (
        "Google Veo 视频生成节点。\n"
        "支持文生视频和图生视频（图生视频支持首帧、尾帧、参考图）。\n"
        "视频保存到 ComfyUI/output/video/ 目录。\n\n"
        "【模型说明】\n"
        "• Veo3.1：Google 最新视频生成模型\n\n"
        "【分辨率说明】\n"
        "• 720p：标清\n"
        "• 1080p：高清\n"
        "• 4K：超高清\n\n"
        "【时长说明】\n"
        "• 4秒：短视频\n"
        "• 6秒：标准\n"
        "• 8秒：长视频（默认）\n\n"
        "【图生视频说明】\n"
        "• 首帧：视频开始的第一帧图像\n"
        "• 尾帧：视频结束时的最后一帧图像\n"
        "• 参考图：参考图像（与首帧/尾帧配合使用）\n"
        "• 至少需要提供首帧或参考图之一"
    )

    def generate_video(
        self,
        prompt: str,
        模型: str,
        **kwargs,
    ) -> Tuple[str]:
        分辨率 = kwargs.pop("分辨率", "720p")
        宽高比 = kwargs.pop("宽高比", "9:16")
        视频时长 = kwargs.pop("视频时长", "8")
        seed = kwargs.pop("seed", 0)
        生成数量 = kwargs.pop("生成数量", 1)
        start_time = time.time()

        # 解析视频时长
        seconds = int(视频时长)

        # 解析分辨率和宽高比，映射到模型名称
        size_key = f"{分辨率}_{宽高比}"
        actual_size = VEO_RESOLUTION_MAP.get(size_key)
        if not actual_size:
            # 默认值
            actual_size = "720x1280"  # 720p 9:16

        # 检查是否有参考图输入
        首帧 = kwargs.get("首帧")
        尾帧 = kwargs.get("尾帧")
        参考图 = kwargs.get("参考图")

        has_image = 首帧 is not None or 尾帧 is not None or 参考图 is not None

        # 根据是否有图片选择模型前缀
        if has_image:
            model_prefix = "veo3.1"
        else:
            model_prefix = "veo3.1"

        # 构建完整模型名称
        # 格式: veo3.1-portrait / veo3.1-landscape / veo3.1-portrait-fl / veo3.1-landscape-fl 等
        if 分辨率 == "720p":
            res_suffix = ""
            if 宽高比 == "9:16":
                orientation = "portrait"
            else:
                orientation = "landscape"
        elif 分辨率 == "1080p":
            res_suffix = "-hd"
            if 宽高比 == "9:16":
                orientation = "portrait"
            else:
                orientation = "landscape"
        else:  # 4K
            res_suffix = "-4k"
            if 宽高比 == "9:16":
                orientation = "portrait"
            else:
                orientation = "landscape"

        # 图生视频添加 -fl 后缀
        if has_image:
            model_suffix = f"-{orientation}-fl{res_suffix}"
        else:
            model_suffix = f"-{orientation}{res_suffix}"

        model = f"{model_prefix}{model_suffix}"

        # 准备图片字节
        first_frame_bytes = None
        last_frame_bytes = None
        reference_bytes = None

        if 首帧 is not None:
            pil_images = tensor_to_pil(首帧)
            if pil_images:
                first_frame_bytes = _compress_image_to_bytes(pil_images[0], target_size=actual_size)

        if 尾帧 is not None:
            pil_images = tensor_to_pil(尾帧)
            if pil_images:
                last_frame_bytes = _compress_image_to_bytes(pil_images[0], target_size=actual_size)

        if 参考图 is not None:
            pil_images = tensor_to_pil(参考图)
            if pil_images:
                reference_bytes = _compress_image_to_bytes(pil_images[0], target_size=actual_size)

        mode_str = "图生视频" if has_image else "文生视频"
        print(f"Veo: {mode_str} | 并发{生成数量}个 | 模型: {model} | {seconds}秒 | {分辨率} {宽高比}")

        # 准备保存路径
        video_dir = _get_video_output_dir()
        counter = _get_next_counter(video_dir, "veo")

        # ProgressBar
        pbar = None
        if PROGRESS_BAR_AVAILABLE:
            pbar = ProgressBar(生成数量 if 生成数量 > 1 else 100)

        try:
            if self.client is None:
                self.client = VeoClient()

            if 生成数量 == 1:
                save_path = os.path.join(video_dir, f"veo_{counter:05d}.mp4")
                last_progress = [0]

                def progress_callback(progress_pct: int):
                    print(
                        f"\rVeo: 生成中... 进度: {progress_pct}%",
                        end="", flush=True
                    )
                    if pbar is not None and progress_pct > last_progress[0]:
                        pbar.update(progress_pct - last_progress[0])
                        last_progress[0] = progress_pct

                def on_stage(stage: str):
                    if stage == "submitting":
                        print("Veo: 正在提交视频生成任务...")
                    elif stage.startswith("submitted:"):
                        vid = stage.split(":", 1)[1]
                        print(f"Veo: 视频任务已提交，ID: {vid}")
                    elif stage == "polling":
                        print("Veo: 等待视频生成...")
                    elif stage == "downloading":
                        print("")
                        print("Veo: 视频生成完成，正在下载...")

                result_path = self.client.generate_video_sync(
                    prompt=prompt,
                    model=model,
                    seconds=seconds,
                    size=actual_size,
                    save_path=save_path,
                    first_frame_bytes=first_frame_bytes,
                    last_frame_bytes=last_frame_bytes,
                    reference_bytes=reference_bytes,
                    seed=seed,
                    progress_callback=progress_callback,
                    on_stage=on_stage,
                )
                result_paths = [result_path]

            else:
                save_paths = [
                    os.path.join(video_dir, f"veo_{counter + i:05d}.mp4")
                    for i in range(生成数量)
                ]
                success_count = [0]

                def batch_progress_callback(current: int, total: int, success: bool, error_msg):
                    if success:
                        success_count[0] += 1
                        print(f"Veo: 第 {current}/{total} 个视频完成 ✓")
                    else:
                        print(f"Veo: 第 {current}/{total} 个视频失败 ✗")
                        if error_msg:
                            print(f"原始错误详情:\n{error_msg}")
                    if pbar is not None:
                        pbar.update(1)

                print(f"Veo: 正在并发提交 {生成数量} 个视频任务，请耐心等待...")
                result_paths = self.client.generate_batch_videos_sync(
                    prompt=prompt,
                    model=model,
                    seconds=seconds,
                    size=actual_size,
                    save_paths=save_paths,
                    first_frame_bytes=first_frame_bytes,
                    last_frame_bytes=last_frame_bytes,
                    reference_bytes=reference_bytes,
                    seed=seed,
                    progress_callback=batch_progress_callback,
                )

            elapsed = time.time() - start_time
            time_str = f"{elapsed:.2f}s" if elapsed >= 1 else f"{elapsed:.3f}s"
            print(f"Veo: 完成！总耗时 {time_str} | 已生成 {len(result_paths)} 个视频")
            for p in result_paths:
                print(f"  → {p}")

            output_path = "\n".join(result_paths)
            return (output_path,)

        except ValueError as e:
            error_msg = str(e)
            print(f"\nVeo: ❌ {error_msg}")
            raise ValueError(error_msg) from None

        except RuntimeError as e:
            error_msg = str(e)
            print(f"\nVeo: ❌ {error_msg}")
            raise RuntimeError(error_msg) from None

        except Exception as e:
            error_msg = str(e)
            print(f"\nVeo: ❌ {error_msg}")
            raise type(e)(error_msg) from None

        finally:
            if self.client is not None:
                try:
                    balance_data = self.client.query_balance_sync()
                    balance_info = self.client.format_balance_info(balance_data)
                    print(f"Veo: {balance_info}")
                except Exception:
                    pass


NODE_CLASS_MAPPINGS = {
    "GoogleVeo": GoogleVeo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleVeo": "Google Veo - ab",
}
