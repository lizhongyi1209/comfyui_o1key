"""
Sora 视频生成节点
ComfyUI 自定义节点，调用 Sora API 生成视频
"""

import os
import re
import time
from typing import Optional, Tuple

import torch

from ..utils.image_utils import tensor_to_pil
from ..clients.sora_client import SoraClient
from ..models_config import (
    get_enabled_sora_models,
    get_all_sora_seconds,
    get_all_sora_sizes,
    get_sora_supported_seconds,
    get_sora_supported_sizes,
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
    print("⚠️ SoraVideo: comfy.utils.ProgressBar 不可用，将只使用终端进度显示")


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

    策略 (Cover Crop)：
    1. 比较图片宽高比和目标宽高比
    2. 等比缩放，使图片最短边刚好覆盖目标对应边（图片完全覆盖目标区域）
    3. 居中裁剪多余部分，得到精确目标尺寸

    Args:
        image: PIL Image 对象
        target_size: 目标分辨率字符串，格式 "WxH"（如 "720x1280"）

    Returns:
        适配后的 PIL Image 对象
    """
    from PIL import Image as PILImage

    # 解析目标尺寸
    parts = target_size.lower().split("x")
    target_w, target_h = int(parts[0]), int(parts[1])

    src_w, src_h = image.size
    src_ratio = src_w / src_h
    target_ratio = target_w / target_h

    # 宽高比一致且尺寸不超过目标，无需处理
    if abs(src_ratio - target_ratio) < 0.01 and src_w <= target_w and src_h <= target_h:
        return image

    print(f"Sora: 参考图片 {src_w}x{src_h} (比例 {src_ratio:.2f}) → 目标 {target_w}x{target_h} (比例 {target_ratio:.2f})")

    # 获取高质量重采样滤波器
    resample = PILImage.Resampling.LANCZOS if hasattr(PILImage, "Resampling") else PILImage.LANCZOS

    # Cover Crop: 缩放使图片完全覆盖目标区域，然后居中裁剪
    if src_ratio > target_ratio:
        # 图片更宽：以高度为基准缩放，裁左右
        scale = target_h / src_h
        new_w = round(src_w * scale)
        new_h = target_h
        image = image.resize((new_w, new_h), resample=resample)
        # 居中裁剪宽度
        left = (new_w - target_w) // 2
        image = image.crop((left, 0, left + target_w, target_h))
    else:
        # 图片更高（或一样）：以宽度为基准缩放，裁上下
        scale = target_w / src_w
        new_w = target_w
        new_h = round(src_h * scale)
        image = image.resize((new_w, new_h), resample=resample)
        # 居中裁剪高度
        top = (new_h - target_h) // 2
        image = image.crop((0, top, target_w, top + target_h))

    print(f"Sora: 参考图片已适配为 {image.size[0]}x{image.size[1]}")
    return image


def _compress_image_for_upload(
    image,
    target_size: Optional[str] = None,
) -> bytes:
    """
    将 PIL Image 适配目标分辨率并编码为 PNG 字节，用于上传。

    ============================================================
    ⚠️  已验证可用的标准做法，请勿随意修改以下编码逻辑！
    ============================================================
    经过多轮调试（2026-02-28），以下参数组合为唯一验证成功的方案：

    1. 图片格式：PNG（format="PNG"）
       - 不可改为 JPEG —— API 会校验 Content-Type，抓包确认服务端使用 image/png
       - 不可使用 base64 字符串 —— 会报 "expected a file, got a string"
       - 不可使用 data URI —— 服务端不识别，返回 500

    2. 图片尺寸：必须与视频分辨率完全一致（target_size）
       - 不可缩放降采样 —— 会报 "Inpaint image must match the requested width and height"
       - 尺寸由 _fit_image_to_target() 保证（等比缩放 + 居中裁剪）

    3. 上传方式：由调用方（sora_client.py）以 multipart/form-data 文件字段上传
       - filename="reference.png", content_type="image/png"
       - 不可改回 application/json —— 服务端校验 input_reference 必须为 file 类型
    ============================================================

    Args:
        image: PIL Image 对象
        target_size: 目标分辨率字符串 "WxH"（如 "720x1280"）

    Returns:
        PNG 格式的二进制字节
    """
    from io import BytesIO

    # 统一转换为 RGB（去除透明通道及其他模式）
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 适配到目标分辨率（等比缩放 + 居中裁剪）
    # ⚠️ 必须保持此尺寸不变，API 强制要求参考图片与视频分辨率完全一致
    if target_size:
        image = _fit_image_to_target(image, target_size)

    # ⚠️ 必须使用 PNG 格式，不可改为 JPEG 或其他格式
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    size_kb = buffered.tell() / 1024
    print(f"Sora: 参考图片编码为 PNG，{size_kb:.0f} KB ({image.size[0]}x{image.size[1]})")
    return buffered.getvalue()


class SoraVideo:
    """
    Sora 视频生成节点

    功能：
    - 文生视频：基于提示词生成视频
    - 图生视频：基于参考图片和提示词生成视频
    - 异步轮询：自动等待生成完成并下载
    """

    def __init__(self):
        self.client = None

    @classmethod
    def INPUT_TYPES(cls):
        enabled_models = get_enabled_sora_models()
        if not enabled_models:
            enabled_models = ["请在 models_config.py 中启用至少一个 Sora 模型"]

        all_seconds = get_all_sora_seconds()
        seconds_str = [str(s) for s in all_seconds] if all_seconds else ["4", "8", "12"]

        all_sizes = get_all_sora_sizes()
        if not all_sizes:
            all_sizes = ["720x1280", "1280x720", "1024x1792", "1792x1024"]

        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "A calico cat playing a piano on stage",
                    "multiline": True,
                }),
                "模型": (enabled_models, {
                    "default": enabled_models[0],
                }),
                "视频时长（秒）": (seconds_str, {
                    "default": "4",
                }),
                "分辨率": (all_sizes, {
                    "default": all_sizes[0],
                }),
                "生成数量": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            },
            "optional": {
                "参考图片": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("预览视频",)
    FUNCTION = "generate_video"
    CATEGORY = "video/generation"

    DESCRIPTION = (
        "Sora 视频生成节点。\n"
        "支持文生视频和图生视频，自动轮询任务状态并下载视频。\n"
        "视频保存到 ComfyUI/output/video/ 目录。"
    )

    def generate_video(
        self,
        prompt: str,
        模型: str,
        **kwargs,
    ) -> Tuple[str]:
        视频时长 = kwargs.pop("视频时长（秒）", "4")
        分辨率 = kwargs.pop("分辨率", "720x1280")
        生成数量 = kwargs.pop("生成数量", 1)
        seed = kwargs.pop("seed", 0)
        start_time = time.time()
        seconds = int(视频时长)

        # 检查参考图片
        ref_image = kwargs.get("参考图片")
        ref_image_bytes = None
        if ref_image is not None:
            pil_images = tensor_to_pil(ref_image)
            if pil_images:
                ref_image_bytes = _compress_image_for_upload(pil_images[0], target_size=分辨率)

        mode_str = "图生视频 (含参考图)" if ref_image_bytes else "文生视频"
        if 生成数量 > 1:
            print(f"Sora: {mode_str} | 并发{生成数量}个 | {模型} | {seconds}秒 | {分辨率}")
        else:
            print(f"Sora: {mode_str} | {模型} | {seconds}秒 | {分辨率}")

        # 校验参数兼容性
        supported_seconds = get_sora_supported_seconds(模型)
        if supported_seconds and seconds not in supported_seconds:
            raise ValueError(
                f"时长 {seconds}s 与模型 \"{模型}\" 不兼容！"
                f"支持的时长: {', '.join(str(s) for s in supported_seconds)}"
            )
        supported_sizes = get_sora_supported_sizes(模型)
        if supported_sizes and 分辨率 not in supported_sizes:
            raise ValueError(
                f"分辨率 \"{分辨率}\" 与模型 \"{模型}\" 不兼容！"
                f"支持的分辨率: {', '.join(supported_sizes)}"
            )

        # 准备保存路径
        video_dir = _get_video_output_dir()
        counter = _get_next_counter(video_dir, "sora")

        # ProgressBar
        pbar = None
        if PROGRESS_BAR_AVAILABLE:
            pbar = ProgressBar(生成数量 if 生成数量 > 1 else 100)

        try:
            if self.client is None:
                self.client = SoraClient()

            if 生成数量 == 1:
                # ── 单个视频：保留详细进度（提交→轮询→下载）
                save_path = os.path.join(video_dir, f"sora_{counter:05d}.mp4")
                last_progress = [0]

                def progress_callback(progress_pct: int, elapsed: float):
                    elapsed_str = f"{int(elapsed)}s"
                    print(
                        f"\rSora: 生成中... 进度: {progress_pct}% (已等待 {elapsed_str})",
                        end="", flush=True
                    )
                    if pbar is not None and progress_pct > last_progress[0]:
                        pbar.update(progress_pct - last_progress[0])
                        last_progress[0] = progress_pct

                def on_stage(stage: str):
                    if stage == "submitting":
                        print("Sora: 正在提交视频生成任务...")
                    elif stage.startswith("submitted:"):
                        vid = stage.split(":", 1)[1]
                        print(f"Sora: 视频任务已提交，ID: {vid}")
                    elif stage == "polling":
                        print("Sora: 等待视频生成...")
                    elif stage == "downloading":
                        print("")  # 换行（结束 \r 行）
                        print("Sora: 视频生成完成，正在下载...")

                result_path = self.client.generate_video_sync(
                    prompt=prompt,
                    model=模型,
                    seconds=seconds,
                    size=分辨率,
                    save_path=save_path,
                    input_reference_bytes=ref_image_bytes,
                    seed=seed,
                    progress_callback=progress_callback,
                    on_stage=on_stage,
                )
                result_paths = [result_path]

            else:
                # ── 批量并发：同时提交多个任务
                save_paths = [
                    os.path.join(video_dir, f"sora_{counter + i:05d}.mp4")
                    for i in range(生成数量)
                ]
                success_count = [0]
                fail_count = [0]

                def batch_progress_callback(current: int, total: int, success: bool, error_msg):
                    if success:
                        success_count[0] += 1
                        print(f"Sora: 第 {current}/{total} 个视频完成 ✓")
                    else:
                        fail_count[0] += 1
                        print(f"Sora: 第 {current}/{total} 个视频失败 ✗ ({error_msg})")
                    if pbar is not None:
                        pbar.update(1)

                print(f"Sora: 正在并发提交 {生成数量} 个视频任务，请耐心等待...")
                result_paths = self.client.generate_batch_videos_sync(
                    prompt=prompt,
                    model=模型,
                    seconds=seconds,
                    size=分辨率,
                    save_paths=save_paths,
                    input_reference_bytes=ref_image_bytes,
                    seed=seed,
                    progress_callback=batch_progress_callback,
                )

            elapsed = time.time() - start_time
            time_str = f"{elapsed:.2f}s" if elapsed >= 1 else f"{elapsed:.3f}s"
            print(f"Sora: 完成！总耗时 {time_str} | 已生成 {len(result_paths)} 个视频")
            for p in result_paths:
                print(f"  → {p}")

            output_path = "\n".join(result_paths)
            return (output_path,)

        except ValueError as e:
            error_msg = str(e).split("\n")[0]
            print(f"\nSora: ❌ {error_msg}")
            raise ValueError(error_msg) from None

        except RuntimeError as e:
            error_msg = str(e).split("\n")[0]
            print(f"\nSora: ❌ {error_msg}")
            raise RuntimeError(error_msg) from None

        except Exception as e:
            error_msg = str(e).split("\n")[0]
            print(f"\nSora: ❌ {error_msg}")
            raise type(e)(error_msg) from None

        finally:
            if self.client is not None:
                try:
                    balance_data = self.client.query_balance_sync()
                    balance_info = self.client.format_balance_info(balance_data)
                    print(f"Sora: {balance_info}")
                except Exception:
                    pass
