"""
Sora 视频生成节点
ComfyUI 自定义节点，调用 Sora API 生成视频
"""

import os
import re
import time
from math import gcd
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
    get_sora_seconds_with_labels,
    get_sora_sizes_with_labels,
    SORA_MODELS,
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


def _size_to_display(size: str) -> str:
    """
    将 'WxH' 格式的分辨率转换为友好显示名。

    例如：
        "720x1280"  → "720P  9:16"
        "1280x720"  → "720P 16:9"
        "1024x1792" → "1K  4:7"
        "1792x1024" → "1K  7:4"

    Args:
        size: 分辨率字符串，格式 "WxH"

    Returns:
        友好显示名字符串
    """
    parts = size.lower().split("x")
    w, h = int(parts[0]), int(parts[1])
    short_side = min(w, h)
    if short_side >= 3840:
        res = "4K"
    elif short_side >= 1920:
        res = "2K"
    elif short_side >= 1080:
        res = "1K"
    elif short_side >= 720:
        res = "720P"
    elif short_side >= 480:
        res = "480P"
    else:
        res = f"{short_side}P"
    g = gcd(w, h)
    ratio = f"{w // g}:{h // g}"
    return f"{res} {ratio} ({size})"


def _build_size_display_map(sizes: list) -> dict:
    """
    构建 显示名 → 实际值 映射字典。

    Args:
        sizes: 实际分辨率列表，如 ["720x1280", "1280x720"]

    Returns:
        字典，key 为显示名，value 为实际分辨率字符串
    """
    mapping = {}
    for size in sizes:
        display = _size_to_display(size)
        if display in mapping:
            # 极少数情况下防止重名
            display = f"{display} ({size})"
        mapping[display] = size
    return mapping


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
        from ..models_config import SECONDS_DISPLAY_MAP, RESOLUTION_DISPLAY_MAP
        
        enabled_models = get_enabled_sora_models()
        if not enabled_models:
            enabled_models = ["请在 models_config.py 中启用至少一个 Sora 模型"]

        # 构建秒数选项列表（按数字顺序排序）
        # 格式: ["4", "8", "10", "12", "15", "25(pro)"]
        all_seconds_display = []
        seen_seconds = set()
        for model_id in enabled_models:
            supported = get_sora_supported_seconds(model_id)
            for s in supported:
                if s not in seen_seconds:
                    seen_seconds.add(s)
                    display = SECONDS_DISPLAY_MAP.get(s, str(s))
                    all_seconds_display.append((s, display))
        # 按秒数数值排序
        all_seconds_display = sorted(all_seconds_display, key=lambda x: x[0])
        seconds_options = [d for _, d in all_seconds_display] if all_seconds_display else ["4", "8", "12"]

        # 构建分辨率选项列表（去重）
        # 格式: ["720P", "1080P"]
        seen_resolutions = set()
        for model_id in enabled_models:
            supported = get_sora_supported_sizes(model_id)
            for size in supported:
                if size in RESOLUTION_DISPLAY_MAP:
                    res_name, _ = RESOLUTION_DISPLAY_MAP[size]
                    seen_resolutions.add(res_name)
        resolution_options = sorted(list(seen_resolutions)) if seen_resolutions else ["720P"]

        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "A calico cat playing a piano on stage",
                    "multiline": True,
                }),
                "模型": (enabled_models, {
                    "default": enabled_models[0],
                }),
                "分辨率": (resolution_options, {
                    "default": resolution_options[0] if resolution_options else "720P",
                }),
                "宽高比": (["竖屏", "横屏"], {
                    "default": "竖屏",
                }),
                "视频时长": (seconds_options, {
                    "default": seconds_options[0] if seconds_options else "4",
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
        "视频保存到 ComfyUI/output/video/ 目录。\n\n"
        "【模型说明】\n"
        "• sora-2：官方模型，支持 4/8/12秒、720P 分辨率\n"
        "• sora-2-pro：增强模型，支持全时长（含25秒）、1080P 分辨率\n\n"
        "【时长说明】\n"
        "• 25(pro)：仅 sora-2-pro 支持的25秒时长\n\n"
        "【分辨率说明】\n"
        "• 720P：sora-2 和 sora-2-pro 均支持\n"
        "• 1080P：仅 sora-2-pro 支持的高清分辨率"
    )

    def generate_video(
        self,
        prompt: str,
        模型: str,
        **kwargs,
    ) -> Tuple[str]:
        from ..models_config import SECONDS_DISPLAY_MAP, RESOLUTION_DISPLAY_MAP
        
        视频时长_display = kwargs.pop("视频时长", "4")
        分辨率_display = kwargs.pop("分辨率", "720P")
        宽高比 = kwargs.pop("宽高比", "竖屏")
        生成数量 = kwargs.pop("生成数量", 1)
        seed = kwargs.pop("seed", 0)
        start_time = time.time()

        # 解析秒数显示值（如 "25(pro)" → 25）
        seconds = 4  # 默认
        for actual, display in SECONDS_DISPLAY_MAP.items():
            if display == 视频时长_display:
                seconds = actual
                break
        # 如果找不到映射，尝试直接解析数字
        if seconds == 4 and 视频时长_display != "4":
            try:
                seconds = int(视频时长_display.replace("(pro)", ""))
            except ValueError:
                seconds = 4

        # 根据分辨率和宽高比确定实际分辨率值
        分辨率 = "720x1280"  # 默认
        for actual, (res_name, orientation) in RESOLUTION_DISPLAY_MAP.items():
            if res_name == 分辨率_display and orientation == 宽高比:
                分辨率 = actual
                break

        # 检查参考图片
        ref_image = kwargs.get("参考图片")
        ref_image_bytes = None
        if ref_image is not None:
            pil_images = tensor_to_pil(ref_image)
            if pil_images:
                ref_image_bytes = _compress_image_for_upload(pil_images[0], target_size=分辨率)

        mode_str = "图生视频 (含参考图)" if ref_image_bytes else "文生视频"
        # 获取用户友好的显示值用于日志
        seconds_display = SECONDS_DISPLAY_MAP.get(seconds, str(seconds))
        res_display = f"{分辨率_display} {宽高比}"
        if 生成数量 > 1:
            print(f"Sora: {mode_str} | 并发{生成数量}个 | {模型} | {seconds_display} | {res_display}")
        else:
            print(f"Sora: {mode_str} | {模型} | {seconds_display} | {res_display}")

        # 校验参数兼容性
        supported_seconds = get_sora_supported_seconds(模型)
        if supported_seconds and seconds not in supported_seconds:
            # 构建带标签的支持时长列表
            supported_labels = []
            for s in supported_seconds:
                display = SECONDS_DISPLAY_MAP.get(s, str(s))
                supported_labels.append(display)
            raise ValueError(
                f"时长 {SECONDS_DISPLAY_MAP.get(seconds, str(seconds))} 与模型 \"{模型}\" 不兼容！\n"
                f"该模型支持的时长: {', '.join(supported_labels)}"
            )
        
        supported_sizes = get_sora_supported_sizes(模型)
        if supported_sizes and 分辨率 not in supported_sizes:
            # 检查该分辨率是否为Pro独占
            pro_only_sizes = ["1024x1792", "1792x1024"]
            _, orientation = RESOLUTION_DISPLAY_MAP.get(分辨率, (分辨率, ""))
            extra_hint = f"\n提示：1080P {orientation} 为 sora-2-pro 独占，请切换模型或选择720P。" if 分辨率 in pro_only_sizes else ""
            raise ValueError(
                f"分辨率 \"{分辨率_display} {宽高比}\" 与模型 \"{模型}\" 不兼容！"
                f"支持的分辨率: {', '.join(supported_sizes)}" + extra_hint
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

                def progress_callback(progress_pct: int):
                    print(
                        f"\rSora: 生成中... 进度: {progress_pct}%",
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
                        print(f"Sora: 第 {current}/{total} 个视频失败 ✗")
                        if error_msg:
                            print(f"原始错误详情:\n{error_msg}")
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
            error_msg = str(e)
            print(f"\nSora: ❌ {error_msg}")
            raise ValueError(error_msg) from None

        except RuntimeError as e:
            error_msg = str(e)
            print(f"\nSora: ❌ {error_msg}")
            raise RuntimeError(error_msg) from None

        except Exception as e:
            error_msg = str(e)
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
