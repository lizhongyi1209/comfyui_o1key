"""
Sora 视频生成节点
ComfyUI 自定义节点，调用 Sora API 生成视频
"""

import os
import re
import time
from typing import Optional, Tuple

import torch

from ..utils.image_utils import tensor_to_pil, encode_image_to_base64
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
                "视频时长": (seconds_str, {
                    "default": "4",
                }),
                "分辨率": (all_sizes, {
                    "default": all_sizes[0],
                }),
            },
            "optional": {
                "参考图片": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
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
        视频时长: str,
        分辨率: str,
        **kwargs,
    ) -> Tuple[str]:
        start_time = time.time()
        seconds = int(视频时长)

        # 检查参考图片
        ref_image = kwargs.get("参考图片")
        ref_base64 = None
        if ref_image is not None:
            pil_images = tensor_to_pil(ref_image)
            if pil_images:
                ref_base64 = encode_image_to_base64(pil_images[0])

        mode_str = f"图生视频 (含参考图)" if ref_base64 else "文生视频"
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
        filename = f"sora_{counter:05d}.mp4"
        save_path = os.path.join(video_dir, filename)

        # ProgressBar: 100 步表示百分比
        pbar = None
        if PROGRESS_BAR_AVAILABLE:
            pbar = ProgressBar(100)
        last_progress = 0

        def progress_callback(progress_pct: int, elapsed: float):
            nonlocal last_progress
            elapsed_str = f"{int(elapsed)}s"
            print(f"\rSora: 生成中... 进度: {progress_pct}% (已等待 {elapsed_str})", end="", flush=True)
            if pbar is not None and progress_pct > last_progress:
                pbar.update(progress_pct - last_progress)
                last_progress = progress_pct

        video_id_holder = [None]

        def on_stage(stage: str):
            if stage == "submitting":
                print("Sora: 正在提交视频生成任务...")
            elif stage.startswith("submitted:"):
                vid = stage.split(":", 1)[1]
                video_id_holder[0] = vid
                print(f"Sora: 视频任务已提交，ID: {vid}")
            elif stage == "polling":
                print("Sora: 等待视频生成...")
            elif stage == "downloading":
                print("")  # 换行（结束 \r 行）
                print("Sora: 视频生成完成，正在下载...")
            elif stage == "done":
                pass

        try:
            if self.client is None:
                self.client = SoraClient()

            result_path = self.client.generate_video_sync(
                prompt=prompt,
                model=模型,
                seconds=seconds,
                size=分辨率,
                save_path=save_path,
                input_reference_base64=ref_base64,
                progress_callback=progress_callback,
                on_stage=on_stage,
            )

            elapsed = time.time() - start_time
            time_str = f"{elapsed:.2f}s" if elapsed >= 1 else f"{elapsed:.3f}s"
            print(f"Sora: 完成！总耗时 {time_str} | 已保存到 {result_path}")

            return (result_path,)

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
