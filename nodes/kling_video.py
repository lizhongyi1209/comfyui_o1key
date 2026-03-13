"""
Kling 可灵 视频生成节点
ComfyUI 自定义节点，调用 Kling API 生成图生视频
"""

import base64
import os
import re
import time
from io import BytesIO
from typing import Optional, Tuple

import torch

from ..utils.image_utils import tensor_to_pil
from ..clients.kling_client import KlingClient

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
    print("⚠️ KlingVideo: comfy.utils.ProgressBar 不可用，将只使用终端进度显示")


# 模型选项
KLING_MODELS = [
    "kling-v1-5",
    "kling-v1-6",
    "kling-v2-6",
]

# 时长选项
DURATION_OPTIONS = [
    ("5", "5秒"),
    ("10", "10秒"),
]

# 模式选项
MODE_OPTIONS = [
    ("std", "std - 高性能"),
    ("pro", "pro - 高表现"),
]

# 声音选项
SOUND_OPTIONS = [
    ("off", "关闭"),
]


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


def image_to_base64(image) -> str:
    """
    将 PIL Image 转换为 Base64 编码字符串（data URI 格式）

    Args:
        image: PIL Image 对象

    Returns:
        Base64 编码的图片字符串，可直接用于 API 请求
    """
    from PIL import Image as PILImage

    # 统一转换为 RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 编码为 PNG Base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return f"data:image/png;base64,{img_base64}"


class KlingVideo:
    """
    Kling 可灵 图生视频生成节点

    功能：
    - 图生视频：基于参考图片和提示词生成视频
    - 异步轮询：自动等待生成完成并下载
    """

    def __init__(self):
        self.client = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "参考图片": ("IMAGE",),  # ComfyUI 输入的图片 tensor
                "模型": (KLING_MODELS, {
                    "default": "kling-v2-6",
                }),
                "提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "负向提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "视频时长": ([d for d, _ in DURATION_OPTIONS], {
                    "default": "5",
                }),
                "生成模式": ([m for m, _ in MODE_OPTIONS], {
                    "default": "pro",
                }),
                "生成数量": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                }),
            },
            "optional": {
                "尾帧图片": ("IMAGE",),  # 可选的尾帧控制图片
                "cfg_scale": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "round": 0.1,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("视频路径",)
    FUNCTION = "generate_video"
    CATEGORY = "video/generation"

    DESCRIPTION = (
        "Kling 可灵 图生视频生成节点。\n"
        "基于参考图片和提示词生成视频，自动轮询任务状态并下载视频。\n"
        "视频保存到 ComfyUI/output/video/ 目录。\n\n"
        "【模型说明】\n"
        "• kling-v1-5：Kling v1.5 模型\n"
        "• kling-v1-6：Kling v1.6 模型\n"
        "• kling-v2-6：Kling v2.6 模型（最新）\n\n"
        "【时长说明】\n"
        "• 5秒：支持尾帧控制\n"
        "• 10秒：不支持尾帧控制\n\n"
        "【生成模式】\n"
        "• std：高性能模式\n"
        "• pro：高表现模式（推荐）\n\n"
        "【尾帧控制】\n"
        "可选输入，用于控制视频的结尾画面\n"
        "注意：使用尾帧时，视频时长只能为 5 秒"
    )

    def generate_video(
        self,
        参考图片,
        模型: str,
        提示词: str = "",
        负向提示词: str = "",
        视频时长: str = "5",
        生成模式: str = "pro",
        生成数量: int = 1,
        尾帧图片=None,
        cfg_scale: float = 0.5,
    ) -> Tuple[str]:
        start_time = time.time()

        # 转换图片为 Base64
        pil_images = tensor_to_pil(参考图片)
        if not pil_images:
            raise ValueError("参考图片转换失败")
        
        # 使用第一张图片
        main_image = pil_images[0]
        image_base64 = image_to_base64(main_image)
        
        # 处理尾帧图片
        image_tail_base64 = None
        if 尾帧图片 is not None:
            pil_tail_images = tensor_to_pil(尾帧图片)
            if pil_tail_images:
                # 有尾帧时，时长只能是 5 秒
                if 视频时长 != "5":
                    print("⚠️ Kling: 检测到尾帧图片，已自动将视频时长调整为 5 秒")
                    视频时长 = "5"
                image_tail_base64 = image_to_base64(pil_tail_images[0])

        print(f"Kling: 图生视频 | 并发{生成数量}个 | {模型} | {视频时长}秒 | {生成模式}")

        # 准备保存路径
        video_dir = _get_video_output_dir()
        counter = _get_next_counter(video_dir, "kling")

        # ProgressBar
        pbar = None
        if PROGRESS_BAR_AVAILABLE:
            pbar = ProgressBar(生成数量 if 生成数量 > 1 else 100)

        try:
            if self.client is None:
                self.client = KlingClient()

            if 生成数量 == 1:
                # ── 单个视频：保留详细进度
                save_path = os.path.join(video_dir, f"kling_{counter:05d}.mp4")
                last_progress = [0]

                def progress_callback(progress_pct: int):
                    print(
                        f"\rKling: 生成中... 进度: {progress_pct}%",
                        end="", flush=True
                    )
                    if pbar is not None and progress_pct > last_progress[0]:
                        pbar.update(progress_pct - last_progress[0])
                        last_progress[0] = progress_pct

                def on_stage(stage: str):
                    if stage == "submitting":
                        print("Kling: 正在提交视频生成任务...")
                    elif stage.startswith("submitted:"):
                        vid = stage.split(":", 1)[1]
                        print(f"Kling: 视频任务已提交，ID: {vid}")
                    elif stage == "polling":
                        print("Kling: 等待视频生成...")
                    elif stage == "downloading":
                        print("")  # 换行
                        print("Kling: 视频生成完成，正在下载...")

                result_path = self.client.generate_video_sync(
                    image=image_base64,
                    prompt=提示词,
                    negative_prompt=负向提示词,
                    model_name=模型,
                    duration=视频时长,
                    mode=生成模式,
                    image_tail=image_tail_base64,
                    cfg_scale=cfg_scale,
                    save_path=save_path,
                    progress_callback=progress_callback,
                    on_stage=on_stage,
                )
                result_paths = [result_path]

            else:
                # ── 批量并发
                save_paths = [
                    os.path.join(video_dir, f"kling_{counter + i:05d}.mp4")
                    for i in range(生成数量)
                ]
                success_count = [0]
                fail_count = [0]

                def batch_progress_callback(current: int, total: int, success: bool, error_msg):
                    if success:
                        success_count[0] += 1
                        print(f"Kling: 第 {current}/{total} 个视频完成 ✓")
                    else:
                        fail_count[0] += 1
                        print(f"Kling: 第 {current}/{total} 个视频失败 ✗")
                        if error_msg:
                            print(f"原始错误详情:\n{error_msg}")
                    if pbar is not None:
                        pbar.update(1)

                # Kling 暂不支持批量并发，这里简化处理
                result_paths = []
                for i, save_path in enumerate(save_paths):
                    try:
                        path = self.client.generate_video_sync(
                            image=image_base64,
                            prompt=提示词,
                            negative_prompt=负向提示词,
                            model_name=模型,
                            duration=视频时长,
                            mode=生成模式,
                            image_tail=image_tail_base64,
                            cfg_scale=cfg_scale,
                            save_path=save_path,
                        )
                        result_paths.append(path)
                        success_count[0] += 1
                        print(f"Kling: 第 {i+1}/{生成数量} 个视频完成 ✓")
                        if pbar is not None:
                            pbar.update(1)
                    except Exception as e:
                        fail_count[0] += 1
                        print(f"Kling: 第 {i+1}/{生成数量} 个视频失败 ✗")
                        print(f"原始错误详情:\n{str(e)}")

            elapsed = time.time() - start_time
            time_str = f"{elapsed:.2f}s" if elapsed >= 1 else f"{elapsed:.3f}s"
            print(f"Kling: 完成！总耗时 {time_str} | 已生成 {len(result_paths)} 个视频")
            for p in result_paths:
                print(f"  → {p}")

            output_path = "\n".join(result_paths)
            return (output_path,)

        except ValueError as e:
            error_msg = str(e)
            print(f"\nKling: ❌ {error_msg}")
            raise ValueError(error_msg) from None

        except RuntimeError as e:
            error_msg = str(e)
            print(f"\nKling: ❌ {error_msg}")
            raise RuntimeError(error_msg) from None

        except Exception as e:
            error_msg = str(e)
            print(f"\nKling: ❌ {error_msg}")
            raise type(e)(error_msg) from None
