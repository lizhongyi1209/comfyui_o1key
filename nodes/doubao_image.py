"""
豆包生图节点
1:1 复刻字节跳动 Seedream 4 节点的前端外观（输入/输出/参数/样式）
后端通过 new-api 兼容层调用豆包官方 API
"""

import time
import numpy as np
import torch
from PIL import Image
from typing import List

from ..clients.doubao_image_client import DoubaoImageClient
from ..utils.image_utils import tensor_to_pil


# ── 尺寸预设 ──────────────────────────────────────────────────────────────────
# (显示名, 宽, 高) —— 宽高用于构造 "WxH" size 字符串
RECOMMENDED_PRESETS_SEEDREAM_4 = [
    ("2048×2048 (1:1)",  2048, 2048),
    ("2304×1728 (4:3)",  2304, 1728),
    ("1728×2304 (3:4)",  1728, 2304),
    ("2560×1440 (16:9)", 2560, 1440),
    ("1440×2560 (9:16)", 1440, 2560),
    ("2496×1664 (3:2)",  2496, 1664),
    ("1664×2496 (2:3)",  1664, 2496),
    ("3024×1296 (21:9)", 3024, 1296),
    ("3072×3072 (1:1)",  3072, 3072),
    ("4096×4096 (1:1)",  4096, 4096),
    ("自定义",           None, None),
]

_PRESET_LABELS = [label for label, _, _ in RECOMMENDED_PRESETS_SEEDREAM_4]

# ── 模型列表 ──────────────────────────────────────────────────────────────────
# 节点下拉选项 = new-api 后台配置的模型 ID（直接透传给 API）
_MODELS = [
    "doubao-seedream-5-0-260128",
    "doubao-seedream-4-5-251128",
]


def _pil_list_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """
    PIL Image 列表 → ComfyUI IMAGE tensor [B, H, W, C]，值域 [0, 1]。

    多张尺寸不同时，以最大尺寸为准，较小图像丢弃（与项目其他节点策略一致）。
    """
    if not images:
        placeholder = Image.new("RGB", (512, 512), color=(128, 128, 128))
        images = [placeholder]

    # 找最大尺寸
    base_size = max(images, key=lambda img: img.size[0] * img.size[1]).size
    matched = [img for img in images if img.size == base_size]
    skipped = len(images) - len(matched)
    if skipped:
        print(f"[豆包生图] 丢弃 {skipped} 张非最大尺寸图像，仅输出 {base_size[0]}×{base_size[1]} 的 {len(matched)} 张")

    tensors = []
    for img in matched:
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr))

    return torch.stack(tensors, dim=0)   # [B, H, W, C]


class DoubaoImage:
    """豆包生图 —— 1:1 复刻字节跳动 Seedream 4 节点前端，后端对接豆包官方 API"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": (
                    _MODELS,
                    {"default": _MODELS[0]},
                ),
                "提示词": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "用于创建或编辑图像的文本提示",
                    },
                ),
                "尺寸预设": (
                    _PRESET_LABELS,
                    {
                        "default": _PRESET_LABELS[0],
                        "tooltip": '选择推荐尺寸。选择"自定义"可使用下方的宽度和高度',
                    },
                ),
                "宽度": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 1024,
                        "max": 6240,
                        "step": 64,
                        "tooltip": '图像的自定义宽度。仅当尺寸预设设置为"自定义"时生效',
                    },
                ),
                "高度": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 1024,
                        "max": 4992,
                        "step": 64,
                        "tooltip": '图像的自定义高度。仅当尺寸预设设置为"自定义"时生效',
                    },
                ),
                "顺序图像生成": (
                    ["disabled", "auto"],
                    {
                        "default": "disabled",
                        "tooltip": (
                            '分组图像生成模式。'
                            '"disabled"生成单张图像；'
                            '"auto"让模型决定是否生成多张相关图像（如故事场景、角色变体）'
                        ),
                    },
                ),
                "最大图片数": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 15,
                        "step": 1,
                        "tooltip": (
                            "当顺序图像生成='auto'时生成的最大图像数量。"
                            "总图像数（输入+生成）不能超过15张"
                        ),
                    },
                ),
                "种子": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "step": 1,
                        "control_after_generate": True,
                        "tooltip": "用于生成的随机种子",
                    },
                ),
                "部分失败时停止": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "如果启用，当任何请求的图像缺失或返回错误时将中止执行",
                    },
                ),
            },
            "optional": {
                "图像": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "用于图生图的输入图像。"
                            "单参考或多参考生成时，可输入1-10张图像列表"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "generate"
    CATEGORY = "comfyui_o1key/豆包"

    # ── 核心生成方法 ──────────────────────────────────────────────────────────

    def generate(
        self,
        模型: str,
        提示词: str,
        尺寸预设: str,
        宽度: int,
        高度: int,
        顺序图像生成: str,
        最大图片数: int,
        种子: int,
        部分失败时停止: bool,
        图像=None,
    ):
        start_time = time.time()

        # ── 1. 校验提示词 ─────────────────────────────────────────────────────
        if not 提示词.strip():
            raise ValueError("提示词不能为空，请输入图像描述后重试。")

        # ── 2. 解析尺寸 ───────────────────────────────────────────────────────
        w, h = None, None
        for label, tw, th in RECOMMENDED_PRESETS_SEEDREAM_4:
            if label == 尺寸预设:
                w, h = tw, th
                break

        if w is None or h is None:
            # 自定义尺寸
            w, h = 宽度, 高度
            print(f"[豆包生图] 自定义尺寸：{w}×{h}")

        size_str = f"{w}x{h}"

        # ── 3. 打印概要 ───────────────────────────────────────────────────────
        mode_str = "图生图" if 图像 is not None else "文生图"
        seq_str = f" | 顺序生成=auto(最多{最大图片数}张)" if 顺序图像生成 == "auto" else ""
        print(
            f"[豆包生图] {mode_str} | 模型={模型} | 尺寸={size_str}"
            f" | 种子={种子}{seq_str}"
        )

        # ── 4. 进度条 ─────────────────────────────────────────────────────────
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(100)
        except Exception:
            pbar = None

        def _pb(pct: int):
            if pbar:
                pbar.update_absolute(pct, 100)

        _pb(0)

        # ── 5. 调用客户端 ─────────────────────────────────────────────────────
        try:
            client = DoubaoImageClient()
        except ValueError as e:
            raise ValueError(str(e)) from None

        _pb(5)

        try:
            pil_images: List[Image.Image] = client.generate_sync(
                model=模型,
                prompt=提示词,
                size=size_str,
                seed=种子,
                sequential_image_generation=顺序图像生成,
                max_images=最大图片数,
                image_tensor=图像,
            )
        except RuntimeError as e:
            raise RuntimeError(str(e)) from None
        except Exception as e:
            raise RuntimeError(f"豆包生图请求失败: {e}") from None

        _pb(90)

        # ── 6. 部分失败判断 ───────────────────────────────────────────────────
        if 顺序图像生成 == "auto" and 部分失败时停止:
            if len(pil_images) < 最大图片数:
                raise RuntimeError(
                    f"部分图像生成失败：期望 {最大图片数} 张，"
                    f"实际返回 {len(pil_images)} 张。"
                    "（可将【部分失败时停止】设为 False 以接受不完整结果）"
                )

        # ── 7. PIL → tensor ───────────────────────────────────────────────────
        output_tensor = _pil_list_to_tensor(pil_images)

        _pb(100)

        # ── 8. 完成日志 ───────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        time_str = f"{elapsed:.1f}s"
        print(
            f"[豆包生图] 完成！耗时 {time_str}，"
            f"输出 {output_tensor.shape[0]} 张 "
            f"{output_tensor.shape[2]}×{output_tensor.shape[1]}"
        )

        return (output_tensor,)


# ── 节点注册 ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "DoubaoImage": DoubaoImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoImage": "豆包生图",
}
