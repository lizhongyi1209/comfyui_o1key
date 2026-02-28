"""
高级图像拼接节点
支持最多 10 张图像按指定方向（上、下、左、右）依次拼接，
支持调整图像大小匹配和添加间隔。
"""

from typing import Optional, Tuple, List

import torch
from PIL import Image

from ..utils.image_utils import tensor_to_pil, pil_to_tensor
from ..utils.file_utils import load_images_from_folder


# 间隔颜色映射
SPACING_COLOR_MAP = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}


def _resize_to_match(img: Image.Image, ref: Image.Image, direction: str) -> Image.Image:
    """
    按拼接方向将 img 缩放，使其与 ref 在垂直于拼接轴的尺寸上一致。

    - 水平拼接 (right/left)：统一高度
    - 垂直拼接 (down/up)：统一宽度
    """
    ref_w, ref_h = ref.size
    img_w, img_h = img.size

    if direction in ("right", "left"):
        if img_h != ref_h:
            scale = ref_h / img_h
            new_w = max(1, int(img_w * scale))
            img = img.resize((new_w, ref_h), Image.LANCZOS)
    else:
        if img_w != ref_w:
            scale = ref_w / img_w
            new_h = max(1, int(img_h * scale))
            img = img.resize((ref_w, new_h), Image.LANCZOS)

    return img


def _make_spacer(ref: Image.Image, spacing_width: int,
                 direction: str, color: Tuple[int, int, int]) -> Image.Image:
    """创建间隔色块"""
    if direction in ("right", "left"):
        return Image.new("RGB", (spacing_width, ref.size[1]), color)
    else:
        return Image.new("RGB", (ref.size[0], spacing_width), color)


def _stitch_two(img_a: Image.Image, img_b: Image.Image,
                direction: str, match_size: bool,
                spacing_width: int, spacing_color: Tuple[int, int, int]) -> Image.Image:
    """
    将两张 PIL 图像按指定方向拼接。
    img_a 为基准图像，img_b 拼接在 img_a 的指定方向侧。
    direction="right" → img_b 在 img_a 右侧
    direction="left"  → img_b 在 img_a 左侧
    direction="down"  → img_b 在 img_a 下方
    direction="up"    → img_b 在 img_a 上方
    """
    if img_a.mode != "RGB":
        img_a = img_a.convert("RGB")
    if img_b.mode != "RGB":
        img_b = img_b.convert("RGB")

    if match_size:
        img_b = _resize_to_match(img_b, img_a, direction)

    if direction == "right":
        pieces = [img_a, img_b]
    elif direction == "left":
        pieces = [img_b, img_a]
    elif direction == "down":
        pieces = [img_a, img_b]
    else:  # up
        pieces = [img_b, img_a]

    if spacing_width > 0:
        interleaved: List[Image.Image] = []
        for idx, piece in enumerate(pieces):
            interleaved.append(piece)
            if idx < len(pieces) - 1:
                interleaved.append(_make_spacer(piece, spacing_width, direction, spacing_color))
        pieces = interleaved

    if direction in ("right", "left"):
        total_w = sum(p.size[0] for p in pieces)
        max_h = max(p.size[1] for p in pieces)
        canvas = Image.new("RGB", (total_w, max_h), spacing_color)
        x = 0
        for piece in pieces:
            canvas.paste(piece, (x, 0))
            x += piece.size[0]
    else:
        max_w = max(p.size[0] for p in pieces)
        total_h = sum(p.size[1] for p in pieces)
        canvas = Image.new("RGB", (max_w, total_h), spacing_color)
        y = 0
        for piece in pieces:
            canvas.paste(piece, (0, y))
            y += piece.size[1]

    return canvas


def _natural_sort_key(filename: str):
    """按数字优先的文件名排序，使 1, 2, 3, 10 而非 1, 10, 2, 3"""
    try:
        return (0, int(filename))
    except ValueError:
        return (1, filename.lower())


class ImageStitchPro:
    """
    高级图像拼接节点

    在 ComfyUI 原生拼接节点基础上扩展，支持同时输入最多 10 张图像，
    按指定方向依次拼接，并可在图像间添加任意颜色的间隔。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "方向": (["right", "down", "left", "up"], {"default": "down"}),
                "匹配图像尺寸": ("BOOLEAN", {"default": True}),
                "间距宽度": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 2}),
                "间距颜色": (["white", "black", "red", "green", "blue"], {"default": "white"}),
            },
            "optional": {
                "图1": ("IMAGE",),
                "图2":  ("IMAGE",),
                "图3":  ("IMAGE",),
                "图4":  ("IMAGE",),
                "图5":  ("IMAGE",),
                "图6":  ("IMAGE",),
                "图7":  ("IMAGE",),
                "图8":  ("IMAGE",),
                "图9":  ("IMAGE",),
                "图10": ("IMAGE",),
                "图11": ("IMAGE",),
                "图12": ("IMAGE",),
                "图片路径（可选）": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("拼接图像",)
    FUNCTION = "stitch"
    CATEGORY = "image"

    DESCRIPTION = (
        "高级图像拼接节点，支持最多 12 张图像按指定方向（右/下/左/上）依次拼接。\n"
        "可选择是否将后续图像缩放以匹配第一张图像的尺寸，并可在图像间添加彩色间隔。\n"
        "可选填「图片路径」：仅处理该文件夹内图片，按文件名顺序依次拼接；与输入端图片不可同时使用。"
    )

    def stitch(
        self,
        方向: str = "down",
        匹配图像尺寸: bool = True,
        间距宽度: int = 0,
        间距颜色: str = "white",
        图1:  Optional[torch.Tensor] = None,
        图2:  Optional[torch.Tensor] = None,
        图3:  Optional[torch.Tensor] = None,
        图4:  Optional[torch.Tensor] = None,
        图5:  Optional[torch.Tensor] = None,
        图6:  Optional[torch.Tensor] = None,
        图7:  Optional[torch.Tensor] = None,
        图8:  Optional[torch.Tensor] = None,
        图9:  Optional[torch.Tensor] = None,
        图10: Optional[torch.Tensor] = None,
        图11: Optional[torch.Tensor] = None,
        图12: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Tuple[torch.Tensor]:

        color = SPACING_COLOR_MAP.get(间距颜色, (255, 255, 255))
        raw_tensors = [图1, 图2, 图3, 图4, 图5, 图6, 图7, 图8, 图9, 图10, 图11, 图12]
        tensors = [t for t in raw_tensors if t is not None]
        has_input_images = len(tensors) > 0
        image_folder = (kwargs.get("图片路径（可选）") or "").strip()

        if image_folder and has_input_images:
            raise ValueError("不可同时使用「图片路径（可选）」与输入端图片，请二选一。")

        if image_folder:
            infos = load_images_from_folder(image_folder)
            if not infos:
                raise ValueError(f"文件夹中未找到可用的图片，或路径无效: {image_folder}")
            infos.sort(key=lambda x: _natural_sort_key(x.filename))
            pil_list = [info.image for info in infos]
            if len(pil_list) == 1:
                return (pil_to_tensor(pil_list),)
            base = pil_list[0]
            for next_img in pil_list[1:]:
                base = _stitch_two(
                    base, next_img,
                    direction=方向,
                    match_size=匹配图像尺寸,
                    spacing_width=间距宽度,
                    spacing_color=color,
                )
            return (pil_to_tensor([base]),)
        else:
            if not has_input_images:
                raise ValueError("请至少接入一张图片，或填写「图片路径（可选）」中的文件夹路径。")

        if len(tensors) == 1:
            return (tensors[0],)

        pil_batches: List[List[Image.Image]] = [tensor_to_pil(t) for t in tensors]

        batch_size = min(len(b) for b in pil_batches)
        result_images: List[Image.Image] = []

        for i in range(batch_size):
            frames = [batch[i] for batch in pil_batches]
            base = frames[0]
            for next_img in frames[1:]:
                base = _stitch_two(
                    base, next_img,
                    direction=方向,
                    match_size=匹配图像尺寸,
                    spacing_width=间距宽度,
                    spacing_color=color,
                )
            result_images.append(base)

        return (pil_to_tensor(result_images),)
