"""
图像元数据去除节点
替代 ComfyUI 原生"保存图像"节点，保存时不写入提示词、工作流等 AI 元数据

提供两种节点：
1. SaveCleanImage   - 接收 IMAGE 张量，去除元数据后直接保存到 output 目录
2. BatchCleanMetadata - 指定文件夹路径，批量去除已有图片中的元数据
"""

import os
import re
from typing import List

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ..utils.image_utils import tensor_to_pil

# 尝试导入 ComfyUI 的 folder_paths
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

# 支持的图片格式
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}


def _get_output_dir() -> str:
    """
    获取 ComfyUI output 目录

    Returns:
        output 目录的绝对路径
    """
    if FOLDER_PATHS_AVAILABLE:
        return folder_paths.get_output_directory()
    # fallback: 相对于插件目录推断
    plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(os.path.dirname(os.path.dirname(plugin_dir)), "output")


def _get_next_counter(directory: str, prefix: str) -> int:
    """
    扫描目录，获取下一个可用的文件计数器

    Args:
        directory: 目标目录
        prefix: 文件名前缀

    Returns:
        下一个计数器值
    """
    if not os.path.exists(directory):
        return 1

    pattern = re.compile(rf'^{re.escape(prefix)}_(\d+)')
    max_counter = 0

    for f in os.listdir(directory):
        m = pattern.match(f)
        if m:
            counter = int(m.group(1))
            max_counter = max(max_counter, counter)

    return max_counter + 1


def _save_image_clean(image: Image.Image, path: str, fmt: str = None, quality: int = 95) -> None:
    """
    保存图像，不包含任何元数据

    通过提取纯像素数据并重建全新的 Image 对象，确保没有任何元数据残留。

    Args:
        image: PIL Image 对象
        path: 保存路径
        fmt: 图像格式（PNG/JPEG/WEBP），为 None 时根据扩展名推断
        quality: JPEG/WEBP 质量（1-100）
    """
    # 确保 RGB 模式
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 提取纯像素数据，重建全新的 Image 对象
    # 使用 tobytes() + frombytes() 确保只保留像素数据，彻底断开与原图像的关联
    pixel_data = image.tobytes()
    clean = Image.frombytes('RGB', image.size, pixel_data)

    # 显式清空 info 字典，确保不会有任何残留元数据
    clean.info = {}

    # 推断格式
    if fmt is None:
        ext = os.path.splitext(path)[1].lower()
        format_map = {
            '.png': 'PNG',
            '.jpg': 'JPEG',
            '.jpeg': 'JPEG',
            '.webp': 'WEBP',
            '.bmp': 'BMP',
            '.tiff': 'TIFF',
            '.tif': 'TIFF',
        }
        fmt = format_map.get(ext, 'PNG')

    # 构建保存参数（确保不写入任何元数据）
    save_kwargs = {}
    if fmt == 'PNG':
        save_kwargs['pnginfo'] = PngInfo()  # 空的 PngInfo，不包含任何文本块
    elif fmt == 'JPEG':
        save_kwargs['quality'] = quality
        # 不传 exif 参数，自然不会写入 EXIF 数据
    elif fmt == 'WEBP':
        save_kwargs['quality'] = quality
        save_kwargs['exif'] = b""  # 显式清空 EXIF

    clean.save(path, format=fmt, **save_kwargs)


# ============================================================================
# 节点 1：保存干净图像
# ============================================================================

class SaveCleanImage:
    """
    保存干净图像节点（不含元数据）

    功能：
    - 接收 IMAGE 张量（支持单图和批次）
    - 去除所有元数据后保存到 ComfyUI/output 目录
    - 文件名自动添加 nometa 标识，方便辨认
    - 支持 PNG/JPEG/WEBP 格式
    - 作为终端节点，替代 ComfyUI 原生"保存图像"节点

    使用场景：
    - 生图完成后，直接保存不含 AI 元数据的干净图像
    - 分享图像时不暴露提示词和工作流
    """

    SAVE_FORMATS = ["PNG", "JPEG", "WEBP"]

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义输入参数

        Returns:
            输入参数配置字典
        """
        return {
            "required": {
                "图像": ("IMAGE",),
                "文件名前缀": ("STRING", {"default": "ComfyUI_nometa"}),
                "保存格式": (cls.SAVE_FORMATS, {"default": "PNG"}),
            },
            "optional": {
                "JPEG/WEBP质量": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_clean"
    CATEGORY = "image"

    DESCRIPTION = (
        "保存干净图像（不含元数据）。\n"
        "替代 ComfyUI 原生'保存图像'节点，保存时不写入提示词、工作流等 AI 元数据。\n"
        "文件保存到 ComfyUI/output 目录。"
    )

    def save_clean(
        self,
        图像: torch.Tensor,
        文件名前缀: str = "ComfyUI_nometa",
        保存格式: str = "PNG",
        **kwargs
    ) -> dict:
        """
        去除元数据并保存图像

        Args:
            图像: ComfyUI 图像张量 [B, H, W, C]
            文件名前缀: 保存文件名前缀
            保存格式: 图像格式（PNG/JPEG/WEBP）
            **kwargs: 可选参数（JPEG/WEBP质量）

        Returns:
            UI 结果字典，包含保存的图像信息用于前端预览
        """
        quality = kwargs.get("JPEG/WEBP质量", 95)

        output_dir = _get_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        # 格式与扩展名映射
        ext_map = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
        ext = ext_map.get(保存格式, ".png")

        # 转换为 PIL 图像
        pil_images = tensor_to_pil(图像)

        # 获取计数器起始值
        counter = _get_next_counter(output_dir, 文件名前缀)

        results = []
        saved_paths = []
        for img in pil_images:
            filename = f"{文件名前缀}_{counter:05d}{ext}"
            filepath = os.path.join(output_dir, filename)

            _save_image_clean(img, filepath, fmt=保存格式, quality=quality)

            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "output"
            })
            saved_paths.append(filepath)
            counter += 1

        # 打印详细日志，方便用户定位保存的文件
        print(f"保存干净图像: 已保存 {len(pil_images)} 张无元数据图像 (格式: {保存格式})")
        for p in saved_paths:
            print(f"  → {p}")

        return {"ui": {"images": results}}


# ============================================================================
# 节点 2：批量去除元数据
# ============================================================================

class BatchCleanMetadata:
    """
    批量去除文件夹中图片元数据的节点

    功能：
    - 指定文件夹路径，批量处理其中所有图片
    - 去除 EXIF、PNG tEXt 块、ComfyUI 工作流等所有元数据
    - 支持保存到原目录（添加 _nometa 后缀）或覆盖原文件
    - 支持 PNG/JPG/JPEG/WEBP/BMP/TIFF 格式

    使用场景：
    - 已经保存了一批含有 AI 元数据的图片，需要批量清理
    - 批量处理指定文件夹中的所有图片
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义输入参数

        Returns:
            输入参数配置字典
        """
        return {
            "required": {
                "文件夹路径": ("STRING", {"default": ""}),
                "覆盖原文件": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("处理结果",)
    OUTPUT_NODE = True
    FUNCTION = "batch_clean"
    CATEGORY = "image"

    DESCRIPTION = (
        "批量去除文件夹中图片的元数据。\n"
        "支持 PNG/JPG/JPEG/WEBP/BMP/TIFF 格式。\n"
        "默认在原文件名后添加 _nometa 后缀保存，也可选择覆盖原文件。"
    )

    def batch_clean(
        self,
        文件夹路径: str,
        覆盖原文件: bool = False,
    ) -> tuple:
        """
        批量去除文件夹中图片的元数据

        Args:
            文件夹路径: 待处理图片所在的文件夹路径
            覆盖原文件: 是否覆盖原文件（False 则添加 _nometa 后缀）

        Returns:
            处理结果字符串

        Raises:
            ValueError: 文件夹路径无效
        """
        if not 文件夹路径 or not 文件夹路径.strip():
            raise ValueError("请输入文件夹路径")

        folder = 文件夹路径.strip()

        if not os.path.isdir(folder):
            raise ValueError(f"文件夹路径无效或不存在: {folder}")

        # 扫描支持的图片文件
        files = []
        for f in sorted(os.listdir(folder)):
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(f)

        if not files:
            msg = f"文件夹中未找到支持的图片文件 ({', '.join(SUPPORTED_EXTENSIONS)})"
            print(f"批量去除元数据: {msg}")
            return (msg,)

        print(f"批量去除元数据: 找到 {len(files)} 张图片，开始处理...")

        success_count = 0
        fail_count = 0

        for f in files:
            try:
                src_path = os.path.join(folder, f)
                img = Image.open(src_path)

                if 覆盖原文件:
                    dst_path = src_path
                else:
                    name, ext = os.path.splitext(f)
                    dst_path = os.path.join(folder, f"{name}_nometa{ext}")

                _save_image_clean(img, dst_path)
                success_count += 1

            except Exception as e:
                print(f"批量去除元数据: 处理 {f} 失败 - {str(e)}")
                fail_count += 1

        # 构建结果消息
        if fail_count > 0:
            msg = f"处理完成: 成功 {success_count} 张, 失败 {fail_count} 张"
        else:
            msg = f"处理完成: 全部 {success_count} 张成功"

        if not 覆盖原文件:
            msg += " (已添加 _nometa 后缀)"
        else:
            msg += " (已覆盖原文件)"

        print(f"批量去除元数据: {msg}")

        return (msg,)
