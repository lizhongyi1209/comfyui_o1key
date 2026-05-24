"""
图像元数据去除节点

提供批量去除已有图片中元数据的功能
"""

import os

from PIL import Image
from PIL.PngImagePlugin import PngInfo

# 支持的图片格式
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}


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
# 批量去除元数据
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
