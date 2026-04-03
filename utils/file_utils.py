"""
文件处理工具模块
提供文件夹图片加载、智能命名、图片配对等功能
"""

import os
import re
import uuid
import time
import random
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Tuple, Optional, NamedTuple

from PIL import Image


def _get_server_port() -> Optional[int]:
    """获取当前 ComfyUI 实例的端口号，失败返回 None"""
    try:
        import comfy.cli_args
        port = getattr(comfy.cli_args.args, 'port', None) or getattr(comfy.cli_args, 'server_port', None) or getattr(comfy.cli_args, 'port', None)
        if port is not None:
            return int(port)
    except Exception:
        pass
    # 备用：从 listen 环境变量或命令行参数尝试
    try:
        import sys
        for arg in sys.argv:
            if '--port' in arg or '--listen-port' in arg:
                parts = arg.split('=')
                if len(parts) == 2:
                    return int(parts[1].strip())
                elif arg in ('--port', '--listen-port'):
                    idx = sys.argv.index(arg)
                    if idx + 1 < len(sys.argv):
                        return int(sys.argv[idx + 1])
    except Exception:
        pass
    return None


def _get_port_suffix() -> str:
    """
    返回非默认端口的后缀字符串（如 "_8189"），默认端口 8188 或获取失败时返回空字符串。
    """
    try:
        port = _get_server_port()
        if port is not None and port != 8188:
            return f"_{port}"
    except Exception:
        pass
    return ""


# 支持的图片格式
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}


class ImageInfo(NamedTuple):
    """图片信息结构"""
    image: Image.Image
    filename: str  # 不含扩展名的文件名
    extension: str  # 扩展名（如 .png）
    source_path: str  # 原始文件路径


def load_images_from_folder(
    folder_path: str,
    recursive: bool = False
) -> List[ImageInfo]:
    """
    从文件夹加载所有图片
    
    Args:
        folder_path: 文件夹路径
        recursive: 是否递归加载子文件夹
    
    Returns:
        ImageInfo 列表，包含图片和元数据
    
    Raises:
        ValueError: 文件夹不存在或为空
    
    Example:
        >>> images = load_images_from_folder("D:/images")
        >>> for info in images:
        ...     print(f"{info.filename}: {info.image.size}")
    """
    folder_path = folder_path.strip()
    
    if not folder_path:
        return []
    
    path = Path(folder_path)
    
    if not path.exists():
        raise ValueError(f"文件夹不存在: {folder_path}")
    
    if not path.is_dir():
        raise ValueError(f"路径不是文件夹: {folder_path}")
    
    images = []
    
    # 获取文件列表
    if recursive:
        files = list(path.rglob("*"))
    else:
        files = list(path.iterdir())
    
    # 按文件名排序，确保顺序一致
    files = sorted(files, key=lambda x: x.name.lower())
    
    for file_path in files:
        if not file_path.is_file():
            continue
        
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        
        try:
            img = Image.open(file_path)
            img.load()  # 确保图片完全加载
            
            # 转换为 RGB 模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            images.append(ImageInfo(
                image=img,
                filename=file_path.stem,
                extension=ext,
                source_path=str(file_path)
            ))
        except Exception as e:
            print(f"警告: 无法加载图片 {file_path}: {e}")
            continue
    
    return images


def pair_images_indexed(
    *image_lists: List[ImageInfo]
) -> List[Tuple[ImageInfo, ...]]:
    """
    1:1 索引配对
    
    按索引位置配对多个图片列表，以最短列表长度为准。
    
    Args:
        *image_lists: 多个 ImageInfo 列表
    
    Returns:
        配对后的元组列表
    
    Example:
        >>> list_a = [a1, a2, a3]
        >>> list_b = [b1, b2, b3]
        >>> pairs = pair_images_indexed(list_a, list_b)
        >>> # [(a1, b1), (a2, b2), (a3, b3)]
    """
    if not image_lists:
        return []
    
    # 过滤空列表
    non_empty_lists = [lst for lst in image_lists if lst]
    
    if not non_empty_lists:
        return []
    
    # 使用 zip 进行索引配对（以最短列表为准）
    return list(zip(*non_empty_lists))


def pair_images_by_name(
    *image_lists: List[ImageInfo]
) -> List[Tuple[ImageInfo, ...]]:
    """
    按文件名配对（同名匹配）

    取所有文件夹中文件名（不含扩展名）的交集，按文件名字母升序排列后配对。
    只有在所有文件夹中都存在同名文件，该文件名才会被纳入配对。
    扩展名不同的文件（如 1.jpg 与 1.png）视为同名。

    Args:
        *image_lists: 多个 ImageInfo 列表

    Returns:
        配对后的元组列表，按文件名字母升序排列

    Raises:
        ValueError: 所有文件夹之间没有任何相同文件名时抛出

    Example:
        >>> list_a = [ImageInfo(filename="1", ...), ImageInfo(filename="2", ...)]
        >>> list_b = [ImageInfo(filename="1", ...), ImageInfo(filename="3", ...)]
        >>> pairs = pair_images_by_name(list_a, list_b)
        >>> # [(list_a[0], list_b[0])]  # 只有 "1" 匹配
    """
    if not image_lists:
        return []

    non_empty_lists = [lst for lst in image_lists if lst]
    if not non_empty_lists:
        return []

    # 单文件夹直接返回（无需配对）
    if len(non_empty_lists) == 1:
        return [(img,) for img in non_empty_lists[0]]

    # 为每个文件夹建立 filename（stem）-> ImageInfo 的映射
    name_maps = [
        {img.filename: img for img in lst}
        for lst in non_empty_lists
    ]

    # 取所有文件夹文件名的交集
    common_names = set(name_maps[0].keys())
    for nm in name_maps[1:]:
        common_names &= set(nm.keys())

    if not common_names:
        # 收集各文件夹的文件名示例，帮助用户排查问题
        folder_samples = []
        for i, nm in enumerate(name_maps):
            sample = sorted(nm.keys())[:3]
            sample_str = "、".join(f'"{n}"' for n in sample)
            folder_samples.append(f"文件夹{i + 1}：{sample_str}")
        samples_info = "\n".join(folder_samples)
        raise ValueError(
            f"所有文件夹中没有找到任何同名图片，无法进行配对！\n"
            f"请确保各文件夹内存在文件名相同的图片后重试。\n"
            f"（文件名比较不含扩展名，例如「1.jpg」与「1.png」视为同名）\n\n"
            f"各文件夹当前文件名示例：\n{samples_info}"
        )

    # 按文件名字母升序排列，保证顺序稳定
    sorted_names = sorted(common_names, key=lambda x: x.lower())

    return [
        tuple(nm[name] for nm in name_maps)
        for name in sorted_names
    ]


def pair_images_cartesian(
    *image_lists: List[ImageInfo]
) -> List[Tuple[ImageInfo, ...]]:
    """
    笛卡尔积配对
    
    生成多个图片列表的所有组合。
    
    Args:
        *image_lists: 多个 ImageInfo 列表
    
    Returns:
        配对后的元组列表
    
    Example:
        >>> list_a = [a1, a2]
        >>> list_b = [b1, b2]
        >>> pairs = pair_images_cartesian(list_a, list_b)
        >>> # [(a1, b1), (a1, b2), (a2, b1), (a2, b2)]
    """
    if not image_lists:
        return []
    
    # 过滤空列表
    non_empty_lists = [lst for lst in image_lists if lst]
    
    if not non_empty_lists:
        return []
    
    # 使用 itertools.product 生成笛卡尔积
    return list(product(*non_empty_lists))


def generate_timestamp_filename(output_folder: str, prefix: str = "", extension: str = ".png", port_suffix: str = "") -> str:
    """
    生成基于时间戳的文件名，确保按文件名排序 = 按生成时间排序。

    格式：{prefix}{HHMMSS_YYYYMMDD_mmm}{port_suffix}{extension}
    例如：161700_20260322_001.png 或 去除ai_161700_20260322_001.png

    Args:
        output_folder: 输出目录
        prefix: 文件名前缀（如 "去除ai_"）
        extension: 文件扩展名（如 ".png"）
        port_suffix: 端口后缀（如 "_8189"），为空时自动获取

    Returns:
        完整文件路径
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    if not port_suffix:
        port_suffix = _get_port_suffix()

    date_part = datetime.now().strftime("%Y%m%d")
    time_part = datetime.now().strftime("%H%M%S")
    ms = random.randint(0, 999)

    while True:
        filename = f"{prefix}{time_part}_{date_part}_{ms:03d}{port_suffix}{extension}"
        full_path = Path(output_folder) / filename
        if not full_path.exists():
            return str(full_path)
        ms = (ms + 1) % 1000


def save_image(
    image: Image.Image,
    output_path: str,
    quality: int = 95
) -> str:
    """
    保存图片到指定路径
    
    Args:
        image: PIL Image 对象
        output_path: 输出文件路径
        quality: JPEG 质量（仅对 JPEG 格式有效）
    
    Returns:
        实际保存的文件路径
    """
    # 确保目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 根据扩展名选择保存参数
    ext = Path(output_path).suffix.lower()
    
    if ext in {'.jpg', '.jpeg'}:
        # 转换为 RGB（JPEG 不支持 alpha 通道）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(output_path, quality=quality)
    elif ext == '.webp':
        image.save(output_path, quality=quality)
    else:
        image.save(output_path)
    
    return output_path


def get_folder_image_count(folder_path: str) -> int:
    """
    获取文件夹中的图片数量（不加载图片）
    
    Args:
        folder_path: 文件夹路径
    
    Returns:
        图片数量
    """
    folder_path = folder_path.strip()
    
    if not folder_path:
        return 0
    
    path = Path(folder_path)
    
    if not path.exists() or not path.is_dir():
        return 0
    
    count = 0
    for file_path in path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            count += 1
    
    return count
