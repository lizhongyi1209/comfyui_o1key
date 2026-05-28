"""
图像处理工具模块
提供 ComfyUI Tensor 与 PIL Image 之间的转换功能
"""

import base64
from io import BytesIO
import json
from typing import Callable, List, Tuple

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """
    将 ComfyUI 的 Tensor 转换为 PIL Image 列表
    
    Args:
        tensor: 形状为 [B, H, W, C] 的张量，值范围 [0, 1]
    
    Returns:
        PIL Image 列表
    
    Example:
        >>> images = tensor_to_pil(input_tensor)
        >>> for img in images:
        ...     img.save(f"output_{i}.png")
    """
    images = []
    
    # 转换为 numpy 数组
    np_images = tensor.cpu().numpy()
    
    # 处理每张图像
    for i in range(np_images.shape[0]):
        img_array = np_images[i]
        
        # 转换值范围从 [0, 1] 到 [0, 255]
        img_array = (img_array * 255).astype(np.uint8)
        
        # 创建 PIL Image
        img = Image.fromarray(img_array)
        images.append(img)
    
    return images


def pil_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """
    将 PIL Image 列表转换为 ComfyUI 的 Tensor

    Args:
        images: PIL Image 列表

    Returns:
        形状为 [B, H, W, C] 的张量，值范围 [0, 1]

    Example:
        >>> pil_images = [Image.open("test.png")]
        >>> tensor = pil_to_tensor(pil_images)
        >>> print(tensor.shape)  # [1, H, W, 3]
    """
    tensors = []

    for img in images:
        # 确保是 RGB 模式
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 转换为 numpy 数组
        img_array = np.array(img).astype(np.float32)

        # 转换值范围从 [0, 255] 到 [0, 1]
        img_array = img_array / 255.0

        tensors.append(img_array)

    # 堆叠为批次
    batch_tensor = np.stack(tensors, axis=0)

    # 转换为 torch tensor
    return torch.from_numpy(batch_tensor)


def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    将 PIL Image 编码为 base64 字符串
    
    Args:
        image: PIL Image 对象
        format: 图像格式，默认 PNG
    
    Returns:
        base64 编码的字符串
    
    Example:
        >>> img = Image.open("test.png")
        >>> b64_str = encode_image_to_base64(img)
    """
    buffered = BytesIO()
    
    # 转换为 RGB 模式（如果是 RGBA）
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    
    return base64.b64encode(img_bytes).decode('utf-8')


_MAX_REQUEST_BODY_BYTES = 50 * 1024 * 1024  # 50MB 请求体上限


def _encode_image_to_base64_with_quality(image: Image.Image, quality: int) -> str:
    buffered = BytesIO()
    working = image
    if working.mode != 'RGB':
        working = working.convert('RGB')

    working.save(
        buffered,
        format="JPEG",
        quality=quality,
        optimize=True,
        subsampling=2,
    )
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def encode_images_for_request_body_limit(
    images: List[Image.Image],
    build_body: Callable[[List[Tuple[str, str]]], dict],
    max_body_bytes: int = _MAX_REQUEST_BODY_BYTES,
) -> List[Tuple[str, str]]:
    """
    为请求体编码图片，并保证完整 JSON 请求体不超过 max_body_bytes。

    策略：
    - 先按原始 PNG 编码估算完整请求体；
    - 若超过限制，改用 JPEG 质量压缩，逐步降低 quality；
    - 全程不缩放图片尺寸。

    Returns:
        [(mime_type, base64), ...]
    """
    encoded = [("image/png", encode_image_to_base64(img, format="PNG")) for img in images]
    body_size = len(json.dumps(build_body(encoded)).encode("utf-8"))
    if body_size <= max_body_bytes:
        return encoded

    for quality in [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1]:
        encoded = [
            ("image/jpeg", _encode_image_to_base64_with_quality(img, quality))
            for img in images
        ]
        body_size = len(json.dumps(build_body(encoded)).encode("utf-8"))
        if body_size <= max_body_bytes:
            print(
                f"输入图片已通过 JPEG 质量压缩控制请求体积: "
                f"quality={quality}, 请求体积={body_size / 1024 / 1024:.2f}MB "
                f"(限制 {max_body_bytes / 1024 / 1024:.0f}MB)"
            )
            return encoded

    raise ValueError(
        f"请求体超过 {max_body_bytes / 1024 / 1024:.0f}MB，"
        "即使压缩到最低图片质量仍无法满足限制；请减少参考图数量或输入图片内容复杂度"
    )


_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB 单张图片上限


def _encode_image_to_bytes(image: Image.Image, format: str = "PNG", quality: int = None) -> bytes:
    buffered = BytesIO()
    working = image

    if format.upper() == "JPEG" and working.mode != 'RGB':
        working = working.convert('RGB')
    elif working.mode == 'RGBA':
        working = working.convert('RGB')

    save_kwargs = {"format": format}
    if quality is not None:
        save_kwargs.update({
            "quality": quality,
            "optimize": True,
            "subsampling": 2,
        })

    working.save(buffered, **save_kwargs)
    return buffered.getvalue()


def encode_images_for_image_size_limit(
    images: List[Image.Image],
    max_image_bytes: int = _MAX_IMAGE_BYTES,
) -> List[Tuple[str, str]]:
    """
    将图片编码为 base64，并保证每张编码前的图片文件体积不超过 max_image_bytes。

    策略：
    - 先尝试 PNG 原图尺寸编码；
    - 单张超过限制时，改用 JPEG 质量压缩；
    - 全程不缩放图片尺寸。

    Returns:
        [(mime_type, base64), ...]
    """
    encoded = []

    for idx, img in enumerate(images, start=1):
        png_bytes = _encode_image_to_bytes(img, format="PNG")
        if len(png_bytes) <= max_image_bytes:
            encoded.append(("image/png", base64.b64encode(png_bytes).decode('utf-8')))
            continue

        for quality in [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1]:
            jpg_bytes = _encode_image_to_bytes(img, format="JPEG", quality=quality)
            if len(jpg_bytes) <= max_image_bytes:
                print(
                    f"输入图片 {idx} 已通过 JPEG 质量压缩控制单图体积: "
                    f"quality={quality}, 图片体积={len(jpg_bytes) / 1024 / 1024:.2f}MB "
                    f"(限制 {max_image_bytes / 1024 / 1024:.0f}MB)，尺寸保持 {img.width}x{img.height}"
                )
                encoded.append(("image/jpeg", base64.b64encode(jpg_bytes).decode('utf-8')))
                break
        else:
            raise ValueError(
                f"输入图片 {idx} 超过 {max_image_bytes / 1024 / 1024:.0f}MB，"
                "即使压缩到最低图片质量仍无法满足限制；请减少图片内容复杂度或手动处理图片"
            )

    return encoded


def encode_image_to_base64_limited(
    image: Image.Image,
    format: str = "PNG",
    max_bytes: int = _MAX_IMAGE_BYTES,
) -> str:
    """
    将 PIL Image 编码为 base64，若超过 max_bytes 则自动缩放直到满足限制。

    策略：等比缩放，每轮缩小到上一轮的 80%，最多 10 轮。

    Args:
        image: PIL Image 对象
        format: 图像格式，默认 PNG
        max_bytes: base64 字符串最大字节数，默认 10MB

    Returns:
        base64 编码的字符串（保证 <= max_bytes）
    """
    working = image
    if working.mode == 'RGBA':
        working = working.convert('RGB')

    for attempt in range(10):
        buffered = BytesIO()
        working.save(buffered, format=format)
        b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        if len(b64) <= max_bytes:
            if attempt > 0:
                print(
                    f"图片已自动缩放: {image.width}x{image.height} → "
                    f"{working.width}x{working.height} "
                    f"({len(b64) / 1024 / 1024:.2f}MB)"
                )
            return b64

        # 缩放到 80%
        scale = 0.8
        new_w = max(1, int(working.width * scale))
        new_h = max(1, int(working.height * scale))
        working = working.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 兜底：返回最后一次编码结果
    buffered = BytesIO()
    working.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def decode_base64_to_pil(base64_string: str) -> Image.Image:
    """
    将 base64 字符串解码为 PIL Image
    
    Args:
        base64_string: base64 编码的图像字符串
    
    Returns:
        PIL Image 对象
    
    Example:
        >>> img = decode_base64_to_pil(b64_str)
        >>> img.save("decoded.png")
    """
    img_bytes = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_bytes))
    
    return img


def parse_batch_prompts(prompt: str) -> List[str]:
    """
    解析批量提示词
    
    检测单独行的 --- 分隔符，分割提示词。
    如果 --- 不是单独占据一行，则返回空列表（表示单提示词模式）。
    
    Args:
        prompt: 用户输入的提示词文本
    
    Returns:
        提示词列表。如果未检测到单独行的 ---，返回空列表（表示单提示词模式）
    
    Raises:
        ValueError: 如果所有提示词都为空
    
    Example:
        >>> prompts = parse_batch_prompts("a woman\\n---\\na man")
        >>> print(prompts)  # ['a woman', 'a man']
        
        >>> prompts = parse_batch_prompts("a woman --- a man")
        >>> print(prompts)  # []  (单提示词模式)
    """
    lines = prompt.split('\n')
    
    # 检查是否存在单独行的 ---
    has_separator = False
    for line in lines:
        if line.strip() == '---':
            has_separator = True
            break
    
    # 如果没有单独行的 ---，返回空列表（单提示词模式）
    if not has_separator:
        return []
    
    # 按单独行的 --- 分割
    # 先将所有单独行的 --- 替换为特殊标记
    processed_lines = []
    for line in lines:
        if line.strip() == '---':
            processed_lines.append('<<<SEPARATOR>>>')
        else:
            processed_lines.append(line)
    
    # 重新组合并分割
    processed_text = '\n'.join(processed_lines)
    raw_prompts = processed_text.split('<<<SEPARATOR>>>')
    
    # 过滤空提示词
    filtered_prompts = []
    for p in raw_prompts:
        stripped = p.strip()
        if stripped:
            filtered_prompts.append(stripped)
    
    # 如果所有提示词都为空，抛出错误
    if not filtered_prompts:
        raise ValueError("批量提示词模式下，所有提示词都为空，请至少提供一个有效的提示词")
    
    return filtered_prompts
