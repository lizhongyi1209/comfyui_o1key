"""
图像处理工具模块
提供 ComfyUI Tensor 与 PIL Image 之间的转换功能
"""

import base64
from io import BytesIO
from typing import List

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