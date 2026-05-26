"""
背景移除工具模块
基于 rembg 库实现，支持 CPU 推理
"""

import numpy as np
import torch
from PIL import Image

_session = None


def _get_session():
    """懒加载 rembg session，避免启动时加载模型"""
    global _session
    if _session is None:
        try:
            from rembg import new_session
            _session = new_session("isnet-general-use")
            print("[o1key] rembg 模型加载完成 (isnet-general-use)")
        except ImportError:
            raise RuntimeError(
                "未安装 rembg，请执行: pip install rembg[cpu]>=2.0.50"
            )
    return _session


def remove_background_pil(image: Image.Image) -> Image.Image:
    """
    移除 PIL Image 背景，返回 RGBA 图像（背景透明）
    """
    from rembg import remove
    session = _get_session()
    result = remove(image, session=session)
    return result.convert("RGBA")


def remove_background_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    移除 ComfyUI IMAGE tensor 的背景
    输入: [B, H, W, C] (3或4通道)
    输出: [B, H, W, 4] RGBA tensor
    """
    from rembg import remove
    session = _get_session()

    results = []
    batch_size = tensor.shape[0]

    for i in range(batch_size):
        frame = tensor[i]  # [H, W, C]
        arr = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        if arr.shape[2] == 4:
            pil_img = Image.fromarray(arr, mode="RGBA")
        else:
            pil_img = Image.fromarray(arr, mode="RGB")

        result = remove(pil_img, session=session)
        result_rgba = result.convert("RGBA")

        result_arr = np.array(result_rgba).astype(np.float32) / 255.0
        results.append(torch.from_numpy(result_arr))

    return torch.stack(results, dim=0)
