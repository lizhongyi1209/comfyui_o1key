"""
o1key 去背景节点
基于 rembg 实现，支持 CPU 推理
"""

import numpy as np
import torch


class O1keyRemoveBackground:
    """
    移除图像背景，输出 RGBA 透明图层

    基于 rembg (ISNet-General-Use) 模型，支持 CPU 推理。
    首次运行会自动下载模型（约 170MB）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGBA图像",)
    FUNCTION = "remove_bg"
    CATEGORY = "o1key/image"

    def remove_bg(self, image):
        from ..utils.rembg_utils import remove_background_tensor
        print("[o1key 去背景] 正在处理...")
        result = remove_background_tensor(image)
        print(f"[o1key 去背景] 完成，输出 {result.shape[0]} 张 RGBA")
        return (result,)
