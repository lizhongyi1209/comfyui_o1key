"""
o1key 颜色去背景节点
基于颜色距离计算，精确可控，不依赖 AI 模型
"""

import numpy as np
import torch
from PIL import Image


class O1keyColorRemoveBG:
    """
    颜色去背景 - 精确移除纯色背景

    模式说明：
      - 白色(white): 移除白色背景，适合大多数场景
      - 白色保护(white-preserve): 移除白底但保护浅色前景物体
      - 自动检测(corner): 自动采样四角颜色作为背景色
      - 指定颜色(color): 手动指定要移除的背景颜色
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "模式": (["白色", "白色保护", "自动检测", "指定颜色"], {
                    "default": "白色",
                }),
                "容差": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "颜色距离阈值，越大去除范围越广",
                }),
                "羽化": ("FLOAT", {
                    "default": 45.0,
                    "min": 0.0,
                    "max": 200.0,
                    "step": 1.0,
                    "tooltip": "边缘过渡范围，越大边缘越柔和",
                }),
            },
            "optional": {
                "背景色R": ("INT", {"default": 255, "min": 0, "max": 255}),
                "背景色G": ("INT", {"default": 255, "min": 0, "max": 255}),
                "背景色B": ("INT", {"default": 255, "min": 0, "max": 255}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGBA图像",)
    FUNCTION = "remove_bg"
    CATEGORY = "o1key/image"

    _MODE_MAP = {
        "白色": "white",
        "白色保护": "white-preserve",
        "自动检测": "corner",
        "指定颜色": "color",
    }

    def remove_bg(self, image, 模式, 容差, 羽化, 背景色R=255, 背景色G=255, 背景色B=255):
        from ..utils.color_key import remove_background

        mode = self._MODE_MAP.get(模式, "white")
        bg_color = (背景色R, 背景色G, 背景色B)

        batch_size = image.shape[0]
        results = []

        for i in range(batch_size):
            frame = image[i]  # [H, W, C]
            arr = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            if arr.shape[2] == 4:
                pil_img = Image.fromarray(arr, mode="RGBA")
            else:
                pil_img = Image.fromarray(arr, mode="RGB")

            result = remove_background(
                pil_img, mode=mode, bg_color=bg_color,
                tolerance=容差, feather=羽化,
            )

            result_arr = np.array(result.convert("RGBA")).astype(np.float32) / 255.0
            results.append(torch.from_numpy(result_arr))

        output = torch.stack(results, dim=0)
        print(f"[o1key 颜色去背景] 模式={模式}, 容差={容差}, 羽化={羽化}, "
              f"处理 {batch_size} 张")
        return (output,)
