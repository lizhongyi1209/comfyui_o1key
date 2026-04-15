"""
o1key GPT Image 节点
支持 gpt-image-1 / gpt-image-1.5 模型的文生图、图生图、图像编辑（带蒙版）
"""

import time
import torch

from ..clients.gpt_image_client import GptImageClient


class O1keyGPTImage:
    """
    o1key GPT Image 节点

    功能：
      - 文生图：仅提供 prompt
      - 图生图：提供 prompt + image（无 mask）
      - 图像编辑：提供 prompt + image + mask（白色区域将被替换）

    参数：
      - prompt      : 文本提示词（多行）
      - seed        : 随机种子（0 表示不指定）
      - quality     : 图像质量 low / medium / high
      - background  : 背景模式 auto / opaque / transparent
      - size        : 图像尺寸（auto 让 API 自动决定）
      - n           : 生成数量 1-8
      - image       : 可选参考图（用于图生图或编辑）
      - mask        : 可选蒙版（白色区域将被替换）
      - model       : 模型选择 gpt-image-1 / gpt-image-1.5
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text prompt for GPT Image",
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "step": 1,
                    "display": "number",
                    "control_after_generate": True,
                    "tooltip": "Random seed (0 = not specified)",
                }),
                "quality": (["low", "medium", "high"], {
                    "default": "low",
                    "tooltip": "Image quality, affects cost and generation time.",
                }),
                "background": (["auto", "opaque", "transparent"], {
                    "default": "auto",
                    "tooltip": "Return image with or without background",
                }),
                "size": (["auto", "1024x1024", "1024x1536", "1536x1024"], {
                    "default": "auto",
                    "tooltip": "Image size (auto = API decides)",
                }),
                "n": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "display": "number",
                    "tooltip": "How many images to generate",
                }),
                "image": ("IMAGE", {
                    "tooltip": "Optional reference image for image editing.",
                }),
                "mask": ("MASK", {
                    "tooltip": "Optional mask for inpainting (white areas will be replaced)",
                }),
                "model": (["gpt-image-1", "gpt-image-1.5"], {
                    "default": "gpt-image-1.5",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "o1key/image"
    OUTPUT_NODE = False

    def generate(
        self,
        prompt: str,
        seed: int = 0,
        quality: str = "low",
        background: str = "auto",
        size: str = "auto",
        n: int = 1,
        image=None,
        mask=None,
        model: str = "gpt-image-1.5",
    ):
        """
        生成图像（文生图 / 图生图 / 图像编辑）

        路由逻辑：
          - 无 image           → generations 接口（文生图）
          - 有 image，无 mask  → generations 接口（图生图）
          - 有 image，有 mask  → edits 接口（图像编辑 + 蒙版）
        """
        start_time = time.time()

        # ── 1. 参数校验 ───────────────────────────────────────────────────────
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空")

        if mask is not None and image is None:
            raise ValueError("提供了蒙版但未提供图像，请同时提供 image 和 mask")

        # ── 2. 创建客户端 ─────────────────────────────────────────────────────
        try:
            client = GptImageClient()
        except ValueError as e:
            if str(e) == "未授权！":
                print("[o1key GPT Image] 请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            raise

        # ── 3. 调用 API ───────────────────────────────────────────────────────
        try:
            pil_images = client.run_sync(
                prompt=prompt,
                model=model,
                quality=quality,
                background=background,
                size=size,
                n=n,
                seed=seed,
                image_tensor=image,
                mask_tensor=mask,
            )
        except Exception as e:
            error_msg = str(e).split('\n')[0]
            print(f"[o1key GPT Image] ❌ {error_msg}")
            raise RuntimeError(error_msg) from None

        # ── 4. PIL → tensor ───────────────────────────────────────────────────
        output_tensor = GptImageClient._pil_list_to_tensor(pil_images)

        # ── 5. 完成日志 ───────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        print(
            f"[o1key GPT Image] 完成！耗时 {elapsed:.1f}s，"
            f"输出 {output_tensor.shape[0]} 张 "
            f"{output_tensor.shape[2]}×{output_tensor.shape[1]}"
        )

        return (output_tensor,)
