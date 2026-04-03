"""
多分辨率图像预览节点
ComfyUI 自定义节点，支持同时预览多张不同分辨率的图像

背景：
  ComfyUI 原生「预览图像」节点要求 batch 内所有图片分辨率相同（因为它们被
  stack 成一个 [B, H, W, C] tensor）。当 API 返回多张不同尺寸的图片时
  （例如 nano-banana-2 同时返回 1K + 2K），原生节点会报错。

解决方案：
  声明 INPUT_IS_LIST = True，ComfyUI 会将连入的所有图像作为
  Python list[Tensor] 传入，而不是强行 stack 成单个 tensor。
  节点逐张单独保存为临时 PNG，再通过 ui.images 列表返回给前端并列展示，
  完全不受分辨率一致性的限制。
"""

import os
import uuid
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False


def _get_temp_dir() -> str:
    """获取 ComfyUI temp 目录，不可用时回退到系统临时目录"""
    if FOLDER_PATHS_AVAILABLE:
        return folder_paths.get_temp_directory()
    import tempfile
    return tempfile.gettempdir()


def _tensor_to_pil(tensor) -> list:
    """
    将单个 IMAGE tensor 转换为 PIL Image 列表。

    ComfyUI IMAGE tensor 格式：[B, H, W, C]，float32，值域 [0, 1]
    支持：
      - 单张图 tensor: shape [H, W, C] 或 [1, H, W, C]
      - batch tensor:  shape [B, H, W, C]（B 张相同尺寸图）
    """
    import torch
    if not isinstance(tensor, torch.Tensor):
        return []

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    results = []
    for i in range(tensor.shape[0]):
        img_np = tensor[i].cpu().numpy()
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        results.append(Image.fromarray(img_np))
    return results


class MultiResPreview:
    """
    多分辨率图像预览节点

    功能：
    - 单个「图像」输入端口，支持接入批次图像
    - INPUT_IS_LIST = True：ComfyUI 将每张图作为独立 tensor 传入，
      不强制要求尺寸相同，彻底解决不同分辨率无法共存的问题
    - 每张图像独立保存为临时 PNG，在节点上并列展示所有图像

    用法：
      将 Nano Banana 节点的输出直接连入「图像」端口即可，
      无论返回几张、分辨率是否相同，都能正确展示。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    # 关键：告知 ComfyUI 以 list[Tensor] 而非 stacked Tensor 传入图像
    # 这样不同分辨率的图片可以共存于同一个输入中
    INPUT_IS_LIST = True

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "image"

    DESCRIPTION = (
        "多分辨率图像预览节点。\n"
        "单个图像输入端口，支持任意数量、任意分辨率的批次图像。\n"
        "解决了原生「预览图像」节点要求 batch 内图片尺寸相同的限制。\n"
        "常用场景：nano-banana-2 同时返回 1K + 2K 图时，直接连入本节点即可。"
    )

    def preview(self, 图像, prompt=None, extra_pnginfo=None) -> dict:
        """
        逐张将图像保存到 temp 目录，返回 ui.images 供前端展示。

        Args:
            图像:          list[Tensor]，每个元素是一张或一批图（INPUT_IS_LIST）
            prompt:        ComfyUI 注入的 prompt 元数据（可选）
            extra_pnginfo: ComfyUI 注入的额外 PNG 信息（可选）

        Returns:
            {"ui": {"images": [...]}} 格式，每项对应一张图
        """
        temp_dir = _get_temp_dir()
        os.makedirs(temp_dir, exist_ok=True)

        # 构建 PNG 元数据（与原生预览节点行为一致）
        metadata = PngInfo()
        # INPUT_IS_LIST 时 hidden 值也会被包装成 list，取第一个元素
        _prompt = prompt[0] if isinstance(prompt, list) else prompt
        _extra = extra_pnginfo[0] if isinstance(extra_pnginfo, list) else extra_pnginfo
        if _prompt is not None:
            try:
                metadata.add_text("prompt", json.dumps(_prompt))
            except Exception:
                pass
        if _extra is not None:
            try:
                for k, v in _extra.items():
                    metadata.add_text(k, json.dumps(v))
            except Exception:
                pass

        saved = []
        total_input = 0
        total_saved = 0

        # 图像 是 list[Tensor]，逐个处理（每个 Tensor 可能自身是个 batch）
        for tensor in 图像:
            pil_images = _tensor_to_pil(tensor)
            total_input += len(pil_images)

            for pil_img in pil_images:
                try:
                    filename = f"multi_res_preview_{uuid.uuid4().hex[:12]}.png"
                    filepath = os.path.join(temp_dir, filename)
                    pil_img.save(filepath, pnginfo=metadata, compress_level=1)

                    saved.append({
                        "filename": filename,
                        "subfolder": "",
                        "type": "temp",
                    })
                    total_saved += 1
                except Exception as e:
                    print(f"多分辨率预览: ⚠️ 保存图像失败 - {e}")

        if total_input == 0:
            print("多分辨率预览: ⚠️ 没有接收到任何图像")

        return {"ui": {"images": saved}}
