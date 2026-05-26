"""
o1key SavePSD 节点
将多个 IMAGE 图层合成为分层 PSD 文件
手写 PSD 二进制格式，零外部依赖（仅 numpy + Pillow）
"""

import os
import struct
import time
import numpy as np
import torch
from PIL import Image

import folder_paths


def _pad_even(data: bytes) -> bytes:
    if len(data) % 2:
        return data + b"\x00"
    return data


def _pad4(data: bytes) -> bytes:
    return data + (b"\x00" * ((4 - (len(data) % 4)) % 4))


def _pascal_name(name: str) -> bytes:
    raw = name.encode("macroman", errors="replace")[:255]
    data = bytes([len(raw)]) + raw
    return _pad4(data)


def _unicode_name_block(name: str) -> bytes:
    payload = struct.pack(">I", len(name)) + name.encode("utf-16be")
    block = b"8BIM" + b"luni" + struct.pack(">I", len(payload)) + _pad_even(payload)
    return block


def _layer_extra_data(name: str) -> bytes:
    data = b""
    data += struct.pack(">I", 0)  # layer mask data length
    data += struct.pack(">I", 0)  # layer blending ranges length
    data += _pascal_name(name)
    data += _unicode_name_block(name)
    return data


def _alpha_bbox(rgba_arr: np.ndarray):
    """找到 RGBA 数组中非透明区域的 bounding box。"""
    alpha = rgba_arr[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any():
        return None
    top = int(np.argmax(rows))
    bottom = int(len(rows) - np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = int(len(cols) - np.argmax(cols[::-1]))
    return top, left, bottom, right


def write_psd(filepath: str, layers: list, canvas_w: int, canvas_h: int):
    """
    写入 PSD 文件。

    layers: [(name, rgba_array), ...] 从底到顶排列
    rgba_array: numpy uint8 [H, W, 4]
    """
    records = []
    channel_data_blocks = []
    layers_top_to_bottom = list(reversed(layers))

    for name, rgba in layers_top_to_bottom:
        bbox = _alpha_bbox(rgba)
        if not bbox:
            continue
        top, left, bottom, right = bbox
        cropped = rgba[top:bottom, left:right]

        # PLACEHOLDER_CHANNELS

        channels = [
            (0, cropped[:, :, 0].tobytes(order="C")),
            (1, cropped[:, :, 1].tobytes(order="C")),
            (2, cropped[:, :, 2].tobytes(order="C")),
            (-1, cropped[:, :, 3].tobytes(order="C")),
        ]
        channel_info = b""
        data_block = b""
        for channel_id, data in channels:
            channel_info += struct.pack(">hI", channel_id, 2 + len(data))
            data_block += struct.pack(">H", 0) + data  # raw compression

        extra = _layer_extra_data(name)
        record = b""
        record += struct.pack(">iiii", top, left, bottom, right)
        record += struct.pack(">H", len(channels))
        record += channel_info
        record += b"8BIM" + b"norm"
        record += bytes([255, 0, 0, 0])  # opacity=255, clipping, flags, filler
        record += struct.pack(">I", len(extra)) + extra
        records.append(record)
        channel_data_blocks.append(data_block)

    if not records:
        raise ValueError("所有图层均为空（完全透明），无法生成 PSD")

    # Layer and Mask Information
    layer_info = struct.pack(">h", len(records))
    layer_info += b"".join(records) + b"".join(channel_data_blocks)
    layer_info = _pad_even(layer_info)
    layer_info_block = struct.pack(">I", len(layer_info)) + layer_info
    global_mask = struct.pack(">I", 0)
    layer_mask_payload = layer_info_block + global_mask
    layer_and_mask = struct.pack(">I", len(layer_mask_payload)) + layer_mask_payload

    # PLACEHOLDER_COMPOSITE

    # Composite preview (flattened image for compatibility)
    comp = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    for name, rgba in layers:
        layer_img = Image.fromarray(rgba, "RGBA")
        comp.alpha_composite(layer_img)
    comp_rgb = np.asarray(comp.convert("RGB"), dtype=np.uint8)
    composite_data = (
        struct.pack(">H", 0)
        + comp_rgb[:, :, 0].tobytes(order="C")
        + comp_rgb[:, :, 1].tobytes(order="C")
        + comp_rgb[:, :, 2].tobytes(order="C")
    )

    # Write PSD file
    with open(filepath, "wb") as f:
        # Header
        f.write(b"8BPS")
        f.write(struct.pack(">H", 1))  # version
        f.write(b"\x00" * 6)  # reserved
        f.write(struct.pack(">HIIHH", 3, canvas_h, canvas_w, 8, 3))
        # Color Mode Data
        f.write(struct.pack(">I", 0))
        # Image Resources
        f.write(struct.pack(">I", 0))
        # Layer and Mask
        f.write(layer_and_mask)
        # Composite Image Data
        f.write(composite_data)


# PLACEHOLDER_NODE

class O1keySavePSD:
    """
    将多个 IMAGE 输入合成为分层 PSD 文件

    每个输入作为独立图层，支持 RGBA 透明通道。
    图层从下到上排列（图层1在最底部）。
    使用 bbox 裁剪优化文件大小，包含合成预览层。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "批次图像": ("IMAGE", {
                    "tooltip": "批次图像输入，每张图自动作为独立图层（支持RGBA透明）",
                }),
            },
            "optional": {
                "图层名称": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "每行一个图层名称，与图层顺序对应。留空则自动命名。",
                }),
                "文件名前缀": ("STRING", {
                    "default": "o1key_layers",
                    "tooltip": "输出 PSD 文件名前缀",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文件路径",)
    FUNCTION = "save_psd"
    CATEGORY = "o1key/image"
    OUTPUT_NODE = True

    def save_psd(self, 批次图像, 图层名称: str = "", 文件名前缀: str = "o1key_layers", **kwargs):
        # 将批次 tensor [B, H, W, C] 拆为单张列表
        if 批次图像.dim() == 3:
            layer_tensors = [批次图像]
        else:
            layer_tensors = [批次图像[i] for i in range(批次图像.shape[0])]

        names = [n.strip() for n in 图层名称.split("\n") if n.strip()]

        # 确定画布尺寸
        max_h, max_w = 0, 0
        for t in layer_tensors:
            h, w = t.shape[0], t.shape[1]
            max_h = max(max_h, h)
            max_w = max(max_w, w)

        # 转换为 [(name, rgba_array), ...] 格式
        layers = []
        for idx, tensor in enumerate(layer_tensors):
            arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            h, w = arr.shape[0], arr.shape[1]
            channels = arr.shape[2] if arr.ndim == 3 else 1

            if channels == 3:
                rgba = np.zeros((max_h, max_w, 4), dtype=np.uint8)
                rgba[:h, :w, :3] = arr
                rgba[:h, :w, 3] = 255
            elif channels == 4:
                rgba = np.zeros((max_h, max_w, 4), dtype=np.uint8)
                rgba[:h, :w] = arr
            else:
                rgba = np.zeros((max_h, max_w, 4), dtype=np.uint8)
                rgba[:h, :w, 0] = rgba[:h, :w, 1] = rgba[:h, :w, 2] = arr[:, :, 0] if arr.ndim == 3 else arr
                rgba[:h, :w, 3] = 255

            name = names[idx] if idx < len(names) else f"图层 {idx + 1}"
            layers.append((name, rgba))
            print(f"[o1key SavePSD] 图层 '{name}': {w}×{h}")

        # 写入 PSD
        output_dir = folder_paths.get_output_directory()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{文件名前缀}_{timestamp}.psd"
        filepath = os.path.join(output_dir, filename)

        write_psd(filepath, layers, max_w, max_h)

        size_kb = os.path.getsize(filepath) / 1024
        print(f"[o1key SavePSD] 完成: {filepath} ({size_kb:.0f}KB, "
              f"{len(layers)} 层, {max_w}×{max_h})")
        return (filepath,)
