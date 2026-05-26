"""
颜色去背景工具模块
基于颜色距离计算实现精确可控的背景移除，不依赖 AI 模型。

支持模式：
  - white: 白色背景去除
  - white-preserve: 白色背景但保护浅色前景物体
  - corner: 自动采样四角颜色作为背景色
  - color: 指定任意颜色去除
"""

import numpy as np
from PIL import Image


def background_to_alpha(
    image: Image.Image,
    bg_color: tuple = (255, 255, 255),
    tolerance: float = 8.0,
    feather: float = 45.0,
    strength: float = 1.0,
    min_alpha: int = 2,
) -> Image.Image:
    """
    将纯色背景转为透明。

    对白色背景使用 white-to-alpha 恢复算法，保持彩色文字和抗锯齿边缘清晰。
    对其他颜色使用欧氏距离计算。
    """
    rgba = np.asarray(image.convert("RGBA")).astype(np.float32)
    rgb = rgba[:, :, :3] / 255.0
    existing_alpha = rgba[:, :, 3] / 255.0
    bg = np.array(bg_color, dtype=np.float32) / 255.0

    if max(bg_color) >= 245 and min(bg_color) >= 245:
        alpha = (1.0 - np.min(rgb, axis=2)) * float(strength)
        if tolerance > 0:
            dist = np.linalg.norm((1.0 - rgb) * 255.0, axis=2)
            gate = np.clip(
                (dist - float(tolerance)) / max(1.0, float(feather) * 0.25),
                0.0, 1.0,
            )
            alpha *= gate
    else:
        dist = np.linalg.norm((rgb - bg) * 255.0, axis=2)
        denom = max(1.0, float(feather))
        alpha = np.clip((dist - float(tolerance)) / denom, 0.0, 1.0)
        alpha *= float(strength)

    alpha = np.clip(alpha, 0.0, 1.0) * existing_alpha
    alpha[alpha < (float(min_alpha) / 255.0)] = 0.0

    # 从 alpha 混合中恢复前景色，避免白边
    out_rgb = rgb.copy()
    mask = alpha > 1e-6
    out_rgb[mask] = (rgb[mask] - bg * (1.0 - alpha[mask, None])) / alpha[mask, None]
    out_rgb = np.clip(out_rgb, 0.0, 1.0)

    out = np.dstack([
        (out_rgb * 255.0).astype(np.uint8),
        (alpha * 255.0).astype(np.uint8),
    ])
    return Image.fromarray(out, "RGBA")


def corner_color(image: Image.Image, sample: int = 12) -> tuple:
    """采样图片四角像素的中位数颜色，用于自动检测背景色。"""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    sample = max(1, min(sample, h, w))
    patches = [
        rgb[:sample, :sample],
        rgb[:sample, w - sample:],
        rgb[h - sample:, :sample],
        rgb[h - sample:, w - sample:],
    ]
    merged = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    return tuple(np.median(merged, axis=0).astype(int))


# PLACEHOLDER_PRESERVE

def preserve_light_foreground_to_alpha(
    image: Image.Image,
    tolerance: float = 10.0,
    preserve_opacity: float = 0.72,
    min_area_ratio: float = 0.00025,
) -> Image.Image:
    """
    白底去除 + 浅色前景保护。

    适用于前景包含白色/浅色物体（白盘子、白帆、白色包装）的场景。
    使用 OpenCV 连通区域分析保护大面积浅色前景结构。
    如果 OpenCV 不可用，回退到普通 white-to-alpha。
    """
    base = background_to_alpha(image, (255, 255, 255), tolerance=tolerance)
    try:
        import cv2
    except ImportError:
        return base

    rgb_u8 = np.asarray(image.convert("RGB"))
    h, w = rgb_u8.shape[:2]
    dist = np.sqrt(np.sum((255.0 - rgb_u8.astype(np.float32)) ** 2, axis=2))
    rough = (dist > float(tolerance)).astype(np.uint8) * 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    rough = cv2.morphologyEx(rough, cv2.MORPH_OPEN, kernel_open, iterations=1)
    rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(rough, 8)
    keep = np.zeros_like(rough)
    min_area = max(24, int(w * h * float(min_area_ratio)))
    for idx in range(1, count):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == idx] = 255

    # PLACEHOLDER_FLOOD

    flood = keep.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    filled = cv2.bitwise_or(keep, cv2.bitwise_not(flood))
    soft = cv2.GaussianBlur(filled, (0, 0), 5).astype(np.float32) / 255.0

    near_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
    near = cv2.dilate(
        (dist > (float(tolerance) * 0.65)).astype(np.uint8) * 255,
        near_kernel, iterations=1,
    )
    near = cv2.GaussianBlur(near, (0, 0), 8).astype(np.float32) / 255.0
    lift = np.minimum(soft, near) * float(preserve_opacity)

    arr = np.asarray(base.convert("RGBA")).copy()
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    alpha = np.maximum(alpha, lift)
    alpha[alpha < (2.0 / 255.0)] = 0.0

    original = np.asarray(image.convert("RGB"))
    very_light = (np.mean(original, axis=2) > 224) & (lift > 0.12)
    arr[:, :, :3][very_light] = original[very_light]
    arr[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def remove_background(
    image: Image.Image,
    mode: str = "white",
    bg_color: tuple = (255, 255, 255),
    tolerance: float = 8.0,
    feather: float = 45.0,
    strength: float = 1.0,
) -> Image.Image:
    """
    统一入口：根据模式移除背景。

    mode:
      - white: 白色背景去除
      - white-preserve: 白底 + 保护浅色前景
      - corner: 自动采样四角颜色
      - color: 使用指定 bg_color
    """
    if mode == "white":
        return background_to_alpha(image, (255, 255, 255), tolerance, feather, strength)
    elif mode == "white-preserve":
        return preserve_light_foreground_to_alpha(image, tolerance)
    elif mode == "corner":
        bg = corner_color(image)
        return background_to_alpha(image, bg, tolerance, feather, strength)
    elif mode == "color":
        return background_to_alpha(image, bg_color, tolerance, feather, strength)
    else:
        return image.convert("RGBA")
