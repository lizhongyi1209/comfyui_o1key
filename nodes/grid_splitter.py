"""
Merged grid image splitter.

This node is designed for AI-generated contact sheets such as 3x3 or 2x3
grids. Auto mode scores common layouts by looking for strong seams or flat
separator bands near the expected grid lines, then crops each cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from ..utils.image_utils import pil_to_tensor, tensor_to_pil


_AUTO_LAYOUTS: Sequence[Tuple[int, int]] = (
    (3, 3),
    (2, 3),
    (3, 2),
    (2, 2),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (4, 4),
    (3, 4),
    (4, 3),
)

_LAYOUTS = [
    "auto",
    "1x2",
    "2x1",
    "1x3",
    "3x1",
    "2x2",
    "2x3",
    "3x2",
    "3x3",
    "3x4",
    "4x3",
    "4x4",
    "custom",
]


@dataclass(frozen=True)
class _AxisCut:
    seam: int
    span_start: int
    span_end: int
    score: float


@dataclass(frozen=True)
class _AxisPlan:
    intervals: List[Tuple[int, int]]
    cuts: List[_AxisCut]
    score: float


def _to_float_array(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0


def _axis_texture(arr: np.ndarray, axis: str) -> np.ndarray:
    if axis == "x":
        profile = arr.std(axis=(0, 2))
    else:
        profile = arr.std(axis=(1, 2))
    high = np.percentile(profile, 95) + 1e-6
    return np.clip(profile / high, 0.0, 1.0)


def _axis_edge(arr: np.ndarray, axis: str) -> np.ndarray:
    if axis == "x":
        diff = np.abs(np.diff(arr, axis=1)).mean(axis=(0, 2))
        length = arr.shape[1]
    else:
        diff = np.abs(np.diff(arr, axis=0)).mean(axis=(1, 2))
        length = arr.shape[0]

    padded = np.zeros(length, dtype=np.float32)
    if diff.size:
        padded[1:] = diff
    high = np.percentile(padded, 95) + 1e-6
    return np.clip(padded / high, 0.0, 1.5)


def _smooth(profile: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0 or profile.size < radius * 2 + 1:
        return profile
    kernel = np.ones(radius * 2 + 1, dtype=np.float32) / float(radius * 2 + 1)
    return np.convolve(profile, kernel, mode="same")


def _separator_span(
    texture: np.ndarray,
    seam: int,
    search_px: int,
    min_separator_px: int,
) -> Tuple[int, int]:
    length = texture.size
    if length <= 1:
        return 0, length

    limit = max(1, min(search_px, length // 8))
    threshold = max(0.08, min(0.28, float(np.percentile(texture, 12)) * 1.8))

    left = seam
    while left > 0 and seam - left < limit and texture[left - 1] <= threshold:
        left -= 1

    right = seam
    while right < length and right - seam < limit and texture[right] <= threshold:
        right += 1

    if right - left >= max(1, min_separator_px):
        return left, right

    return seam, seam


def _edge_trim(texture: np.ndarray, search_px: int, min_cell: int) -> Tuple[int, int]:
    length = texture.size
    if length <= 2:
        return 0, length

    max_trim = max(0, min(search_px * 2, min_cell // 3, length // 6))
    if max_trim <= 0:
        return 0, length

    threshold = max(0.08, min(0.24, float(np.percentile(texture, 12)) * 1.6))

    start = 0
    while start < max_trim and texture[start] <= threshold:
        start += 1

    end = length
    while length - end < max_trim and end > start + min_cell and texture[end - 1] <= threshold:
        end -= 1

    return start, end


def _axis_plan(
    arr: np.ndarray,
    cells: int,
    axis: str,
    search_px: int,
    crop_separators: bool,
    trim_outer: bool,
    min_separator_px: int,
) -> _AxisPlan:
    length = arr.shape[1] if axis == "x" else arr.shape[0]
    if cells <= 1:
        return _AxisPlan(intervals=[(0, length)], cuts=[], score=0.0)

    raw_texture = _axis_texture(arr, axis)
    raw_edge = _axis_edge(arr, axis)
    texture = _smooth(raw_texture, radius=2)
    edge = _smooth(raw_edge, radius=1)
    evidence = np.maximum(edge, (1.0 - texture) * 0.75)
    exact_evidence = np.maximum(raw_edge, (1.0 - raw_texture) * 0.75)

    cuts: List[_AxisCut] = []
    scores: List[float] = []
    for idx in range(1, cells):
        expected = round(length * idx / cells)
        start = max(1, expected - search_px)
        end = min(length - 1, expected + search_px)
        if start >= end:
            seam = expected
            score = 0.0
        else:
            window = evidence[start:end + 1]
            offset = int(window.argmax())
            coarse = start + offset
            fine_start = max(start, coarse - 2)
            fine_end = min(end, coarse + 2)
            fine_window = exact_evidence[fine_start:fine_end + 1]
            seam = fine_start + int(fine_window.argmax())
            score = float(window[offset])

        span_start, span_end = _separator_span(
            raw_texture,
            seam,
            search_px=search_px,
            min_separator_px=min_separator_px,
        )
        cuts.append(_AxisCut(seam=seam, span_start=span_start, span_end=span_end, score=score))
        scores.append(score)

    min_cell = max(1, length // cells)
    outer_start, outer_end = _edge_trim(raw_texture, search_px, min_cell) if trim_outer else (0, length)

    intervals: List[Tuple[int, int]] = []
    cursor = outer_start
    for cut in cuts:
        split_start = cut.span_start if crop_separators else cut.seam
        split_end = cut.span_end if crop_separators else cut.seam
        intervals.append((cursor, split_start))
        cursor = split_end
    intervals.append((cursor, outer_end))

    cleaned: List[Tuple[int, int]] = []
    for start, end in intervals:
        start = max(0, min(length - 1, int(start)))
        end = max(start + 1, min(length, int(end)))
        cleaned.append((start, end))

    return _AxisPlan(
        intervals=cleaned,
        cuts=cuts,
        score=float(np.mean(scores)) if scores else 0.0,
    )


def _parse_layout(layout: str, custom_rows: int, custom_cols: int) -> Tuple[int, int]:
    if layout == "custom":
        return max(1, int(custom_rows)), max(1, int(custom_cols))
    rows_text, cols_text = layout.split("x", 1)
    return int(rows_text), int(cols_text)


def _fallback_layout(width: int, height: int) -> Tuple[int, int]:
    aspect = width / max(1, height)
    if 0.82 <= aspect <= 1.22:
        return 3, 3
    if aspect > 1.22:
        return 2, 3
    return 3, 2


def _choose_auto_layout(
    arr: np.ndarray,
    search_px: int,
    crop_separators: bool,
    trim_outer: bool,
    min_separator_px: int,
) -> Tuple[int, int, _AxisPlan, _AxisPlan, float, bool]:
    height, width = arr.shape[:2]
    best = None

    for rows, cols in _AUTO_LAYOUTS:
        x_plan = _axis_plan(arr, cols, "x", search_px, crop_separators, trim_outer, min_separator_px)
        y_plan = _axis_plan(arr, rows, "y", search_px, crop_separators, trim_outer, min_separator_px)
        score = (x_plan.score + y_plan.score) / 2.0

        # Prefer common 3x3 / 2x3 / 3x2 layouts when the image gives weak signals.
        if (rows, cols) in ((3, 3), (2, 3), (3, 2)):
            score += 0.025

        if best is None or score > best[0]:
            best = (score, rows, cols, x_plan, y_plan)

    assert best is not None
    score, rows, cols, x_plan, y_plan = best
    confident = score >= 0.22

    if confident:
        return rows, cols, x_plan, y_plan, score, True

    rows, cols = _fallback_layout(width, height)
    x_plan = _axis_plan(arr, cols, "x", search_px, crop_separators, trim_outer, min_separator_px)
    y_plan = _axis_plan(arr, rows, "y", search_px, crop_separators, trim_outer, min_separator_px)
    return rows, cols, x_plan, y_plan, score, False


def _normalize_sizes(crops: List[Image.Image]) -> List[Image.Image]:
    min_w = min(crop.width for crop in crops)
    min_h = min(crop.height for crop in crops)
    normalized = []
    for crop in crops:
        left = max(0, (crop.width - min_w) // 2)
        top = max(0, (crop.height - min_h) // 2)
        normalized.append(crop.crop((left, top, left + min_w, top + min_h)))
    return normalized


def _split_one(
    image: Image.Image,
    layout: str,
    custom_rows: int,
    custom_cols: int,
    search_px: int,
    crop_separators: bool,
    trim_outer: bool,
    min_separator_px: int,
) -> Tuple[List[Image.Image], str]:
    arr = _to_float_array(image)

    if layout == "auto":
        rows, cols, x_plan, y_plan, confidence, confident = _choose_auto_layout(
            arr,
            search_px=search_px,
            crop_separators=crop_separators,
            trim_outer=trim_outer,
            min_separator_px=min_separator_px,
        )
        mode_note = "auto" if confident else "auto-low-confidence-fallback"
    else:
        rows, cols = _parse_layout(layout, custom_rows, custom_cols)
        x_plan = _axis_plan(arr, cols, "x", search_px, crop_separators, trim_outer, min_separator_px)
        y_plan = _axis_plan(arr, rows, "y", search_px, crop_separators, trim_outer, min_separator_px)
        confidence = (x_plan.score + y_plan.score) / 2.0
        mode_note = "manual"

    crops: List[Image.Image] = []
    for y0, y1 in y_plan.intervals:
        for x0, x1 in x_plan.intervals:
            crops.append(image.crop((x0, y0, x1, y1)))

    crops = _normalize_sizes(crops)
    info = (
        f"{mode_note}: {rows}x{cols}, cells={len(crops)}, "
        f"confidence={confidence:.3f}, "
        f"x={x_plan.intervals}, y={y_plan.intervals}"
    )
    return crops, info


class O1keyGridSplitter:
    """Split AI-generated grid/contact-sheet images into individual cells."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "布局": (_LAYOUTS, {"default": "auto"}),
                "自定义行数": ("INT", {"default": 3, "min": 1, "max": 12, "step": 1}),
                "自定义列数": ("INT", {"default": 3, "min": 1, "max": 12, "step": 1}),
                "搜索范围px": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1}),
                "裁掉分隔线": ("BOOLEAN", {"default": True}),
                "裁掉外边距": ("BOOLEAN", {"default": True}),
                "最小分隔线px": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1}),
                "最大输出张数": ("INT", {"default": 16, "min": 1, "max": 144, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("切割图像", "检测信息")
    FUNCTION = "split_grid"
    CATEGORY = "o1key/image"
    DESCRIPTION = (
        "智能切割 AI 生成的九宫格、六宫格等合并图。"
        "自动模式会检测常见布局；没有明显分隔线时建议手动选择布局。"
    )

    def split_grid(
        self,
        图像: torch.Tensor,
        布局: str = "auto",
        自定义行数: int = 3,
        自定义列数: int = 3,
        搜索范围px: int = 32,
        裁掉分隔线: bool = True,
        裁掉外边距: bool = True,
        最小分隔线px: int = 2,
        最大输出张数: int = 16,
    ):
        source_images = tensor_to_pil(图像)
        all_crops: List[Image.Image] = []
        info_lines: List[str] = []

        for batch_index, image in enumerate(source_images, start=1):
            crops, info = _split_one(
                image=image,
                layout=布局,
                custom_rows=自定义行数,
                custom_cols=自定义列数,
                search_px=搜索范围px,
                crop_separators=裁掉分隔线,
                trim_outer=裁掉外边距,
                min_separator_px=最小分隔线px,
            )
            if len(crops) > 最大输出张数:
                raise ValueError(
                    f"合并图切割：检测到 {len(crops)} 张，超过最大输出张数 {最大输出张数}。"
                    "请调大最大输出张数，或检查布局设置。"
                )
            all_crops.extend(crops)
            info_lines.append(f"batch {batch_index}: {info}")

        if not all_crops:
            raise ValueError("合并图切割：没有生成任何切片。")

        all_crops = _normalize_sizes(all_crops)
        print("[o1key 合并图切割] " + " | ".join(info_lines))
        return (pil_to_tensor(all_crops), "\n".join(info_lines))
