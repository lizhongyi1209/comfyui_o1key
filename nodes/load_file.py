"""
LoadFile 节点（增强版）
支持单文件路径和文件夹路径，输出 FILE_LIST 类型供全能LLM等节点使用
"""

import base64
import os
from pathlib import Path
from typing import Tuple, List

from ..utils.file_types import FileData, FileList, DOCUMENT_MIME_TYPES, FILE_SIZE_LIMIT, TOTAL_FILE_SIZE_LIMIT


class LoadFile:
    """
    加载文件节点

    - 单文件路径：加载指定文件
    - 文件夹路径：加载文件夹内所有支持的文件（非递归）
    - 两者可同时使用，结果合并输出
    - 输出 FILE_LIST 类型，可直接连接到全能LLM对话助手
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "单文件路径": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "文件完整路径，多个文件用英文逗号分隔",
                }),
                "文件夹路径": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "文件夹路径，自动读取其中所有支持的文件",
                }),
            },
        }

    RETURN_TYPES = ("FILE_LIST", "STRING")
    RETURN_NAMES = ("文件列表", "文件信息")
    FUNCTION = "load_file"
    CATEGORY = "file/input"

    def load_file(self, 单文件路径: str = "", 文件夹路径: str = "") -> Tuple[FileList, str]:
        collected: List[Path] = []

        # 1. 单文件路径（逗号分隔，支持多个）
        if 单文件路径.strip():
            for raw in 单文件路径.split(","):
                p = Path(raw.strip().strip('"').strip("'"))
                if not p.is_absolute():
                    p = Path.cwd() / p
                if not p.exists():
                    raise ValueError(f"文件不存在: {p}")
                if not p.is_file():
                    raise ValueError(f"路径不是文件: {p}")
                collected.append(p)

        # 2. 文件夹路径
        if 文件夹路径.strip():
            folder = Path(文件夹路径.strip().strip('"').strip("'"))
            if not folder.is_absolute():
                folder = Path.cwd() / folder
            if not folder.exists():
                raise ValueError(f"文件夹不存在: {folder}")
            if not folder.is_dir():
                raise ValueError(f"路径不是文件夹: {folder}")
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in DOCUMENT_MIME_TYPES:
                    collected.append(p)
            if not collected:
                raise ValueError(f"文件夹中没有支持的文件: {folder}")

        if not collected:
            raise ValueError("请至少提供一个文件路径或文件夹路径")

        # 去重（保持顺序）
        seen = set()
        unique: List[Path] = []
        for p in collected:
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                unique.append(p)

        # 大小检查 & 读取
        total_size = 0
        file_list: FileList = []
        info_lines = []

        for p in unique:
            ext = p.suffix.lower()
            if ext not in DOCUMENT_MIME_TYPES:
                print(f"LoadFile: 跳过不支持的文件类型 {p.name}")
                continue

            file_size = p.stat().st_size
            if file_size > FILE_SIZE_LIMIT:
                raise ValueError(
                    f"文件 {p.name} 大小 {file_size / 1024 / 1024:.1f}MB 超过单文件 50MB 限制"
                )
            total_size += file_size
            if total_size > TOTAL_FILE_SIZE_LIMIT:
                raise ValueError(f"所有文件总大小超过 50MB 限制")

            mime = DOCUMENT_MIME_TYPES[ext]
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            file_list.append(FileData(
                path=str(p),
                filename=p.stem,
                extension=ext,
                mime_type=mime,
                data=b64,
                size=file_size,
            ))
            info_lines.append(f"  {p.name} ({file_size / 1024:.1f}KB, {mime})")
            print(f"LoadFile: 加载 {p.name} ({file_size / 1024:.1f}KB)")

        info = f"共 {len(file_list)} 个文件，总大小 {total_size / 1024:.1f}KB\n" + "\n".join(info_lines)
        return (file_list, info)
