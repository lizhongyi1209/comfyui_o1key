"""
文件数据类型定义
用于在 ComfyUI 节点间传递文件数据
"""

from typing import NamedTuple


class FileData(NamedTuple):
    """
    文件数据类型，用于在节点间传递
    
    Attributes:
        path: 文件完整路径
        filename: 文件名（不含扩展名）
        extension: 文件扩展名（如 .pdf）
        mime_type: MIME 类型
        data: Base64 编码的文件内容
        size: 文件大小（字节）
    """
    path: str
    filename: str
    extension: str
    mime_type: str
    data: str
    size: int


# 支持的文档 MIME 类型映射
DOCUMENT_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain"
}


# 文件大小限制（字节）
FILE_SIZE_LIMITS = {
    ".pdf": 50 * 1024 * 1024,  # 50MB (Gemini API 官方限制)
    ".txt": 20 * 1024 * 1024   # 20MB (保守限制)
}
