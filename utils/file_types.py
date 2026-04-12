"""
文件数据类型定义
用于在 ComfyUI 节点间传递文件数据
"""

from typing import NamedTuple, List


class FileData(NamedTuple):
    """
    单个文件数据，用于节点间传递

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


# FILE_LIST 类型：FileData 的列表，用于多文件传递
# ComfyUI 自定义类型名，节点 RETURN_TYPES / INPUT_TYPES 中使用 "FILE_LIST"
FileList = List[FileData]


# 支持的文件 MIME 类型映射（与 universal_llm.py 的 MIME_MAP 保持一致）
DOCUMENT_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/javascript",
    ".html": "text/html",
    ".xml": "application/xml",
    ".yaml": "text/plain",
    ".yml": "text/plain",
    ".toml": "text/plain",
    ".ini": "text/plain",
    ".cfg": "text/plain",
    ".log": "text/plain",
    ".sh": "text/plain",
    ".bat": "text/plain",
    ".sql": "text/plain",
    ".css": "text/plain",
    ".scss": "text/plain",
    ".jsx": "text/javascript",
    ".tsx": "text/javascript",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".zip": "application/zip",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
}

# 单文件大小上限：50MB
FILE_SIZE_LIMIT = 50 * 1024 * 1024

# 所有文件总大小上限：50MB
TOTAL_FILE_SIZE_LIMIT = 50 * 1024 * 1024

# 兼容旧代码
FILE_SIZE_LIMITS = {
    ".pdf": FILE_SIZE_LIMIT,
    ".txt": FILE_SIZE_LIMIT,
}
