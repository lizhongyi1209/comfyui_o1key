"""
LoadFile 节点
ComfyUI 自定义节点,用于加载文件并转换为 FILE 类型数据
"""

import base64
import os
from pathlib import Path
from typing import Tuple

from ..utils.file_types import FileData, DOCUMENT_MIME_TYPES, FILE_SIZE_LIMITS


class LoadFile:
    """
    LoadFile 节点
    
    功能：
    - 从文件系统加载文件
    - 支持 PDF 和 TXT 文件
    - 转换为 FILE 类型数据（包含 base64 编码内容）
    - 验证文件大小和格式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        定义输入参数
        """
        return {
            "required": {
                "文件路径": ("STRING", {
                    "default": "",
                    "multiline": False
                })
            }
        }
    
    # 返回值类型
    RETURN_TYPES = ("FILE", "STRING")
    RETURN_NAMES = ("文件", "文件信息")
    
    # 执行函数名
    FUNCTION = "load_file"
    
    # 节点分类
    CATEGORY = "file/input"
    
    def load_file(self, 文件路径: str) -> Tuple[FileData, str]:
        """
        加载文件并转换为 FILE 类型
        
        Args:
            文件路径: 文件的完整路径（支持绝对路径和相对路径）
        
        Returns:
            (FileData, 文件信息预览)
        
        Raises:
            ValueError: 文件不存在、不支持的文件类型或文件过大
        """
        try:
            # 清理路径（去除空格和引号）
            file_path = 文件路径.strip().strip('"').strip("'")
            
            if not file_path:
                raise ValueError("文件路径不能为空")
            
            # 转换为 Path 对象
            path = Path(file_path)
            
            # 如果是相对路径，转换为绝对路径
            if not path.is_absolute():
                # 相对于当前工作目录
                path = Path.cwd() / path
            
            # 验证文件是否存在
            if not path.exists():
                raise ValueError(f"文件不存在: {file_path}")
            
            if not path.is_file():
                raise ValueError(f"路径不是文件: {file_path}")
            
            # 获取文件信息
            extension = path.suffix.lower()
            filename = path.stem
            file_size = path.stat().st_size
            
            # 验证文件类型
            if extension not in DOCUMENT_MIME_TYPES:
                supported_types = ", ".join(DOCUMENT_MIME_TYPES.keys())
                raise ValueError(
                    f"不支持的文件类型: {extension}\n"
                    f"支持的类型: {supported_types}"
                )
            
            # 获取 MIME 类型
            mime_type = DOCUMENT_MIME_TYPES[extension]
            
            # 验证文件大小
            size_limit = FILE_SIZE_LIMITS.get(extension, 20 * 1024 * 1024)
            if file_size > size_limit:
                raise ValueError(
                    f"文件过大 ({file_size / 1024 / 1024:.2f}MB)，"
                    f"最大支持 {size_limit / 1024 / 1024:.0f}MB"
                )
            
            # 读取文件并转换为 base64
            print(f"LoadFile: 正在加载文件 {filename}{extension}")
            print(f"LoadFile: 文件大小 = {file_size / 1024:.2f}KB")
            
            with open(path, "rb") as f:
                file_bytes = f.read()
            
            # Base64 编码
            b64_str = base64.b64encode(file_bytes).decode("utf-8")
            
            # 创建 FileData 对象
            file_data = FileData(
                path=str(path),
                filename=filename,
                extension=extension,
                mime_type=mime_type,
                data=b64_str,
                size=file_size
            )
            
            # 生成文件信息预览
            file_info = (
                f"文件名: {filename}{extension}\n"
                f"类型: {mime_type}\n"
                f"大小: {file_size / 1024:.2f}KB\n"
                f"路径: {path}"
            )
            
            print(f"LoadFile: 加载成功")
            
            return (file_data, file_info)
        
        except ValueError as e:
            print(f"LoadFile: 输入错误 - {str(e)}")
            raise
        
        except Exception as e:
            print(f"LoadFile: 未知错误 - {str(e)}")
            raise
