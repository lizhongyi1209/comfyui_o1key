"""
Google Gemini 节点
ComfyUI 自定义节点,用于调用 Gemini Flash 模型进行多模态文本生成
"""

import base64
import os
import time
from typing import Dict, List, Optional, Tuple

import torch

from ..utils.image_utils import tensor_to_pil, encode_image_to_base64
from ..utils.file_types import FileData
from ..clients.gemini_flash_client import GeminiFlashClient
from ..models_config import get_enabled_flash_models


# 支持的视频 MIME 类型映射
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpg",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".3gp": "video/3gpp",
    ".3gpp": "video/3gpp"
}


class GoogleGemini:
    """
    Google Gemini 节点
    
    功能：
    - 支持多个 Gemini Flash 模型
    - 支持图片、视频和文件输入
    - 支持不同思考等级（不思考/低/中/高）- 通过 thinkingConfig.thinkingLevel 控制
    - 输出生成的文本内容（主要内容 + 思考内容）
    """
    
    # 支持的思考等级选项
    THINKING_LEVELS = ["不思考", "低", "中", "高"]
    
    def __init__(self):
        """初始化节点"""
        self.client = None
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        定义输入参数
        """
        # 从配置获取启用的模型列表
        enabled_models = get_enabled_flash_models()
        default_model = enabled_models[0] if enabled_models else "gemini-3-flash-preview"
        
        return {
            "required": {
                "模型": (enabled_models, {
                    "default": default_model
                }),
                "提示词": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "思考等级": (cls.THINKING_LEVELS, {
                    "default": "不思考"
                })
            },
            "optional": {
                "图片": ("IMAGE",),
                "视频": ("VIDEO",),
                "文件": ("FILE",)
            }
        }
    
    # 返回值类型
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("主要内容", "思考内容")
    
    # 执行函数名
    FUNCTION = "generate"
    
    # 节点分类
    CATEGORY = "text/generation"
    
    # 允许输出到 UI
    OUTPUT_NODE = True
    
    def _prepare_image_data(
        self, 
        images: Optional[torch.Tensor]
    ) -> Optional[List[Dict[str, str]]]:
        """
        准备图片数据
        
        Args:
            images: ComfyUI 图片张量 [B, H, W, C]
        
        Returns:
            图片数据列表，每个元素包含 mime_type 和 data
        """
        if images is None:
            return None
        
        image_data = []
        pil_images = tensor_to_pil(images)
        
        for img in pil_images:
            b64_str = encode_image_to_base64(img)
            image_data.append({
                "mime_type": "image/png",
                "data": b64_str
            })
        
        return image_data if image_data else None
    
    def _prepare_video_data(
        self, 
        video
    ) -> Optional[Dict[str, str]]:
        """
        准备视频数据
        
        ComfyUI VIDEO 类型包含视频文件路径信息。
        读取视频文件并转换为 base64。
        
        Args:
            video: ComfyUI VIDEO 类型数据
        
        Returns:
            视频数据字典，包含 mime_type 和 data
        """
        if video is None:
            return None
        
        # VIDEO 类型处理：支持多种格式
        video_path = None
        
        if isinstance(video, dict):
            # 字典格式：尝试常见的键名
            video_path = video.get("video") or video.get("path") or video.get("file") or video.get("filename")
            # 如果还是找不到，遍历所有键找到有效路径
            if not video_path:
                for key, val in video.items():
                    if isinstance(val, str) and os.path.exists(val):
                        video_path = val
                        break
        elif isinstance(video, str):
            # 字符串格式：直接作为路径
            video_path = video
        else:
            # 对象格式：尝试常见属性
            # 1. 尝试 __file 属性（VideoFromFile 对象）
            if hasattr(video, "__file"):
                video_path = video.__file
            # 2. 尝试其他常见属性
            elif hasattr(video, "video"):
                video_path = video.video
            elif hasattr(video, "path"):
                video_path = video.path
            elif hasattr(video, "filename"):
                video_path = video.filename
            # 3. 尝试从 __dict__ 中查找路径（支持私有属性如 _VideoFromFile__file）
            elif hasattr(video, "__dict__"):
                for attr_name, attr_value in video.__dict__.items():
                    # 查找字符串类型的属性，且包含 file 或 path 关键字
                    if isinstance(attr_value, str):
                        if "file" in attr_name.lower() or "path" in attr_name.lower():
                            # 验证路径是否有效
                            if os.path.exists(attr_value):
                                video_path = attr_value
                                break
                        # 如果属性值本身看起来像文件路径，也尝试使用
                        elif os.path.exists(attr_value) and os.path.isfile(attr_value):
                            video_path = attr_value
                            break
        
        if not video_path or not os.path.exists(video_path):
            print(f"Google Gemini: 视频文件不存在或路径无效: {video_path}")
            return None
        
        # 获取文件扩展名和 MIME 类型
        _, ext = os.path.splitext(video_path)
        ext = ext.lower()
        
        mime_type = VIDEO_MIME_TYPES.get(ext, "video/mp4")
        
        # 检查文件大小（限制 20MB）
        file_size = os.path.getsize(video_path)
        if file_size > 20 * 1024 * 1024:
            raise ValueError(
                f"视频文件过大 ({file_size / 1024 / 1024:.2f}MB)，"
                f"请使用不超过 20MB 的视频文件"
            )
        
        # 读取并编码视频
        try:
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            
            b64_str = base64.b64encode(video_bytes).decode("utf-8")
            
            return {
                "mime_type": mime_type,
                "data": b64_str
            }
        
        except Exception as e:
            print(f"Google Gemini: 读取视频文件失败 - {str(e)}")
            return None
    
    def _prepare_file_data(
        self, 
        file: Optional[FileData]
    ) -> Optional[Dict[str, str]]:
        """
        准备文件数据
        
        从 FILE 类型提取文件数据
        
        Args:
            file: FileData 对象（来自 LoadFile 节点）
        
        Returns:
            文件数据字典，包含 mime_type 和 data
        """
        if file is None:
            return None
        
        return {
            "mime_type": file.mime_type,
            "data": file.data
        }
    
    def _parse_dual_output(self, raw_response: Dict) -> Tuple[str, str]:
        """
        解析包含思考内容和主要内容的响应
        
        Args:
            raw_response: API 原始响应字典
        
        Returns:
            (主要内容, 思考内容)
        """
        candidates = raw_response.get("candidates", [])
        if not candidates:
            return ("", "")
        
        parts = candidates[0].get("content", {}).get("parts", [])
        
        thought_text = ""
        main_text = ""
        
        for part in parts:
            if part.get("thought") is True:
                # 思考部分
                thought_text = part.get("text", "")
            elif "thoughtSignature" in part or "text" in part:
                # 主要内容
                main_text = part.get("text", "")
        
        return (main_text, thought_text)
    
    def generate(
        self,
        模型: str,
        提示词: str,
        思考等级: str,
        图片: Optional[torch.Tensor] = None,
        视频=None,
        文件: Optional[FileData] = None
    ) -> Tuple[str]:
        """
        生成文本
        
        Args:
            模型: 使用的模型名称
            提示词: 用户提示词
            思考等级: 思考等级选项
            图片: 输入图片
            视频: 输入视频
            文件: 输入文件（PDF/TXT）
        
        Returns:
            (主要内容, 思考内容)
        """
        start_time = time.time()
        
        try:
            # 初始化 API 客户端
            if self.client is None:
                try:
                    self.client = GeminiFlashClient()
                except ValueError as e:
                    raise ValueError(f"初始化失败: {str(e)}")
            
            # 准备图片数据
            image_data = self._prepare_image_data(图片)
            if image_data:
                print(f"Google Gemini: 输入 {len(image_data)} 张图片")
            
            # 准备视频数据
            video_data = self._prepare_video_data(视频)
            if video_data:
                print(f"Google Gemini: 输入视频 ({video_data['mime_type']})")
            
            # 准备文件数据
            document_data = self._prepare_file_data(文件)
            if document_data:
                file_type = "PDF" if document_data['mime_type'] == "application/pdf" else "TXT"
                print(f"Google Gemini: 输入文件 ({file_type})")
            
            # 构建输入描述
            input_desc = []
            if 提示词:
                input_desc.append("文本")
            if image_data:
                input_desc.append(f"{len(image_data)}张图片")
            if video_data:
                input_desc.append("视频")
            if document_data:
                input_desc.append("文件")
            
            print(f"Google Gemini: 模型 = {模型}")
            print(f"Google Gemini: 多模态输入 ({', '.join(input_desc)})")
            print(f"Google Gemini: 思考等级 = {思考等级}")
            
            # 获取端点和构建请求体
            endpoint = self.client.get_endpoint(model=模型)
            request_body = self.client.build_request_body(
                prompt=提示词,
                model=模型,
                thinking_level=思考等级,
                image_data=image_data,
                video_data=video_data,
                document_data=document_data
            )
            
            print(f"Google Gemini: 发送请求...")
            
            # 调用底层 API 获取原始响应
            async def get_raw_response():
                return await self.client.request_async(
                    endpoint, 
                    request_body, 
                    session=None
                )
            
            # 在独立线程中执行异步请求
            raw_response = self.client.run_async_in_thread(get_raw_response())
            
            # 计算耗时
            elapsed = time.time() - start_time
            
            # 解析响应，分离主要内容和思考内容
            main_text, thought_text = self._parse_dual_output(raw_response)
            
            # 打印响应 token 用量
            usage = raw_response.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", 0)
            candidates_tokens = usage.get("candidatesTokenCount", 0)
            thoughts_tokens = usage.get("thoughtsTokenCount", 0)
            total_tokens = usage.get("totalTokenCount", 0)
            finish_reason = ""
            candidates = raw_response.get("candidates", [])
            if candidates:
                finish_reason = candidates[0].get("finishReason", "")
            
            print(f"Google Gemini: 生成完成 (耗时: {elapsed:.2f}s)")
            print(f"Google Gemini: finishReason = {finish_reason}")
            print(f"Google Gemini: Token 用量 — 输入: {prompt_tokens}, 输出: {candidates_tokens}, 思考: {thoughts_tokens}, 合计: {total_tokens}")
            print(f"Google Gemini: 主要内容长度: {len(main_text)} 字符")
            print(f"Google Gemini: 思考内容长度: {len(thought_text)} 字符")
            
            # 输出预览
            if main_text:
                preview = main_text[:100] + "..." if len(main_text) > 100 else main_text
                print(f"Google Gemini: 主要内容预览: {preview}")
            
            return (main_text, thought_text)
        
        except ValueError as e:
            # 检测是否为授权错误
            if str(e) == "未授权！":
                print("请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            else:
                # 用户输入错误 - 只显示简洁信息
                error_msg = str(e).split('\n')[0]  # 只取第一行
                print(f"Google Gemini: ❌ {error_msg}")
                raise ValueError(error_msg) from None
        
        except RuntimeError as e:
            # API 或网络错误 - 只显示简洁信息
            error_msg = str(e).split('\n')[0]  # 只取第一行
            print(f"Google Gemini: ❌ {error_msg}")
            raise RuntimeError(error_msg) from None
        
        except Exception as e:
            # 其他未知错误 - 只显示简洁信息
            error_msg = str(e).split('\n')[0]
            print(f"Google Gemini: ❌ {error_msg}")
            raise type(e)(error_msg) from None
