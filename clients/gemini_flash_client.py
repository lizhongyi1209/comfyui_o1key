"""
Gemini Flash API 客户端
用于调用 Gemini 3 Flash 模型进行多模态文本生成
"""

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp

from ..utils.config import get_api_key_or_raise, get_api_base_url
from ..models_config import (
    get_flash_model_endpoint,
    get_enabled_flash_models
)
from .base_client import BaseAPIClient


class GeminiFlashClient(BaseAPIClient):
    """
    Gemini Flash API 客户端
    用于调用 Gemini 3 Flash 模型进行多模态文本生成
    
    特点：
    - 支持图片和视频输入
    - 支持动态思考等级端点（不思考/低/中/高）
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化客户端
        
        Args:
            api_key: API 密钥，如果为 None 则从配置文件或环境变量读取
        """
        if api_key is None:
            api_key = get_api_key_or_raise("O1KEY_API_KEY")
        
        super().__init__(
            base_url=get_api_base_url(),
            api_key=api_key,
            max_request_size=20 * 1024 * 1024  # 20MB
        )
    
    def get_endpoint(
        self, 
        model: str = "gemini-3-flash-preview",
        thinking_level: str = "不思考",
        **kwargs
    ) -> str:
        """
        根据模型和思考等级获取 API 端点
        
        Args:
            model: 模型名称
            thinking_level: 思考等级（不思考/低/中/高）
        
        Returns:
            API 端点路径
        """
        endpoint = get_flash_model_endpoint(model, thinking_level)
        
        if endpoint is None:
            # 回退到默认端点
            default_models = get_enabled_flash_models()
            if default_models:
                endpoint = get_flash_model_endpoint(default_models[0], thinking_level)
        
        if endpoint is None:
            raise ValueError(f"无法获取模型 '{model}' (思考等级: {thinking_level}) 的端点")
        
        return endpoint
    
    def build_request_body(
        self,
        prompt: str = "",
        model: str = "gemini-3-flash-preview",
        thinking_level: str = "不思考",
        image_data: Optional[List[Dict[str, str]]] = None,
        video_data: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        构建 API 请求体
        
        Args:
            prompt: 用户提示词
            model: 模型名称
            thinking_level: 思考等级（不思考/低/中/高）- 通过动态端点控制，不需要在请求体中传递
            image_data: 图片数据列表，每个元素包含 mime_type 和 data
            video_data: 视频数据，包含 mime_type 和 data
        
        Returns:
            请求体字典
        """
        parts = []
        
        # 添加文本部分
        if prompt:
            parts.append({"text": prompt})
        
        # 添加图片部分（如果有）
        if image_data:
            for img in image_data:
                parts.append({
                    "inline_data": {
                        "mime_type": img["mime_type"],
                        "data": img["data"]
                    }
                })
        
        # 添加视频部分（如果有）
        if video_data:
            parts.append({
                "inline_data": {
                    "mime_type": video_data["mime_type"],
                    "data": video_data["data"]
                }
            })
        
        # 构建请求体
        request_body = {
            "contents": [
                {
                    "parts": parts
                }
            ]
        }
        
        # 注意：思考等级现在通过动态端点控制，不再需要 thinkingConfig
        
        return request_body
    
    def parse_response(self, response: Dict[str, Any]) -> str:
        """
        解析 API 响应，提取生成的文本
        
        Args:
            response: API 响应字典
        
        Returns:
            生成的文本内容
        
        Raises:
            RuntimeError: 解析失败或 API 拒绝时
        """
        # 检查 candidatesTokenCount
        usage_metadata = response.get("usageMetadata", {})
        candidates_token_count = usage_metadata.get("candidatesTokenCount", -1)
        
        if candidates_token_count == 0:
            raise RuntimeError(
                "内容审核拒绝 - candidatesTokenCount = 0\n\n"
                "原因：提示词或输入内容包含不适当内容\n"
                "建议：检查并调整输入内容"
            )
        
        # 检查 finishReason
        candidates = response.get("candidates", [])
        if candidates:
            for candidate in candidates:
                finish_reason = candidate.get("finishReason", "")
                
                if finish_reason and finish_reason not in ["STOP", "MAX_TOKENS"]:
                    reason_messages = {
                        "PROHIBITED_CONTENT": "违禁内容拒绝",
                        "SAFETY": "安全过滤器拒绝",
                        "RECITATION": "版权问题"
                    }
                    error_title = reason_messages.get(finish_reason, f"生成异常 ({finish_reason})")
                    raise RuntimeError(f"{error_title}\n建议：调整输入内容后重试")
        
        # 提取文本内容
        text_parts = []
        
        for candidate in candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
        
        if not text_parts:
            raise RuntimeError("API 响应中未找到生成的文本")
        
        # 合并所有文本部分
        return "\n".join(text_parts)
    
    async def generate_async(
        self,
        prompt: str,
        model: str = "gemini-3-flash-preview",
        thinking_level: str = "不思考",
        image_data: Optional[List[Dict[str, str]]] = None,
        video_data: Optional[Dict[str, str]] = None,
        session: Optional[aiohttp.ClientSession] = None
    ) -> str:
        """
        异步生成文本
        
        Args:
            prompt: 用户提示词
            model: 模型名称
            thinking_level: 思考等级（不思考/低/中/高）
            image_data: 图片数据列表
            video_data: 视频数据
            session: aiohttp 会话
        
        Returns:
            生成的文本内容
        """
        endpoint = self.get_endpoint(model=model, thinking_level=thinking_level)
        request_body = self.build_request_body(
            prompt=prompt,
            model=model,
            thinking_level=thinking_level,
            image_data=image_data,
            video_data=video_data
        )
        
        response = await self.request_async(
            endpoint, 
            request_body, 
            session
        )
        
        return self.parse_response(response)
    
    def generate_sync(
        self,
        prompt: str,
        model: str = "gemini-3-flash-preview",
        thinking_level: str = "不思考",
        image_data: Optional[List[Dict[str, str]]] = None,
        video_data: Optional[Dict[str, str]] = None
    ) -> str:
        """
        同步生成文本（用于 ComfyUI 节点）
        
        Args:
            prompt: 用户提示词
            model: 模型名称
            thinking_level: 思考等级（不思考/低/中/高）
            image_data: 图片数据列表
            video_data: 视频数据
        
        Returns:
            生成的文本内容
        """
        coro = self.generate_async(
            prompt=prompt,
            model=model,
            thinking_level=thinking_level,
            image_data=image_data,
            video_data=video_data
        )
        
        return self.run_async_in_thread(coro)
