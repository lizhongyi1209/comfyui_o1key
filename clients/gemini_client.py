"""
Gemini API 客户端
处理与 api.o1key.com 的通信，用于图像生成
"""

import re
import time
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from PIL import Image

from ..utils.image_utils import encode_image_to_base64, decode_base64_to_pil
from ..utils.config import get_api_key_or_raise
from .base_client import BaseAPIClient


# API 基础配置
API_BASE_URL = "https://api.o1key.com"


class GeminiAPIClient(BaseAPIClient):
    """
    Gemini API 客户端
    用于调用 Gemini 3 Pro 模型进行图像生成
    """
    
    @staticmethod
    def get_timeout_by_resolution(resolution: str) -> int:
        """
        根据分辨率获取超时时间
        
        Args:
            resolution: 分辨率（1K, 2K, 4K）
        
        Returns:
            超时时间（秒）
        """
        timeout_map = {
            "1K": 180,  # 3 分钟
            "2K": 300,  # 5 分钟
            "4K": 360   # 6 分钟
        }
        return timeout_map.get(resolution, 300)  # 默认 5 分钟
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化客户端
        
        Args:
            api_key: API 密钥，如果为 None 则从配置文件或环境变量读取
        """
        if api_key is None:
            api_key = get_api_key_or_raise("O1KEY_API_KEY")
        
        super().__init__(
            base_url=API_BASE_URL,
            api_key=api_key,
            max_request_size=20 * 1024 * 1024
        )
    
    def get_endpoint(self, model: str = "", resolution: str = "2K", **kwargs) -> str:
        """
        根据模型和分辨率获取 API 端点
        
        Args:
            model: 模型名称
            resolution: 分辨率（1K, 2K, 4K）
        
        Returns:
            API 端点路径
        """
        from ..models_config import get_model_endpoint
        
        # 特殊处理：动态端点模型（根据分辨率选择）
        if model == "nano-banana-pro":
            if resolution == "1K":
                return "/v1beta/models/nano-banana-pro:generateContent"
            elif resolution == "2K":
                return "/v1beta/models/nano-banana-pro-2k:generateContent"
            elif resolution == "4K":
                return "/v1beta/models/nano-banana-pro-4k:generateContent"
            else:
                return "/v1beta/models/nano-banana-pro-2k:generateContent"
        
        elif model == "gemini-3-pro-image-preview-url":
            if resolution == "1K":
                return "/v1beta/models/gemini-3-pro-image-preview-url:generateContent"
            elif resolution == "2K":
                return "/v1beta/models/gemini-3-pro-image-preview-2k-url:generateContent"
            elif resolution == "4K":
                return "/v1beta/models/gemini-3-pro-image-preview-4k-url:generateContent"
            else:
                return "/v1beta/models/gemini-3-pro-image-preview-2k-url:generateContent"
        
        # 其他模型：从配置文件读取端点
        endpoint = get_model_endpoint(model)
        if endpoint:
            return endpoint
        
        # 兜底：使用标准模式端点
        return "/v1beta/models/gemini-3-pro-image-preview:generateContent"
    
    def build_request_body(
        self,
        prompt: str = "",
        images: Optional[List[Image.Image]] = None,
        aspect_ratio: str = "1:1",
        resolution: str = "2K",
        **kwargs
    ) -> Dict[str, Any]:
        """
        构建 API 请求体
        
        Args:
            prompt: 提示词
            images: 输入图像列表（可选）
            aspect_ratio: 宽高比
            resolution: 分辨率
        
        Returns:
            请求体字典
        """
        parts = []
        
        # 添加文本部分
        parts.append({"text": prompt})
        
        # 添加图像部分（如果有）
        if images:
            for img in images:
                img_base64 = encode_image_to_base64(img)
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_base64
                    }
                })
        
        # 构建请求体
        request_body = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts
                }
            ],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": resolution
                }
            }
        }
        
        return request_body
    
    def parse_response(self, response: Dict[str, Any]) -> List[Image.Image]:
        """
        同步解析 API 响应（保留以满足抽象基类要求）
        
        注意：此方法仅用于兼容基类接口，实际使用请调用 parse_response_async()
        
        Args:
            response: API 响应字典
        
        Returns:
            图像列表
        
        Raises:
            RuntimeError: 此方法不应被直接调用
        """
        raise RuntimeError(
            "parse_response() 不应被直接调用。"
            "请使用 generate_single_async() 或 generate_batch_async() 等高级方法。"
        )
    
    async def parse_response_async(
        self, 
        response: Dict[str, Any],
        session: Optional[aiohttp.ClientSession] = None
    ) -> List[Image.Image]:
        """
        异步解析 API 响应，提取生成的图像
        
        Args:
            response: API 响应字典
            session: aiohttp 会话（用于下载图片）
        
        Returns:
            图像列表
        
        Raises:
            RuntimeError: 解析失败或 API 拒绝时
        """
        
        # ========== 错误检测（按优先级顺序）==========
        
        # 1. 检查 candidatesTokenCount（最高优先级）
        usage_metadata = response.get("usageMetadata", {})
        candidates_token_count = usage_metadata.get("candidatesTokenCount", -1)
        
        if candidates_token_count == 0:
            error_msg = (
                "内容审核拒绝 - candidatesTokenCount = 0\n\n"
                "原因：提示词或参考图包含不适当内容（色情、暴力、敏感话题等），"
                "在内容审核阶段就被拒绝，连候选内容都未生成。\n\n"
                "建议：\n"
                "  - 检查提示词，确保不包含敏感或违规内容\n"
                "  - 如使用参考图，确保图片内容健康合规\n"
                "  - 避免描述暴力、色情等不当内容\n"
                "  - 调整提示词后重试"
            )
            raise RuntimeError(error_msg)
        
        # 2. 检查 finishReason（次优先级）
        candidates = response.get("candidates", [])
        if candidates:
            for candidate in candidates:
                finish_reason = candidate.get("finishReason", "")
                
                if finish_reason and finish_reason != "STOP":
                    # 根据不同的 finishReason 提供具体建议
                    reason_messages = {
                        "PROHIBITED_CONTENT": (
                            "违禁内容拒绝",
                            "生成内容触发了违禁内容策略",
                            [
                                "避免引用未来未发布的产品或概念（知识库截止2025年1月）",
                                "使用专业图片编辑软件处理特殊需求",
                                "确保请求内容在模型知识范围内"
                            ]
                        ),
                        "SAFETY": (
                            "安全过滤器拒绝",
                            "内容触发了安全过滤器",
                            [
                                "使用健康、正面的描述",
                                "避免涉及隐私和伦理问题的内容",
                                "调整提示词后重试"
                            ]
                        ),
                        "RECITATION": (
                            "版权问题",
                            "可能涉及版权或重复已有内容",
                            [
                                "避免涉及版权敏感话题",
                                "使用更原创的描述方式",
                                "调整提示词后重试"
                            ]
                        ),
                        "MAX_TOKENS": (
                            "Token 超限",
                            "生成的内容超过了 Token 限制",
                            [
                                "简化提示词",
                                "减少输入图片数量",
                                "降低请求复杂度"
                            ]
                        )
                    }
                    
                    if finish_reason in reason_messages:
                        title, reason, suggestions = reason_messages[finish_reason]
                        suggestions_text = "\n".join([f"  - {s}" for s in suggestions])
                        error_msg = (
                            f"{title} - finishReason = {finish_reason}\n\n"
                            f"原因：{reason}\n\n"
                            f"建议：\n{suggestions_text}"
                        )
                    else:
                        # 未知的 finishReason
                        error_msg = (
                            f"生成异常 - finishReason = {finish_reason}\n\n"
                            "原因：生成过程中断，具体原因未知\n\n"
                            "建议：\n"
                            "  - 使用健康、正面的描述\n"
                            "  - 避免敏感话题\n"
                            "  - 调整提示词后重试"
                        )
                    
                    raise RuntimeError(error_msg)
        
        # ========== 图像提取 ==========
        
        images = []
        text_responses = []  # 收集文本响应
        
        # 需要关闭 session 的标记
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        
        try:
            for candidate in candidates:
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                
                for part in parts:
                    # 方式1: inline_data 或 inlineData (base64)
                    # 兼容两种命名方式：蛇形（inline_data）和驼峰（inlineData）
                    inline_data_key = None
                    if "inline_data" in part:
                        inline_data_key = "inline_data"
                    elif "inlineData" in part:
                        inline_data_key = "inlineData"
                    
                    if inline_data_key:
                        inline_data = part[inline_data_key]
                        # 同样兼容 data/mimeType 的命名
                        img_data = inline_data.get("data") or inline_data.get("data", "")
                        
                        if img_data:
                            img = decode_base64_to_pil(img_data)
                            images.append(img)
                    
                    # 方式2: text 中的 URL - 改为异步下载
                    elif "text" in part:
                        text = part["text"]
                        
                        # 收集文本响应（用于后续错误检测）
                        text_responses.append(text)
                        
                        # 尝试 markdown 格式: ![alt](url)
                        url_pattern_md = r'!\[.*?\]\((https?://[^\)]+)\)'
                        urls = re.findall(url_pattern_md, text)
                        
                        # 如果没找到，尝试纯 URL 格式
                        if not urls:
                            url_pattern_plain = r'https?://[^\s<>"{}|\\^`\[\]]+'
                            urls = re.findall(url_pattern_plain, text)
                        
                        if urls:
                            for url in urls:
                                try:
                                    # 使用 aiohttp 异步下载，支持更大的超时
                                    download_start = time.time()
                                    timeout = aiohttp.ClientTimeout(total=120)
                                    async with session.get(url, timeout=timeout) as img_response:
                                        if img_response.status == 200:
                                            img_data = await img_response.read()
                                            download_time = time.time() - download_start
                                            img_size_mb = len(img_data) / 1024 / 1024
                                            speed_mbps = img_size_mb / download_time if download_time > 0 else 0
                                            # print(f"🔽 图片下载: {img_size_mb:.2f}MB 耗时 {download_time:.2f}s 速度 {speed_mbps:.2f}MB/s")
                                            img = Image.open(BytesIO(img_data))
                                            images.append(img)
                                        else:
                                            print(f"Nano Banana Pro: 下载图片失败 - HTTP {img_response.status}")
                                except Exception as e:
                                    print(f"Nano Banana Pro: 下载图片失败 - {str(e)}")
                    
                    # 方式3: 直接的 URL 字段 - 也改为异步
                    elif "imageUrl" in part or "url" in part:
                        url = part.get("imageUrl") or part.get("url")
                        try:
                            download_start = time.time()
                            timeout = aiohttp.ClientTimeout(total=120)
                            async with session.get(url, timeout=timeout) as img_response:
                                if img_response.status == 200:
                                    img_data = await img_response.read()
                                    download_time = time.time() - download_start
                                    img_size_mb = len(img_data) / 1024 / 1024
                                    speed_mbps = img_size_mb / download_time if download_time > 0 else 0
                                    # print(f"🔽 图片下载: {img_size_mb:.2f}MB 耗时 {download_time:.2f}s 速度 {speed_mbps:.2f}MB/s")
                                    img = Image.open(BytesIO(img_data))
                                    images.append(img)
                                else:
                                    print(f"Nano Banana Pro: 下载图片失败 - HTTP {img_response.status}")
                        except Exception as e:
                            print(f"Nano Banana Pro: 下载图片失败 - {str(e)}")
        
        except Exception as e:
            raise RuntimeError(f"解析 API 响应失败: {str(e)}")
        
        finally:
            if close_session:
                await session.close()
        
        # 3. 检查 API 文本响应拒绝说明
        if not images and text_responses:
            # API 返回了文本但没有图片，说明请求被拒绝
            combined_text = "\n".join(text_responses)
            error_msg = (
                f"API 拒绝响应\n\n"
                f"API 返回说明：\n{combined_text}\n\n"
                f"建议：\n"
                f"  - 根据上述说明调整请求内容\n"
                f"  - 确保提示词和参考图符合使用规范"
            )
            raise RuntimeError(error_msg)
        
        if not images:
            raise RuntimeError("API 响应中未找到生成的图像")
        
        return images
    
    async def generate_single_async(
        self,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        images: Optional[List[Image.Image]] = None,
        session=None
    ) -> List[Image.Image]:
        """
        单次异步生成请求
        
        Args:
            prompt: 提示词
            model: 模型名称
            resolution: 分辨率
            aspect_ratio: 宽高比
            images: 输入图像列表
            session: aiohttp 会话
        
        Returns:
            生成的图像列表
        """
        endpoint = self.get_endpoint(model=model, resolution=resolution)
        request_body = self.build_request_body(
            prompt=prompt,
            images=images,
            aspect_ratio=aspect_ratio,
            resolution=resolution
        )
        
        # 根据分辨率获取超时时间
        timeout = self.get_timeout_by_resolution(resolution)
        
        response = await self.request_async(endpoint, request_body, session, timeout=timeout)
        # 使用异步解析方法，传入 session 以实现并发图片下载
        return await self.parse_response_async(response, session)
    
    async def generate_batch_async(
        self,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        batch_size: int,
        images: Optional[List[Image.Image]] = None,
        progress_callback: Optional[Callable[[int, int, bool, Optional[str]], None]] = None
    ) -> List[Image.Image]:
        """
        批量全并发生成
        
        Args:
            prompt: 提示词
            model: 模型名称
            resolution: 分辨率
            aspect_ratio: 宽高比
            batch_size: 批次大小
            images: 输入图像列表
            progress_callback: 进度回调，签名为 (completed, total, success, error_msg)
        
        Returns:
            生成的图像列表
        """
        import aiohttp
        import asyncio
        
        all_images = []
        completed = 0
        success_count = 0
        fail_count = 0
        
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            
            for i in range(batch_size):
                task = asyncio.create_task(
                    self.generate_single_async(
                        prompt=prompt,
                        model=model,
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                        images=images,
                        session=session
                    ),
                    name=f"task_{i}"
                )
                tasks.append(task)
            
            # 使用 as_completed 实时获取完成的任务
            for coro in asyncio.as_completed(tasks):
                completed += 1
                try:
                    result = await coro
                    if result:
                        all_images.append(result[0])
                        success_count += 1
                        if progress_callback:
                            progress_callback(completed, batch_size, True, None)
                except Exception as e:
                    fail_count += 1
                    error_msg = str(e)
                    # 截取错误信息的第一行
                    if '\n' in error_msg:
                        error_msg = error_msg.split('\n')[0]
                    if progress_callback:
                        progress_callback(completed, batch_size, False, error_msg)
        
        if not all_images:
            raise RuntimeError(f"批量生成失败，{fail_count} 个请求全部失败")
        
        return all_images
    
    def generate_sync(
        self,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        batch_size: int,
        images: Optional[List[Image.Image]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Image.Image]:
        """
        同步生成接口（用于 ComfyUI）
        
        Args:
            prompt: 提示词
            model: 模型名称
            resolution: 分辨率
            aspect_ratio: 宽高比
            batch_size: 批次大小
            images: 输入图像列表
            progress_callback: 进度回调
        
        Returns:
            生成的图像列表
        """
        coro = self.generate_batch_async(
            prompt=prompt,
            model=model,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            batch_size=batch_size,
            images=images,
            progress_callback=progress_callback
        )
        
        return self.run_async_in_thread(coro)
    
    async def generate_multi_prompts_async(
        self,
        prompts: List[str],
        model: str,
        resolution: str,
        aspect_ratio: str,
        images_per_prompt: int,
        images: Optional[List[Image.Image]] = None,
        progress_callback: Optional[Callable[[int, int, bool, Optional[str]], None]] = None
    ) -> List[Image.Image]:
        """
        多提示词批量生成
        
        为每个提示词生成指定数量的图像，所有请求并发执行。
        
        Args:
            prompts: 提示词列表
            model: 模型名称
            resolution: 分辨率
            aspect_ratio: 宽高比
            images_per_prompt: 每个提示词生成的图像数量
            images: 输入图像列表（所有提示词共享）
            progress_callback: 进度回调，签名为 (completed, total, success, error_msg)
        
        Returns:
            生成的图像列表（长度 = len(prompts) * images_per_prompt）
        """
        import aiohttp
        import asyncio
        
        all_images = []
        completed = 0
        success_count = 0
        fail_count = 0
        total_tasks = len(prompts) * images_per_prompt
        
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            
            # 为每个提示词创建 images_per_prompt 个任务
            task_idx = 0
            for prompt in prompts:
                for _ in range(images_per_prompt):
                    task = asyncio.create_task(
                        self.generate_single_async(
                            prompt=prompt,
                            model=model,
                            resolution=resolution,
                            aspect_ratio=aspect_ratio,
                            images=images,
                            session=session
                        ),
                        name=f"task_{task_idx}"
                    )
                    tasks.append(task)
                    task_idx += 1
            
            # 使用 as_completed 实时获取完成的任务
            for coro in asyncio.as_completed(tasks):
                completed += 1
                try:
                    result = await coro
                    if result:
                        all_images.append(result[0])
                        success_count += 1
                        if progress_callback:
                            progress_callback(completed, total_tasks, True, None)
                except Exception as e:
                    fail_count += 1
                    error_msg = str(e)
                    # 截取错误信息的第一行
                    if '\n' in error_msg:
                        error_msg = error_msg.split('\n')[0]
                    if progress_callback:
                        progress_callback(completed, total_tasks, False, error_msg)
        
        if not all_images:
            raise RuntimeError(f"批量生成失败，{fail_count} 个请求全部失败")
        
        return all_images
    
    def generate_multi_prompts_sync(
        self,
        prompts: List[str],
        model: str,
        resolution: str,
        aspect_ratio: str,
        images_per_prompt: int,
        images: Optional[List[Image.Image]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Image.Image]:
        """
        多提示词批量生成（同步接口，用于 ComfyUI）
        
        Args:
            prompts: 提示词列表
            model: 模型名称
            resolution: 分辨率
            aspect_ratio: 宽高比
            images_per_prompt: 每个提示词生成的图像数量
            images: 输入图像列表
            progress_callback: 进度回调
        
        Returns:
            生成的图像列表
        """
        coro = self.generate_multi_prompts_async(
            prompts=prompts,
            model=model,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            images_per_prompt=images_per_prompt,
            images=images,
            progress_callback=progress_callback
        )
        
        return self.run_async_in_thread(coro)
    
    async def query_balance_async(self) -> Dict[str, Any]:
        """
        异步查询余额信息
        
        Returns:
            余额信息字典，包含：
            - name: API 名称
            - total_available: 可用余额（原始值）
            - 其他字段...
        
        Raises:
            RuntimeError: 查询失败时
        """
        endpoint = "/api/usage/token"
        response = await self.request_get_async(endpoint, use_bearer_token=True)
        
        if not response.get("code"):
            raise RuntimeError("余额查询响应格式错误")
        
        data = response.get("data", {})
        return data
    
    def query_balance_sync(self) -> Dict[str, Any]:
        """
        同步查询余额信息（用于 ComfyUI 节点）
        
        Returns:
            余额信息字典
        
        Raises:
            RuntimeError: 查询失败时
        """
        coro = self.query_balance_async()
        return self.run_async_in_thread(coro)
    
    def format_balance_info(self, balance_data: Dict[str, Any]) -> str:
        """
        格式化余额信息为展示文本
        
        Args:
            balance_data: 余额信息字典
        
        Returns:
            格式化的文本，格式为 "当前余额：$XX.XX | API：xxx"
        
        Example:
            >>> data = {"name": "test-api", "total_available": 50000000}
            >>> client.format_balance_info(data)
            '当前余额：$100.00 | API：test-api'
        """
        api_name = balance_data.get("name", "未知")
        total_available = balance_data.get("total_available", 0)
        
        # 转换公式：实际显示 = total_available / 500000
        balance_in_dollars = total_available / 500000
        
        return f"当前余额：${balance_in_dollars:.2f} | API：{api_name}"