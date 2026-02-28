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
from ..utils.config import get_api_key_or_raise, get_api_base_url
from .base_client import BaseAPIClient


class GeminiAPIClient(BaseAPIClient):
    """
    Gemini API 客户端
    用于调用 Gemini 3 Pro 模型进行图像生成
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
    ) -> tuple[List[Image.Image], Dict[str, Any]]:
        """
        异步解析 API 响应，提取生成的图像
        
        Args:
            response: API 响应字典
            session: aiohttp 会话（用于下载图片）
        
        Returns:
            (图像列表, 格式信息字典)
            格式信息包含: type (base64/url), size, resolution, download_speed (仅URL)
        
        Raises:
            RuntimeError: 解析失败或 API 拒绝时
        """
        
        # 初始化格式信息
        format_info = {
            "type": None,  # "base64" or "url"
            "size": 0,
            "resolution": None,
            "download_speed": None
        }
        
        candidates = response.get("candidates", [])
        
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
            for candidate_idx, candidate in enumerate(candidates):
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                
                for part_idx, part in enumerate(parts):
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
                            
                            # 记录格式信息
                            if format_info["type"] is None:
                                format_info["type"] = "base64"
                                format_info["size"] = len(img_data) * 3 / 4  # Base64 解码后的字节数
                                format_info["resolution"] = f"{img.size[0]}x{img.size[1]}"
                    
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
                            for url_idx, url in enumerate(urls):
                                try:
                                    # 使用 aiohttp 异步下载
                                    download_start = time.time()
                                    async with session.get(url) as img_response:
                                        if img_response.status == 200:
                                            img_data = await img_response.read()
                                            download_time = time.time() - download_start
                                            img_size = len(img_data)
                                            speed = img_size / download_time if download_time > 0 else 0
                                            
                                            img = Image.open(BytesIO(img_data))
                                            images.append(img)
                                            
                                            # 记录格式信息（只记录第一张）
                                            if format_info["type"] is None:
                                                format_info["type"] = "url"
                                                format_info["size"] = img_size
                                                format_info["resolution"] = f"{img.size[0]}x{img.size[1]}"
                                                format_info["download_speed"] = speed
                                except Exception as e:
                                    pass  # 静默失败，继续尝试其他URL
                    
                    # 方式3: 直接的 URL 字段 - 也改为异步
                    elif "imageUrl" in part or "url" in part:
                        url = part.get("imageUrl") or part.get("url")
                        try:
                            download_start = time.time()
                            async with session.get(url) as img_response:
                                if img_response.status == 200:
                                    img_data = await img_response.read()
                                    download_time = time.time() - download_start
                                    img_size = len(img_data)
                                    speed = img_size / download_time if download_time > 0 else 0
                                    
                                    img = Image.open(BytesIO(img_data))
                                    images.append(img)
                                    
                                    # 记录格式信息
                                    if format_info["type"] is None:
                                        format_info["type"] = "url"
                                        format_info["size"] = img_size
                                        format_info["resolution"] = f"{img.size[0]}x{img.size[1]}"
                                        format_info["download_speed"] = speed
                        except Exception as e:
                            pass  # 静默失败
        
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
        
        return images, format_info
    
    async def generate_single_async(
        self,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        images: Optional[List[Image.Image]] = None,
        session=None,
        task_index: Optional[int] = None,
        total_tasks: Optional[int] = None,
        debug: bool = False
    ) -> tuple[List[Image.Image], Dict[str, Any]]:
        """
        单次异步生成请求（极简单行日志）
        
        Args:
            prompt: 提示词
            model: 模型名称
            resolution: 分辨率
            aspect_ratio: 宽高比
            images: 输入图像列表
            session: aiohttp 会话
            task_index: 任务索引（用于批量任务）
            total_tasks: 总任务数（用于批量任务）
        
        Returns:
            (生成的图像列表, 计时信息字典)
        """
        import json
        
        total_start = time.time()
        
        # 任务前缀
        task_prefix = f"[{task_index + 1}/{total_tasks}]" if task_index is not None and total_tasks else ""
        
        # ========== 1. 构建请求 ==========
        build_start = time.time()
        endpoint = self.get_endpoint(model=model, resolution=resolution)
        request_body = self.build_request_body(
            prompt=prompt,
            images=images,
            aspect_ratio=aspect_ratio,
            resolution=resolution
        )
        build_time = time.time() - build_start
        
        # 计算请求体大小
        request_size = len(json.dumps(request_body).encode('utf-8'))
        if request_size < 1024 * 1024:
            size_str = f"{request_size / 1024:.2f}KB"
        else:
            size_str = f"{request_size / (1024 * 1024):.2f}MB"
        
        # ========== 2. 发送网络请求 ==========
        request_start = time.time()
        
        try:
            response = await self.request_async(endpoint, request_body, session)
        except Exception as e:
            request_time = time.time() - request_start
            error_first_line = str(e).split('\n')[0]
            print(f"{task_prefix} 请求 {size_str} → API {request_time:.1f}s → 失败: {error_first_line} ✗")
            raise
        
        request_time = time.time() - request_start
        
        # ========== 调试日志：打印完整响应 ==========
        if debug:
            import json as _json
            # 构建可安全序列化的响应副本（截断 base64 图片数据避免输出过长）
            def _truncate_base64(obj, max_len=200):
                if isinstance(obj, dict):
                    return {k: _truncate_base64(v, max_len) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_truncate_base64(item, max_len) for item in obj]
                elif isinstance(obj, str) and len(obj) > max_len:
                    # 判断是否为 base64 图片数据（不含空格/换行的长字符串）
                    if all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in obj[:50]):
                        return f"<base64 data, {len(obj)} chars>"
                    return obj
                return obj
            
            safe_response = _truncate_base64(response)
            print(
                f"\n{'='*60}\n"
                f"[调试日志] 任务 {task_prefix or '?'} 完整 API 响应：\n"
                f"{_json.dumps(safe_response, ensure_ascii=False, indent=2)}\n"
                f"{'='*60}\n"
            )
        
        # ========== 3. 解析响应 ==========
        parse_start = time.time()
        
        try:
            result_images, format_info = await self.parse_response_async(response, session)
        except Exception as e:
            parse_time = time.time() - parse_start
            error_first_line = str(e).split('\n')[0]
            print(f"{task_prefix} 请求 {size_str} → API {request_time:.1f}s → 解析失败: {error_first_line} ✗")
            raise
        
        parse_time = time.time() - parse_start
        
        # ========== 4. 格式化输出（单行） ==========
        # 格式化图像大小
        img_size = format_info.get("size", 0)
        if img_size < 1024 * 1024:
            img_size_str = f"{img_size / 1024:.2f}KB"
        else:
            img_size_str = f"{img_size / (1024 * 1024):.2f}MB"
        
        # 根据类型构建下载信息
        if format_info.get("type") == "base64":
            download_info = f"Base64 {img_size_str} ({parse_time:.1f}s)"
        elif format_info.get("type") == "url":
            speed = format_info.get("download_speed", 0)
            speed_str = f"{speed / (1024 * 1024):.1f}MB/s"
            download_info = f"URL {img_size_str} ({parse_time:.1f}s, {speed_str})"
        else:
            download_info = f"{img_size_str}"
        
        # 单行输出
        print(f"{task_prefix} 请求 {size_str} → API {request_time:.1f}s → {download_info} ✓")
        
        # 返回结果和计时信息
        total_time = time.time() - total_start
        timing_info = {
            "build_time": build_time,
            "request_time": request_time,
            "parse_time": parse_time,
            "total_time": total_time,
            "format_type": format_info.get("type", "unknown")
        }
        
        return result_images, timing_info
    
    async def generate_batch_async(
        self,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        batch_size: int,
        images: Optional[List[Image.Image]] = None,
        progress_callback: Optional[Callable[[int, int, bool, Optional[str]], None]] = None,
        debug: bool = False
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
        first_error = None  # 保存第一个错误
        
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
                        session=session,
                        task_index=i + 1,
                        total_tasks=batch_size,
                        debug=debug
                    ),
                    name=f"task_{i}"
                )
                tasks.append(task)
            
            # 使用 as_completed 实时获取完成的任务
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                completed += 1
                try:
                    result_images, timing_info = await coro
                    if result_images:
                        all_images.append(result_images[0])
                        success_count += 1
                        if progress_callback:
                            progress_callback(completed, batch_size, True, None)
                except Exception as e:
                    fail_count += 1
                    # 保存第一个错误（用于后续抛出）
                    if first_error is None:
                        first_error = e
                    error_msg = str(e)
                    # 截取错误信息的第一行
                    if '\n' in error_msg:
                        error_msg = error_msg.split('\n')[0]
                    if progress_callback:
                        progress_callback(completed, batch_size, False, error_msg)
        
        if not all_images:
            # 如果有保存的原始错误，直接抛出原始错误
            if first_error:
                raise first_error
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
        progress_callback: Optional[Callable[[int, int], None]] = None,
        debug: bool = False
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
            progress_callback=progress_callback,
            debug=debug
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
        progress_callback: Optional[Callable[[int, int, bool, Optional[str]], None]] = None,
        debug: bool = False
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
        first_error = None  # 保存第一个错误
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
                            session=session,
                            task_index=task_idx + 1,
                            total_tasks=total_tasks,
                            debug=debug
                        ),
                        name=f"task_{task_idx}"
                    )
                    tasks.append(task)
                    task_idx += 1
            
            # 使用 as_completed 实时获取完成的任务
            for coro in asyncio.as_completed(tasks):
                completed += 1
                try:
                    result_images, timing_info = await coro
                    if result_images:
                        all_images.append(result_images[0])
                        success_count += 1
                        if progress_callback:
                            progress_callback(completed, total_tasks, True, None)
                except Exception as e:
                    fail_count += 1
                    # 保存第一个错误（用于后续抛出）
                    if first_error is None:
                        first_error = e
                    error_msg = str(e)
                    # 截取错误信息的第一行
                    if '\n' in error_msg:
                        error_msg = error_msg.split('\n')[0]
                    if progress_callback:
                        progress_callback(completed, total_tasks, False, error_msg)
        
        if not all_images:
            # 如果有保存的原始错误，直接抛出原始错误
            if first_error:
                raise first_error
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
        progress_callback: Optional[Callable[[int, int], None]] = None,
        debug: bool = False
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
            progress_callback=progress_callback,
            debug=debug
        )
        
        return self.run_async_in_thread(coro)
    