"""
API 客户端基类
提供通用的 HTTP 请求、响应解析和错误处理功能
"""

import asyncio
import json
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import aiohttp


class BaseAPIClient(ABC):
    """
    API 客户端抽象基类
    
    子类需要实现以下方法：
    - get_endpoint(): 获取 API 端点
    - build_request_body(): 构建请求体
    - parse_response(): 解析响应
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        max_request_size: int = 20 * 1024 * 1024
    ):
        """
        初始化客户端
        
        Args:
            base_url: API 基础 URL
            api_key: API 密钥
            max_request_size: 最大请求体大小（字节），默认 20MB
        """
        self.base_url = base_url
        self.api_key = api_key
        self.max_request_size = max_request_size
    
    @abstractmethod
    def get_endpoint(self, **kwargs) -> str:
        """
        获取 API 端点路径
        
        Args:
            **kwargs: 额外参数（如模型名、分辨率等）
        
        Returns:
            端点路径字符串
        """
        pass
    
    @abstractmethod
    def build_request_body(self, **kwargs) -> Dict[str, Any]:
        """
        构建 API 请求体
        
        Args:
            **kwargs: 请求参数
        
        Returns:
            请求体字典
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: Dict[str, Any]) -> Any:
        """
        解析 API 响应
        
        Args:
            response: API 响应字典
        
        Returns:
            解析后的结果
        """
        pass
    
    def get_headers(self, use_bearer_token: bool = False) -> Dict[str, str]:
        """
        获取请求头
        
        Args:
            use_bearer_token: 是否使用 Bearer Token 认证（默认为 False）
        
        Returns:
            请求头字典
        """
        if use_bearer_token:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        else:
            return {
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json"
            }
    
    def check_request_size(self, request_body: Dict[str, Any]) -> None:
        """
        检查请求体大小是否超过限制
        
        Args:
            request_body: 请求体字典
        
        Raises:
            ValueError: 如果请求体超过限制
        """
        request_json = json.dumps(request_body)
        request_size = len(request_json.encode('utf-8'))
        
        if request_size > self.max_request_size:
            size_mb = request_size / 1024 / 1024
            limit_mb = self.max_request_size / 1024 / 1024
            raise ValueError(
                f"请求体大小 {size_mb:.2f}MB 超过限制 {limit_mb:.0f}MB，"
                "请降低分辨率或减少图片数量"
            )
    
    async def request_async(
        self,
        endpoint: str,
        request_body: Dict[str, Any],
        session: Optional[aiohttp.ClientSession] = None,
        use_bearer_token: bool = False,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        发送异步 HTTP 请求
        
        Args:
            endpoint: API 端点
            request_body: 请求体
            session: aiohttp 会话（可选）
            use_bearer_token: 是否使用 Bearer Token 认证
        
        Returns:
            响应 JSON
        
        Raises:
            RuntimeError: 请求失败时
        """
        url = f"{self.base_url}{endpoint}"
        headers = self.get_headers(use_bearer_token)
        
        # 检查请求大小
        self.check_request_size(request_body)
        
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        
        try:
            # 设置超时
            timeout_obj = aiohttp.ClientTimeout(total=timeout) if timeout else None
            async with session.post(url, json=request_body, headers=headers, timeout=timeout_obj) as response:
                if response.status != 200:
                    error_text = await response.text()
                    
                    # 针对常见错误状态码提供友好提示
                    if response.status == 504:
                        raise RuntimeError(
                            f"API 请求超时 (504 Gateway Timeout)\n"
                            f"原因：服务器响应超时或该端点暂时不可用\n"
                            f"建议：\n"
                            f"  - 尝试使用其他模型\n"
                            f"  - 稍后重试\n"
                            f"  - 降低分辨率或减少输入图像数量\n"
                            f"详细错误: {error_text[:200]}"
                        )
                    elif response.status == 503:
                        raise RuntimeError(
                            f"服务暂时不可用 (503 Service Unavailable)\n"
                            f"原因：模型服务过载或维护中\n"
                            f"建议：\n"
                            f"  - 稍后重试\n"
                            f"  - 尝试使用其他模型"
                        )
                    elif response.status == 429:
                        raise RuntimeError(
                            f"请求频率超限 (429 Too Many Requests)\n"
                            f"原因：API 配额用尽或请求过于频繁\n"
                            f"建议：\n"
                            f"  - 等待一段时间后重试\n"
                            f"  - 检查 API 配额是否充足"
                        )
                    elif response.status == 404:
                        raise RuntimeError(
                            f"端点不存在 (404 Not Found)\n"
                            f"原因：API 端点路径错误或模型不存在\n"
                            f"建议：\n"
                            f"  - 检查模型名称是否正确\n"
                            f"  - 使用其他可用模型"
                        )
                    else:
                        raise RuntimeError(
                            f"API 请求失败 (状态码: {response.status}): {error_text}"
                        )
                
                return await response.json()
        
        finally:
            if close_session:
                await session.close()
    
    async def request_get_async(
        self,
        endpoint: str,
        session: Optional[aiohttp.ClientSession] = None,
        use_bearer_token: bool = True,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        发送异步 HTTP GET 请求
        
        Args:
            endpoint: API 端点
            session: aiohttp 会话（可选）
            use_bearer_token: 是否使用 Bearer Token 认证（默认为 True）
        
        Returns:
            响应 JSON
        
        Raises:
            RuntimeError: 请求失败时
        """
        url = f"{self.base_url}{endpoint}"
        headers = self.get_headers(use_bearer_token)
        
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        
        try:
            # 设置超时
            timeout_obj = aiohttp.ClientTimeout(total=timeout) if timeout else None
            async with session.get(url, headers=headers, timeout=timeout_obj) as response:
                if response.status != 200:
                    error_text = await response.text()
                    
                    # 针对常见错误状态码提供友好提示
                    if response.status == 504:
                        raise RuntimeError(
                            f"API 请求超时 (504 Gateway Timeout)\n"
                            f"原因：服务器响应超时或该端点暂时不可用\n"
                            f"建议：稍后重试"
                        )
                    elif response.status == 503:
                        raise RuntimeError(
                            f"服务暂时不可用 (503 Service Unavailable)\n"
                            f"原因：服务过载或维护中\n"
                            f"建议：稍后重试"
                        )
                    elif response.status == 429:
                        raise RuntimeError(
                            f"请求频率超限 (429 Too Many Requests)\n"
                            f"原因：API 配额用尽或请求过于频繁\n"
                            f"建议：等待一段时间后重试"
                        )
                    else:
                        raise RuntimeError(
                            f"API 请求失败 (状态码: {response.status}): {error_text}"
                        )
                
                return await response.json()
        
        finally:
            if close_session:
                await session.close()
    
    async def batch_request_async(
        self,
        requests: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """
        批量并发请求
        
        Args:
            requests: 请求列表，每个元素包含 endpoint 和 request_body
            progress_callback: 进度回调函数 (current, total)
        
        Returns:
            响应结果列表
        """
        results = []
        completed = 0
        total = len(requests)
        
        # 创建无限制的连接器
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            
            for req in requests:
                task = self.request_async(
                    endpoint=req['endpoint'],
                    request_body=req['request_body'],
                    session=session
                )
                tasks.append(task)
            
            # 并发执行
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, resp in enumerate(responses):
                if isinstance(resp, Exception):
                    print(f"⚠️ 第 {i+1} 个请求失败: {str(resp)}")
                    continue
                
                try:
                    parsed = self.parse_response(resp)
                    results.append(parsed)
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, total)
                
                except Exception as e:
                    print(f"⚠️ 第 {i+1} 个响应解析失败: {str(e)}")
        
        return results
    
    def run_async_in_thread(self, coro) -> Any:
        """
        在独立线程中运行异步代码（用于 ComfyUI 同步接口）
        
        Args:
            coro: 协程对象
        
        Returns:
            协程执行结果
        """
        result_container = []
        error_container = []
        
        def run_in_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(coro)
                    result_container.append(result)
                finally:
                    loop.close()
            
            except Exception as e:
                error_container.append(e)
        
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        
        if error_container:
            raise error_container[0]
        
        if not result_container:
            raise RuntimeError("异步任务未返回结果")
        
        return result_container[0]
