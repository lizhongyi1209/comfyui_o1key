"""
API 客户端基类
提供通用的 HTTP 请求、响应解析和错误处理功能
"""

import asyncio
import json
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import os
import time

import aiohttp

from ..utils.http_error import HTTP_ERROR_MESSAGES, RETRYABLE_STATUS_CODES, _compute_delay, DEFAULT_MAX_RETRIES, DEFAULT_BASE_DELAY, DEFAULT_MAX_DELAY, DEFAULT_BACKOFF_FACTOR



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
        max_request_size: int = 100 * 1024 * 1024
    ):
        """
        初始化客户端

        Args:
            base_url: API 基础 URL
            api_key: API 密钥
            max_request_size: 最大请求体大小（字节），默认 100MB
        """
        self.base_url = base_url
        self.api_key = api_key
        self.max_request_size = max_request_size
        self.proxy_url: Optional[str] = None  # 由节点在调用前注入，如 "http://127.0.0.1:7897"
    
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
    
    def _make_session(self) -> aiohttp.ClientSession:
        """
        创建统一的 aiohttp ClientSession，全局禁用 SSL 验证。
        所有需要独立创建 session 的地方都应调用此方法，
        避免因客户端系统缺少根证书导致 SSLCertVerificationError。
        """
        connector = aiohttp.TCPConnector(ssl=False, limit=0, limit_per_host=0)
        return aiohttp.ClientSession(connector=connector, trust_env=False)

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
            raise ValueError(
                "请求体积超过100MB限制，请调整分辨率或减少图片数量"
            )
    
    def get_http_error_message(self, status_code: int, error_message: str) -> Optional[str]:
        """
        子类可重写：为指定 HTTP 状态码返回自定义错误文案。
        若返回 None，则使用基类默认拼接文案。
        
        Args:
            status_code: HTTP 状态码（如 429、503）
            error_message: API 返回的原始错误信息
        
        Returns:
            自定义完整错误文案，或 None 表示使用默认
        """
        return None
    
    async def request_async(
        self,
        endpoint: str,
        request_body: Dict[str, Any],
        session: Optional[aiohttp.ClientSession] = None,
        use_bearer_token: bool = False,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        发送异步 HTTP 请求（带详细计时）

        Args:
            endpoint: API 端点
            request_body: 请求体
            session: aiohttp 会话（可选）
            use_bearer_token: 是否使用 Bearer Token 认证
            timeout: 超时时间（秒），默认 900 秒

        Returns:
            响应 JSON

        Raises:
            RuntimeError: 请求失败时
            InterruptProcessingException: 用户点击终止按钮时
        """
        import time

        # 尝试导入 ComfyUI 中断机制
        try:
            from comfy.model_management import processing_interrupted, InterruptProcessingException
            _interrupt_available = True
        except ImportError:
            _interrupt_available = False

        url = f"{self.base_url}{endpoint}"
        headers = self.get_headers(use_bearer_token)

        # 检查请求大小
        self.check_request_size(request_body)

        close_session = False
        if session is None:
            session = self._make_session()
            close_session = True

        # 设置请求超时：连接超时 30s，读取超时 900s（防止服务器出图后卡住）
        _timeout_seconds = timeout if timeout is not None else 900
        _aiohttp_timeout = aiohttp.ClientTimeout(
            total=_timeout_seconds,
            connect=30,
            sock_read=_timeout_seconds
        )

        async def _do_request():
            connect_start = time.time()
            async with session.post(url, json=request_body, headers=headers, timeout=_aiohttp_timeout, proxy=self.proxy_url) as response:
                connect_time = time.time() - connect_start

                if response.status != 200:
                    error_text = await response.text()
                    # 返回状态码和错误文本，由外层处理重试
                    return {"_error": True, "_status": response.status, "_text": error_text}

                wait_start = time.time()
                response_data = await response.json()
                download_time = time.time() - wait_start

                response_size = len(str(response_data))
                if not isinstance(response_data, dict):
                    response_data = {"data": response_data}

                response_data["_timing"] = {
                    "connect_time": connect_time,
                    "download_time": download_time,
                    "response_size": response_size
                }
                return response_data

        async def _poll_interrupt():
            """每 0.5s 轮询一次中断标志"""
            while True:
                await asyncio.sleep(0.5)
                if processing_interrupted():
                    return

        try:
            last_error_status = None
            last_error_text = ""

            for attempt in range(DEFAULT_MAX_RETRIES + 1):
                if _interrupt_available:
                    request_task = asyncio.ensure_future(_do_request())
                    interrupt_task = asyncio.ensure_future(_poll_interrupt())

                    done, pending = await asyncio.wait(
                        [request_task, interrupt_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    for t in pending:
                        t.cancel()
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass

                    if interrupt_task in done and request_task not in done:
                        raise InterruptProcessingException()

                    result = request_task.result()
                else:
                    result = await _do_request()

                if isinstance(result, dict) and result.get("_error"):
                    status = result["_status"]
                    error_text = result["_text"]
                    last_error_status = status
                    last_error_text = error_text

                    if status in RETRYABLE_STATUS_CODES and attempt < DEFAULT_MAX_RETRIES:
                        friendly = HTTP_ERROR_MESSAGES.get(status, f"请求失败 ({status})")
                        delay = _compute_delay(attempt, DEFAULT_BASE_DELAY, DEFAULT_MAX_DELAY, DEFAULT_BACKOFF_FACTOR)
                        print(f"{friendly} {delay:.1f}s 后重试 ({attempt+1}/{DEFAULT_MAX_RETRIES})...")
                        await asyncio.sleep(delay)
                        continue

                    if status in HTTP_ERROR_MESSAGES:
                        raise RuntimeError(HTTP_ERROR_MESSAGES[status])
                    raise RuntimeError(error_text)

                return result

            if last_error_status and last_error_status in HTTP_ERROR_MESSAGES:
                raise RuntimeError(HTTP_ERROR_MESSAGES[last_error_status])
            raise RuntimeError(last_error_text)

        except InterruptProcessingException:
            raise

        except aiohttp.ServerTimeoutError as e:
            raise RuntimeError(
                f"请求超时！等待服务器响应超过 {_timeout_seconds} 秒。\n"
                f"服务器可能仍在生成图片，请稍后重试，或检查网络连接。"
            ) from e

        except aiohttp.ClientConnectorError as e:
            raise RuntimeError(
                f"无法连接到服务器：{str(e)}\n"
                f"请检查网络连接是否正常。"
            ) from e

        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"请求超时！等待服务器响应超过 {_timeout_seconds} 秒。\n"
                f"服务器可能仍在生成图片，请稍后重试，或检查网络连接。"
            ) from e

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
            timeout: 超时时间（秒）- 已废弃，由服务器端控制
        
        Returns:
            响应 JSON
        
        Raises:
            RuntimeError: 请求失败时
        """
        url = f"{self.base_url}{endpoint}"
        headers = self.get_headers(use_bearer_token)
        
        close_session = False
        if session is None:
            session = self._make_session()
            close_session = True

        try:
            _get_start = time.time()
            async with session.get(url, headers=headers) as response:
                _get_elapsed = time.time() - _get_start
                if response.status != 200:
                    error_text = await response.text()
                    
                    # 尝试解析 JSON 错误信息，提取关键内容
                    error_message = error_text
                    try:
                        error_json = json.loads(error_text)
                        # 尝试从多个常见位置提取错误信息
                        if "error" in error_json:
                            if isinstance(error_json["error"], dict):
                                error_message = error_json["error"].get("message", error_text)
                            else:
                                error_message = str(error_json["error"])
                        elif "message" in error_json:
                            error_message = error_json["message"]
                    except:
                        # 如果不是 JSON，使用原始文本
                        pass
                    
                    # 针对常见错误状态码提供友好提示
                    if response.status == 400:
                        raise RuntimeError(
                            f"请求参数错误 (400 Bad Request)\n"
                            f"API 返回错误：{error_message}\n"
                            f"建议：检查请求参数"
                        )
                    elif response.status == 401:
                        raise RuntimeError(
                            f"认证失败 (401 Unauthorized)\n"
                            f"API 返回错误：{error_message}\n"
                            f"建议：检查 API 密钥"
                        )
                    elif response.status == 429:
                        custom = self.get_http_error_message(429, error_message)
                        if custom is not None:
                            raise RuntimeError(custom)
                        raise RuntimeError(HTTP_ERROR_MESSAGES[429])
                    elif response.status == 503:
                        custom = self.get_http_error_message(503, error_message)
                        if custom is not None:
                            raise RuntimeError(custom)
                        raise RuntimeError(HTTP_ERROR_MESSAGES[503])
                    elif response.status == 504:
                        raise RuntimeError(HTTP_ERROR_MESSAGES[504])
                    elif response.status == 502:
                        raise RuntimeError(HTTP_ERROR_MESSAGES[502])
                    else:
                        raise RuntimeError(
                            f"API 请求失败 (状态码: {response.status})\n"
                            f"API 返回错误：{error_message}"
                        )
                
                _resp_data = await response.json()
                return _resp_data
        
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
        async with self._make_session() as session:
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
    
    async def query_balance_async(self) -> Dict[str, Any]:
        """
        异步查询账户余额
        
        Returns:
            余额信息字典，包含 name、total_available 等字段
        
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
        同步查询账户余额（用于 ComfyUI 节点）
        
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
            格式化文本，如 "当前余额：100.00 | API：xxx"
        
        Example:
            >>> data = {"name": "test-api", "total_available": 50000000}
            >>> client.format_balance_info(data)
            '当前余额：100.00 | API：test-api'
        """
        api_name = balance_data.get("name", "未知")
        total_available = balance_data.get("total_available", 0)
        
        # 实际显示余额 = total_available / 500000，单位：美元
        balance_in_dollars = total_available / 500000
        
        return f"当前余额：{balance_in_dollars:.2f} | API：{api_name}"
