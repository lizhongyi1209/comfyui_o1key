"""
异步生图 Provider 抽象基类
定义异步提交+轮询模式的统一接口，支持多种生图模型后端

每个 Provider 封装一种 API 后端的通信协议：
  - 如何提交任务（端点、请求体格式）
  - 如何轮询状态（端点、状态字段语义）
  - 如何解析结果（响应格式、图片提取方式）

新增第三方生图模型时，只需实现此接口即可接入异步节点。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from PIL import Image


class BaseAsyncImageProvider(ABC):
    """异步生图 Provider 抽象基类"""

    def __init__(self, api_key: str, proxy_url: Optional[str] = None):
        self.api_key = api_key
        self.proxy_url = proxy_url

    # ========================================================================
    # 必须实现的抽象方法
    # ========================================================================

    @property
    @abstractmethod
    def api_base_url(self) -> str:
        """异步 API 的基础 URL，如 https://cf-api.o1key.com"""
        ...

    @abstractmethod
    def get_submit_endpoint(self, model: str, resolution: str) -> str:
        """获取提交任务的 API 端点路径（不含 base_url）"""
        ...

    @abstractmethod
    def build_submit_body(
        self,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        images: Optional[List[Image.Image]] = None,
        **kwargs
    ) -> dict:
        """构建提交任务的请求体"""
        ...

    @abstractmethod
    def extract_task_id(self, response: dict) -> str:
        """从提交响应中提取 task_id"""
        ...

    @abstractmethod
    def extract_status(self, response: dict) -> str:
        """从轮询响应中提取任务状态（如 SUBMITTED / IN_PROGRESS / SUCCESS / FAILURE）"""
        ...

    @abstractmethod
    async def parse_result(
        self,
        result_data: dict,
        session
    ) -> List[Image.Image]:
        """从任务完成后的 result data 中解析生成的图像列表"""
        ...

    @abstractmethod
    def get_models(self) -> List[str]:
        """获取此 Provider 支持的模型 ID 列表"""
        ...

    @abstractmethod
    def get_model_aspect_ratios(self, model_id: str) -> List[str]:
        """获取指定模型支持的宽高比"""
        ...

    @abstractmethod
    def get_model_resolutions(self, model_id: str) -> List[str]:
        """获取指定模型支持的分辨率"""
        ...

    # ========================================================================
    # 可选的覆盖方法
    # ========================================================================

    def get_poll_endpoint(self, task_id: str) -> str:
        """获取轮询任务状态的 API 端点路径（默认实现适用于 o1key 异步 API）"""
        return f"/async/v1/tasks/{task_id}"

    def get_headers(self) -> dict:
        """获取 HTTP 请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_all_aspect_ratios(self) -> List[str]:
        """获取所有模型支持的宽高比（去重合并）"""
        seen = set()
        result = []
        for model_id in self.get_models():
            for ratio in self.get_model_aspect_ratios(model_id):
                if ratio not in seen:
                    seen.add(ratio)
                    result.append(ratio)
        return result

    def get_all_resolutions(self) -> List[str]:
        """获取所有模型支持的分辨率（去重，按固定顺序排列）"""
        _ORDER = ["512px", "1K", "2K", "4K"]
        seen = set()
        for model_id in self.get_models():
            for res in self.get_model_resolutions(model_id):
                seen.add(res)
        return [r for r in _ORDER if r in seen]

    def get_extra_inputs(self) -> dict:
        """
        返回此 Provider 特有的额外 ComfyUI 输入参数。
        子类重写以声明 Provider 专有的选项（如 Google Search Grounding）。

        Returns:
            dict，格式与 ComfyUI INPUT_TYPES 的 optional 字段一致
        """
        return {}

    def get_extra_kwargs(self, **kwargs) -> dict:
        """
        从 ComfyUI kwargs 中提取此 Provider 特有的参数，
        转换为 build_submit_body 可接收的 kwargs。

        子类重写以处理 Provider 专有参数。
        """
        return {}

    def extract_progress(self, response: dict) -> Optional[float]:
        """
        从轮询响应中提取生成进度。

        Args:
            response: 轮询接口返回的完整响应字典

        Returns:
            0.0-1.0 之间的进度值，或 None 表示该响应不含进度信息
        """
        return None

    def query_balance_sync(self) -> Optional[dict]:
        """
        同步查询账户余额（可选）。
        返回 None 表示不支持。
        """
        return None

    def format_balance_info(self, balance_data: dict) -> str:
        """格式化余额信息为展示文本"""
        return ""

    # ========================================================================
    # 工具方法
    # ========================================================================

    @staticmethod
    def build_proxy_url(port: str) -> Optional[str]:
        """
        将端口号字符串转为 aiohttp 可用的 HTTP 代理 URL。
        兼容 v2rayN (10808)、Clash Verge (7897) 等。

        Args:
            port: 用户填写的端口号，如 "7897"，空字符串返回 None

        Returns:
            代理 URL 或 None
        """
        port = (port or "").strip()
        if not port or not port.isdigit():
            return None
        return f"http://127.0.0.1:{port}"
