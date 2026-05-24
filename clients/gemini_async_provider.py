"""
Gemini 异步生图 Provider
通过 cf-api.o1key.com 的异步提交+轮询接口调用 Gemini 图像生成模型

协议说明：
  - 提交：POST {base}/async{gemini_endpoint}?image_format=url
  - 轮询：GET  {base}/async/v1/tasks/{task_id}
  - 结果：可能直接返回 image_url，也可能返回 Gemini 标准 candidates 格式
"""

from io import BytesIO
from typing import Dict, List, Optional
from PIL import Image

from .base_async_provider import BaseAsyncImageProvider
from .gemini_client import GeminiAPIClient
from ..utils.config import get_async_api_base_url, get_api_key_or_raise
from ..models_config import (
    get_enabled_models,
    get_model_supported_aspect_ratios,
    get_model_supported_resolutions,
)


class GeminiAsyncImageProvider(BaseAsyncImageProvider):
    """
    Gemini 异步生图 Provider

    委托 GeminiAPIClient 处理：
      - 端点构造（get_endpoint）
      - 请求体构建（build_request_body）
      - 响应解析（parse_response_async）
    """

    def __init__(self, api_key: str = None, proxy_url: str = None):
        if api_key is None:
            api_key = get_api_key_or_raise("O1KEY_API_KEY")
        super().__init__(api_key=api_key, proxy_url=proxy_url)
        self._client = GeminiAPIClient(api_key=api_key)

    # ========================================================================
    # 抽象方法实现
    # ========================================================================

    @property
    def api_base_url(self) -> str:
        return getattr(self, '_route_base_url', None) or get_async_api_base_url()

    def get_submit_endpoint(self, model: str, resolution: str) -> str:
        gemini_endpoint = self._client.get_endpoint(
            model=model, resolution=resolution, image_format="url"
        )
        base = gemini_endpoint.split("?")[0]
        async_endpoint = f"/async{base}"
        if "?" in gemini_endpoint:
            async_endpoint += "?" + gemini_endpoint.split("?", 1)[1]
        return async_endpoint

    def build_submit_body(
        self,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        images: Optional[List[Image.Image]] = None,
        **kwargs
    ) -> dict:
        return self._client.build_request_body(
            prompt=prompt,
            images=images,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            enable_grounding=kwargs.get("enable_grounding", False),
            enable_image_search=kwargs.get("enable_image_search", False),
            image_compression=getattr(self, 'image_compression', None),
        )

    def extract_task_id(self, response: dict) -> str:
        task_id = response.get("task_id")
        if not task_id:
            raise RuntimeError(f"提交响应中未找到 task_id: {response}")
        return task_id

    def extract_status(self, response: dict) -> str:
        return response.get("status", "UNKNOWN")

    async def parse_result(self, result_data: dict, session) -> List[Image.Image]:
        # 异步接口可能直接返回 image_url
        image_url = result_data.get("image_url", "") if isinstance(result_data, dict) else ""
        if image_url:
            async with session.get(image_url) as img_resp:
                if img_resp.status == 200:
                    img_bytes = await img_resp.read()
                    return [Image.open(BytesIO(img_bytes))]
                raise RuntimeError(f"下载图片失败 ({img_resp.status}): {image_url}")

        # 否则按 Gemini 标准格式解析
        images_list, _ = await self._client.parse_response_async(result_data, session=session)
        return images_list

    def get_models(self) -> List[str]:
        return get_enabled_models()

    def get_model_aspect_ratios(self, model_id: str) -> List[str]:
        return get_model_supported_aspect_ratios(model_id)

    def get_model_resolutions(self, model_id: str) -> List[str]:
        return get_model_supported_resolutions(model_id)

    # ========================================================================
    # 可选方法覆盖
    # ========================================================================

    def get_extra_inputs(self) -> dict:
        """Gemini 专有：Google Search Grounding"""
        return {
            "联网功能": (["关闭", "打开"], {"default": "关闭"}),
        }

    def get_extra_kwargs(self, **kwargs) -> dict:
        return {
            "enable_grounding": kwargs.pop("联网功能", "关闭") == "打开",
        }

    def extract_progress(self, response: dict) -> Optional[float]:
        """从轮询响应中提取进度（0.0-1.0）"""
        # 直接字段：progress / percentage
        for field in ("progress", "percentage"):
            val = response.get(field)
            if val is not None and isinstance(val, (int, float)):
                return val / 100.0 if val > 1 else float(val)

        # 嵌套字段：progressInfo / progress_info
        progress_info = response.get("progressInfo") or response.get("progress_info")
        if isinstance(progress_info, dict):
            for field in ("progress", "percentage"):
                val = progress_info.get(field)
                if val is not None and isinstance(val, (int, float)):
                    return val / 100.0 if val > 1 else float(val)

        return None

    def query_balance_sync(self) -> Optional[dict]:
        try:
            return self._client.query_balance_sync()
        except Exception:
            return None

    def format_balance_info(self, balance_data: dict) -> str:
        return self._client.format_balance_info(balance_data)
