"""
OpenAI 兼容 API 客户端
端点固定为 /v1/chat/completions，模型名放入请求体 model 字段
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


# 固定端点
_ENDPOINT = "/v1/chat/completions"


class OpenAIAPIClient(BaseAPIClient):
    """
    OpenAI 兼容格式的图像生成客户端

    与 GeminiAPIClient 的主要区别：
    - 端点固定为 /v1/chat/completions（不再动态拼模型名到 URL）
    - 解析后的模型字符串放入请求体的 model 字段
    - 请求体采用 messages 数组格式，图片以 data URI 内联
    - 顶层追加 modalities 和 image_config 字段
    - 响应解析对应 choices[0].message.content 结构
    """

    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = get_api_key_or_raise("O1KEY_API_KEY")

        super().__init__(
            base_url=get_api_base_url(),
            api_key=api_key,
            max_request_size=100 * 1024 * 1024
        )

    # ------------------------------------------------------------------ #
    #  模型名解析                                                           #
    #  原 GeminiAPIClient.get_endpoint() 里动态拼 URL 的逻辑               #
    #  现在改为：同样的输入 → 返回纯模型名字符串，放进请求体                    #
    # ------------------------------------------------------------------ #

    def resolve_model_name(self, model: str, resolution: str) -> str:
        """
        将「节点选中的模型 ID + 分辨率」解析为实际请求所用的模型名称。

        对应关系与原 GeminiAPIClient.get_endpoint() 完全一致，
        只是把拼在 URL 路径里的模型段提取出来单独返回。

        Args:
            model:      节点下拉框中的模型 ID，如 "nano-banana-pro-次卡"
            resolution: 分辨率字符串，如 "1K" / "2K" / "4K" / "512"

        Returns:
            实际模型名，如 "nano-banana-pro-2k"
        """
        # ── 动态端点模型 ──────────────────────────────────────────────────
        if model == "nano-banana-pro-次卡":
            if resolution == "1K":
                return "nano-banana-pro"
            elif resolution == "4K":
                return "nano-banana-pro-4k"
            else:  # 2K（默认）
                return "nano-banana-pro-2k"

        elif model == "nano-banana-pro-官方计费":
            if resolution == "1K":
                return "nano-banana-pro-1k-official"
            elif resolution == "4K":
                return "nano-banana-pro-4k-official"
            else:  # 2K（默认）
                return "nano-banana-pro-2k-official"

        elif model == "nano-banana-2-官方计费":
            if resolution == "512":
                return "nano-banana-2-0.5k-official"
            elif resolution == "1K":
                return "nano-banana-2-1k-official"
            elif resolution == "4K":
                return "nano-banana-2-4k-official"
            else:  # 2K（默认）
                return "nano-banana-2-2k-official"

        elif model == "gemini-3-pro-image-preview-url":
            if resolution == "1K":
                return "gemini-3-pro-image-preview-url"
            elif resolution == "4K":
                return "gemini-3-pro-image-preview-4k-url"
            else:  # 2K（默认）
                return "gemini-3-pro-image-preview-2k-url"

        # ── 固定端点模型：从 models_config 里取端点，提取模型名段 ──────────
        from ..models_config import get_model_endpoint
        endpoint = get_model_endpoint(model)
        if endpoint:
            # 端点格式：/v1beta/models/<model-name>:generateContent
            # 提取 <model-name> 部分
            match = re.search(r"/models/([^:]+):", endpoint)
            if match:
                return match.group(1)

        # ── 兜底：直接用 model ID ──────────────────────────────────────────
        return model

    # ------------------------------------------------------------------ #
    #  BaseAPIClient 抽象方法实现                                           #
    # ------------------------------------------------------------------ #

    def get_endpoint(self, **kwargs) -> str:
        """固定返回 /v1/chat/completions，模型信息已移入请求体。"""
        return _ENDPOINT

    def build_request_body(
        self,
        prompt: str = "",
        images: Optional[List[Image.Image]] = None,
        aspect_ratio: str = "1:1",
        resolution: str = "2K",
        model: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        构建 OpenAI /v1/chat/completions 格式请求体。

        文生图示例输出：
        {
            "model": "nano-banana-pro-2k",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "一个中国女子的OOTD"}
                    ]
                }
            ],
            "modalities": ["image", "text"],
            "stream": false,
            "extra_body": {
                "google": {
                    "image_config": {
                        "aspect_ratio": "16:9",
                        "image_size": "2K"
                    }
                }
            }
        }

        图生图时 content 数组追加若干 image_url 块：
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,<...>"}
            }

        Args:
            prompt:       提示词
            images:       参考图列表（可选，图生图时传入）
            aspect_ratio: 宽高比，如 "16:9"
            resolution:   分辨率，如 "2K"
            model:        已解析好的模型名（由 resolve_model_name 返回）
        """
        # ── 构建 content 数组 ─────────────────────────────────────────────
        content: List[Dict[str, Any]] = []

        # 1. 文本部分（始终在最前）
        content.append({
            "type": "text",
            "text": prompt
        })

        # 2. 图片部分（图生图时追加，每张图一个 image_url block）
        if images:
            for img in images:
                b64 = encode_image_to_base64(img)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    }
                })

        # ── 分辨率映射（节点内部值 → API 所需值） ────────────────────────────
        _resolution_map = {"512": "0.5K", "1K": "1K", "2K": "2K", "4K": "4K"}
        api_image_size = _resolution_map.get(resolution, resolution)

        # ── 组装完整请求体 ─────────────────────────────────────────────────
        request_body: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "modalities": ["image", "text"],
            "stream": False,
            "extra_body": {
                "google": {
                    "image_config": {
                        "aspect_ratio": aspect_ratio,
                        "image_size": api_image_size
                    }
                }
            }
        }

        return request_body

    def parse_response(self, response: Dict[str, Any]) -> List[Image.Image]:
        """同步 parse_response，仅为满足抽象基类要求，实际不应被直接调用。"""
        raise RuntimeError(
            "parse_response() 不应被直接调用。"
            "请使用 generate_single_async() 等高级方法。"
        )

    def get_http_error_message(self, status_code: int, error_message: str) -> Optional[str]:
        """429 / 503 友好文案。"""
        if status_code == 429:
            return (
                "莫慌!该模型暂时超出速率限制啦\n"
                "解决方案如下(任意一种):\n"
                "1.切换当前模型\n"
                "2.前往后台,修改令牌分组"
            )
        if status_code == 503:
            return (
                "警报!服务器当前过载!\n"
                "解决方案如下:\n"
                "1.摸会儿鱼吧,稍后会恢复,嘿嘿~\n"
                "2.切换其他模型\n"
                "3.前往后台,修改令牌分组"
            )
        return None

    # ------------------------------------------------------------------ #
    #  响应解析                                                             #
    # ------------------------------------------------------------------ #

    async def parse_response_async(
        self,
        response: Dict[str, Any],
        session: Optional[aiohttp.ClientSession] = None
    ) -> tuple[List[Image.Image], Dict[str, Any]]:
        """
        异步解析 /v1/chat/completions 格式响应，提取生成的图像。

        响应结构（OpenAI 格式）：
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text",  "text": "..."},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                            // 或直接 inline_data / inlineData（兼容 Gemini 风格回包）
                        ]
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {...}
        }
        """
        format_info: Dict[str, Any] = {
            "type": None,       # "base64" | "url"
            "size": 0,
            "resolution": None,
            "download_speed": None
        }

        # ── 错误前置检测 ───────────────────────────────────────────────────

        # 1. usage.completion_tokens == 0 → 风控拦截（对齐 Gemini 的 candidatesTokenCount==0）
        usage = response.get("usage", {})
        completion_tokens = usage.get("completion_tokens", -1)
        if completion_tokens == 0:
            raise RuntimeError(
                "Damn!你触发顶级风控啦!还没到生图阶段就被拒了。\n"
                "赶紧调整一下图片或提示词吧!该情况不会返回图片且正常扣费!下次小心哦~"
            )

        # 2. finish_reason 不是 "stop" → 安全过滤 / token 超限等
        choices = response.get("choices", [])
        if choices:
            for choice in choices:
                finish_reason = choice.get("finish_reason", "")
                if finish_reason and finish_reason != "stop":
                    raise RuntimeError(
                        "Ohh no! 生图过程触发风控,图片被拒绝生成!\n"
                        "可能原因如下:\n"
                        "1.违禁内容\n"
                        "2.触发安全过滤器\n"
                        "3.涉及版权问题\n"
                        "4. Token超限\n"
                        "赶紧调整一下图片或提示词吧!该情况不会返回图片且正常扣费!下次小心哦~"
                    )

        # ── 图像提取 ───────────────────────────────────────────────────────
        images: List[Image.Image] = []
        text_responses: List[str] = []

        close_session = False
        if session is None:
            session = self._make_session()
            close_session = True

        try:
            for choice in choices:
                message = choice.get("message", {})

                # ── 优先从 message.images 提取（非标准扩展字段） ──────────────
                # 部分服务端把图片放在独立的 images 字段，content 同时为 null
                msg_images = message.get("images") or []
                for img_part in msg_images:
                    part_type = img_part.get("type", "")
                    if part_type == "image_url":
                        url_obj = img_part.get("image_url", {})
                        url = url_obj.get("url", "")
                        if url.startswith("data:"):
                            try:
                                _, b64_data = url.split(",", 1)
                                img = decode_base64_to_pil(b64_data)
                                images.append(img)
                                if format_info["type"] is None:
                                    format_info["type"] = "base64"
                                    format_info["size"] = len(b64_data) * 3 / 4
                                    format_info["resolution"] = f"{img.size[0]}x{img.size[1]}"
                            except Exception:
                                pass
                        elif url.startswith("http"):
                            try:
                                dl_start = time.time()
                                async with session.get(url) as img_resp:
                                    if img_resp.status == 200:
                                        img_data = await img_resp.read()
                                        dl_time = time.time() - dl_start
                                        speed = len(img_data) / dl_time if dl_time > 0 else 0
                                        img = Image.open(BytesIO(img_data))
                                        images.append(img)
                                        if format_info["type"] is None:
                                            format_info["type"] = "url"
                                            format_info["size"] = len(img_data)
                                            format_info["resolution"] = f"{img.size[0]}x{img.size[1]}"
                                            format_info["download_speed"] = speed
                            except Exception:
                                pass

                # ── 再从 message.content 提取（标准 OpenAI 格式） ─────────────
                # content 为 null 时用空列表兜底，避免 for in None 崩溃
                raw_content = message.get("content") or []

                # content 可能是字符串（纯文本）或数组（多模态）
                if isinstance(raw_content, str):
                    text_responses.append(raw_content)
                    continue

                for part in raw_content:
                    part_type = part.get("type", "")

                    # ── 情况 A：OpenAI image_url 格式 ─────────────────────
                    if part_type == "image_url":
                        url_obj = part.get("image_url", {})
                        url = url_obj.get("url", "")

                        if url.startswith("data:"):
                            # data URI → 直接 base64 解码
                            # 格式：data:image/png;base64,<data>
                            try:
                                header, b64_data = url.split(",", 1)
                                img = decode_base64_to_pil(b64_data)
                                images.append(img)
                                if format_info["type"] is None:
                                    format_info["type"] = "base64"
                                    format_info["size"] = len(b64_data) * 3 / 4
                                    format_info["resolution"] = f"{img.size[0]}x{img.size[1]}"
                            except Exception:
                                pass

                        elif url.startswith("http"):
                            # 远程 URL → 异步下载
                            try:
                                dl_start = time.time()
                                async with session.get(url) as img_resp:
                                    if img_resp.status == 200:
                                        img_data = await img_resp.read()
                                        dl_time = time.time() - dl_start
                                        speed = len(img_data) / dl_time if dl_time > 0 else 0
                                        img = Image.open(BytesIO(img_data))
                                        images.append(img)
                                        if format_info["type"] is None:
                                            format_info["type"] = "url"
                                            format_info["size"] = len(img_data)
                                            format_info["resolution"] = f"{img.size[0]}x{img.size[1]}"
                                            format_info["download_speed"] = speed
                            except Exception:
                                pass

                    # ── 情况 B：Gemini 风格 inline_data / inlineData（兼容） ─
                    elif part_type in ("inline_data", "inlineData") or \
                            "inline_data" in part or "inlineData" in part:
                        inline_key = "inline_data" if "inline_data" in part else "inlineData"
                        inline = part.get(inline_key, {})
                        b64_data = inline.get("data", "")
                        if b64_data:
                            try:
                                img = decode_base64_to_pil(b64_data)
                                images.append(img)
                                if format_info["type"] is None:
                                    format_info["type"] = "base64"
                                    format_info["size"] = len(b64_data) * 3 / 4
                                    format_info["resolution"] = f"{img.size[0]}x{img.size[1]}"
                            except Exception:
                                pass

                    # ── 情况 C：text 中嵌套 URL（markdown 或纯链接） ─────────
                    elif part_type == "text":
                        text = part.get("text", "")
                        text_responses.append(text)

                        # markdown 图片链接：![alt](url)
                        urls = re.findall(r'!\[.*?\]\((https?://[^\)]+)\)', text)
                        if not urls:
                            urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)

                        for url in urls:
                            try:
                                dl_start = time.time()
                                async with session.get(url) as img_resp:
                                    if img_resp.status == 200:
                                        img_data = await img_resp.read()
                                        dl_time = time.time() - dl_start
                                        speed = len(img_data) / dl_time if dl_time > 0 else 0
                                        img = Image.open(BytesIO(img_data))
                                        images.append(img)
                                        if format_info["type"] is None:
                                            format_info["type"] = "url"
                                            format_info["size"] = len(img_data)
                                            format_info["resolution"] = f"{img.size[0]}x{img.size[1]}"
                                            format_info["download_speed"] = speed
                            except Exception:
                                pass

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"解析 API 响应失败: {str(e)}")
        finally:
            if close_session:
                await session.close()

        # ── 3. 无图像但有文本 → API 拒绝说明 ─────────────────────────────
        if not images and text_responses:
            combined = "\n".join(text_responses)
            raise RuntimeError(
                f"API 拒绝响应\n\n"
                f"API 返回说明：\n{combined}\n\n"
                f"建议：\n"
                f"  - 根据上述说明调整请求内容\n"
                f"  - 确保提示词和参考图符合使用规范"
            )

        if not images:
            raise RuntimeError("API 响应中未找到生成的图像")

        return images, format_info

    # ------------------------------------------------------------------ #
    #  核心生成方法（接口与 GeminiAPIClient 保持一致，节点可无缝切换）        #
    # ------------------------------------------------------------------ #

    async def generate_single_async(
        self,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        images: Optional[List[Image.Image]] = None,
        session: Optional[aiohttp.ClientSession] = None,
        task_index: Optional[int] = None,
        total_tasks: Optional[int] = None,
        debug: bool = False,
        debug_request: bool = False,
        enable_grounding: bool = False,      # 保留签名兼容，OpenAI 格式暂不使用
        enable_image_search: bool = False    # 保留签名兼容，OpenAI 格式暂不使用
    ) -> tuple[List[Image.Image], Dict[str, Any]]:
        """
        单次异步生成请求（OpenAI /v1/chat/completions 格式）。

        Args:
            prompt:        提示词
            model:         节点选中的模型 ID（将自动解析为实际模型名）
            resolution:    分辨率
            aspect_ratio:  宽高比
            images:        参考图列表（图生图时传入）
            session:       复用的 aiohttp 会话
            task_index:    任务序号（批量时用于日志）
            total_tasks:   总任务数（批量时用于日志）
            debug:         打印完整 API 响应
            debug_request: 打印请求体（base64 自动截断）

        Returns:
            (生成的图像列表, 计时信息字典)
        """
        import json

        total_start = time.time()
        task_prefix = f"[{task_index}/{total_tasks}]" if task_index is not None and total_tasks else ""

        # ── 1. 解析模型名 & 构建请求体 ────────────────────────────────────
        build_start = time.time()
        resolved_model = self.resolve_model_name(model, resolution)
        endpoint = self.get_endpoint()

        request_body = self.build_request_body(
            prompt=prompt,
            images=images,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            model=resolved_model
        )
        build_time = time.time() - build_start

        # ── 调试：打印请求体 ───────────────────────────────────────────────
        if debug_request:
            import json as _json
            def _shorten_b64(obj):
                if isinstance(obj, dict):
                    return {k: _shorten_b64(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_shorten_b64(i) for i in obj]
                if isinstance(obj, str):
                    if obj.startswith("data:"):
                        header, _, data = obj.partition(",")
                        return f"{header},<base64 {len(data)} chars>"
                    if len(obj) > 200 and all(
                        c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
                        for c in obj[:64]
                    ):
                        return f"<base64 {len(obj)} chars>"
                return obj
            print(
                f"\n{'='*60}\n"
                f"[请求体日志] 任务 {task_prefix or '?'}\n"
                f"端点: {self.base_url}{endpoint}\n"
                f"{_json.dumps(_shorten_b64(request_body), ensure_ascii=False, indent=2)}\n"
                f"{'='*60}\n"
            )

        # ── 2. 计算请求体大小 ─────────────────────────────────────────────
        request_size = len(json.dumps(request_body).encode("utf-8"))
        size_str = (
            f"{request_size / 1024:.2f}KB"
            if request_size < 1024 * 1024
            else f"{request_size / (1024 * 1024):.2f}MB"
        )

        # ── 3. 发送请求（Bearer Token 认证） ─────────────────────────────
        request_start = time.time()
        try:
            response = await self.request_async(
                endpoint,
                request_body,
                session,
                use_bearer_token=True
            )
        except Exception as e:
            request_time = time.time() - request_start
            error_first_line = str(e).split("\n")[0]
            print(f"{task_prefix} 请求 {size_str} → API {request_time:.1f}s → 失败: {error_first_line} ✗")
            raise

        request_time = time.time() - request_start

        # ── 调试：打印完整响应 ─────────────────────────────────────────────
        if debug:
            import json as _json
            def _shorten_b64(obj):
                if isinstance(obj, dict):
                    return {k: _shorten_b64(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_shorten_b64(i) for i in obj]
                if isinstance(obj, str):
                    if obj.startswith("data:"):
                        header, _, data = obj.partition(",")
                        return f"{header},<base64 {len(data)} chars>"
                    if len(obj) > 200 and all(
                        c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
                        for c in obj[:64]
                    ):
                        return f"<base64 {len(obj)} chars>"
                return obj
            print(
                f"\n{'='*60}\n"
                f"[调试日志] 任务 {task_prefix or '?'} 完整 API 响应：\n"
                f"{_json.dumps(_shorten_b64(response), ensure_ascii=False, indent=2)}\n"
                f"{'='*60}\n"
            )

        # ── 4. 解析响应 ───────────────────────────────────────────────────
        parse_start = time.time()
        try:
            result_images, format_info = await self.parse_response_async(response, session)
        except Exception as e:
            parse_time = time.time() - parse_start
            error_first_line = str(e).split("\n")[0]
            print(f"{task_prefix} 请求 {size_str} → API {request_time:.1f}s → 解析失败: {error_first_line} ✗")
            raise

        parse_time = time.time() - parse_start

        # ── 5. 单行日志输出 ───────────────────────────────────────────────
        img_size = format_info.get("size", 0)
        img_size_str = (
            f"{img_size / 1024:.2f}KB"
            if img_size < 1024 * 1024
            else f"{img_size / (1024 * 1024):.2f}MB"
        )

        if format_info.get("type") == "base64":
            download_info = f"Base64 {img_size_str} ({parse_time:.1f}s)"
        elif format_info.get("type") == "url":
            speed = format_info.get("download_speed", 0)
            download_info = f"URL {img_size_str} ({parse_time:.1f}s, {speed / (1024*1024):.1f}MB/s)"
        else:
            download_info = img_size_str

        timing = response.get("_timing", {})
        net_connect = timing.get("connect_time")
        net_download = timing.get("download_time")
        if net_connect is not None and net_download is not None:
            net_str = f" | 连接 {net_connect:.2f}s | 下载 {net_download:.2f}s"
        else:
            net_str = ""
        print(f"{task_prefix} 请求 {size_str} → API {request_time:.1f}s → {download_info} ✓{net_str}")

        total_time = time.time() - total_start
        timing_info = {
            "build_time": build_time,
            "request_time": request_time,
            "parse_time": parse_time,
            "total_time": total_time,
            "format_type": format_info.get("type", "unknown")
        }

        return result_images, timing_info

    # ------------------------------------------------------------------ #
    #  批量 & 同步接口（与 GeminiAPIClient 接口签名一致）                    #
    # ------------------------------------------------------------------ #

    async def generate_batch_async(
        self,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        batch_size: int,
        images: Optional[List[Image.Image]] = None,
        progress_callback: Optional[Callable[[int, int, bool, Optional[str]], None]] = None,
        debug: bool = False,
        debug_request: bool = False,
        enable_grounding: bool = False,
        enable_image_search: bool = False
    ) -> List[Image.Image]:
        """批量全并发生成（单提示词 × batch_size 张）。"""
        import asyncio

        all_images: List[Image.Image] = []
        completed = 0
        success_count = 0
        fail_count = 0
        first_error = None

        max_concurrent = 10
        num_batches = (batch_size + max_concurrent - 1) // max_concurrent

        print(f"OpenAIClient: 批量生成 {batch_size} 张，并发数: {max_concurrent}，分 {num_batches} 批")

        connector = aiohttp.TCPConnector(ssl=False, limit=0, limit_per_host=0)

        async with aiohttp.ClientSession(connector=connector) as session:
            for batch_idx in range(num_batches):
                batch_start = batch_idx * max_concurrent
                batch_end = min(batch_start + max_concurrent, batch_size)
                batch_count = batch_end - batch_start

                if num_batches > 1:
                    print(f"OpenAIClient: 第 {batch_idx + 1}/{num_batches} 批 ({batch_start + 1}-{batch_end})")

                tasks = [
                    asyncio.create_task(
                        self.generate_single_async(
                            prompt=prompt,
                            model=model,
                            resolution=resolution,
                            aspect_ratio=aspect_ratio,
                            images=images,
                            session=session,
                            task_index=batch_start + i + 1,
                            total_tasks=batch_size,
                            debug=debug,
                            debug_request=debug_request
                        ),
                        name=f"task_{batch_start + i}"
                    )
                    for i in range(batch_count)
                ]

                batch_images: List[Image.Image] = []

                for coro in asyncio.as_completed(tasks):
                    completed += 1
                    try:
                        result_imgs, _ = await coro
                        for img in result_imgs:
                            batch_images.append(img)
                            all_images.append(img)
                        success_count += 1
                        if progress_callback:
                            progress_callback(completed, batch_size, True, None)
                        print(f"OpenAIClient: 任务 {completed}/{batch_size} 成功 ✓")
                    except Exception as e:
                        fail_count += 1
                        if first_error is None:
                            first_error = e
                        if progress_callback:
                            progress_callback(completed, batch_size, False, str(e))
                        print(f"OpenAIClient: 任务 {completed}/{batch_size} 失败 ✗")

                if batch_images:
                    print(f"OpenAIClient: 第 {batch_idx + 1} 批完成，生成 {len(batch_images)} 张")
                    import gc
                    gc.collect()
                    await asyncio.sleep(0.1)

                batch_images = []

        if not all_images:
            if first_error:
                raise first_error
            raise RuntimeError(f"批量生成失败，{fail_count} 个请求全部失败")

        print(f"OpenAIClient: 批量完成，成功 {success_count}/{batch_size}，失败 {fail_count}")
        return all_images

    def generate_sync(
        self,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        batch_size: int,
        images: Optional[List[Image.Image]] = None,
        progress_callback: Optional[Callable[[int, int, bool, Optional[str]], None]] = None,
        debug: bool = False,
        debug_request: bool = False,
        enable_grounding: bool = False,
        enable_image_search: bool = False
    ) -> List[Image.Image]:
        """同步生成接口（用于 ComfyUI 节点，接口与 GeminiAPIClient 完全一致）。"""
        coro = self.generate_batch_async(
            prompt=prompt,
            model=model,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            batch_size=batch_size,
            images=images,
            progress_callback=progress_callback,
            debug=debug,
            debug_request=debug_request,
            enable_grounding=enable_grounding,
            enable_image_search=enable_image_search
        )
        return self.run_async_in_thread(coro)
