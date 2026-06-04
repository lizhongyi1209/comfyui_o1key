"""
GPT Image API 客户端
新版节点请求走异步任务接口：
  - POST /async/v1/generateImage
  - GET  /async/v1/tasks/{task_id}

旧同步接口保留兼容代码，但 GPT Image 节点不再使用：
  - POST /v1/images/generations/
  - POST /v1/images/edits

设计原则：
  - 对 ComfyUI 节点暴露同步入口，内部提交异步任务并轮询
  - 图片和蒙版以 data:image/png;base64,... 放入 JSON 请求体
  - 响应优先读取 data.images[].url
"""

import asyncio
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import numpy as np
import torch
from PIL import Image

from ..utils.config import get_api_key_or_raise, get_api_base_url
from ..utils.image_utils import tensor_to_pil, encode_image_to_base64
from ..utils.http_error import RETRYABLE_STATUS_CODES, HTTP_ERROR_MESSAGES, _compute_delay, DEFAULT_MAX_RETRIES, DEFAULT_BASE_DELAY, DEFAULT_MAX_DELAY, DEFAULT_BACKOFF_FACTOR, get_friendly_message

try:
    from comfy.model_management import processing_interrupted, InterruptProcessingException
    _INTERRUPT_AVAILABLE = True
except ImportError:
    _INTERRUPT_AVAILABLE = False
    InterruptProcessingException = RuntimeError
    processing_interrupted = lambda: False

# ── 接口端点 ──────────────────────────────────────────────────────────────────
_ENDPOINT_GENERATIONS = "/v1/images/generations/"
_ENDPOINT_EDITS       = "/v1/images/edits"
_ENDPOINT_ASYNC_GENERATE = "/async/v1/generateImage"
_ENDPOINT_ASYNC_TASK = "/async/v1/tasks/{task_id}"

# ── 模型名映射（UI 显示名 → API 实际参数名）─────────────────────────────────
_MODEL_NAME_MAP = {
    "gpt-image-2-按量": "gpt-image-2",
    "gpt-image-2-次卡": "gpt-image-2-special",
}

# ── 超时 ──────────────────────────────────────────────────────────────────────
_REQUEST_TIMEOUT = 900   # 秒
_ASYNC_POLL_SCHEDULE = [5.0, 20.0]
_ASYNC_POLL_INTERVAL = 3.0
_ASYNC_MAX_WAIT = 600.0
_ASYNC_RETRY_DELAYS = [2.0, 5.0, 10.0]
_ASYNC_RETRYABLE_ERROR_CODES = {
    "image_rate_limited",
    "image_upstream_busy",
    "image_timeout",
    "image_storage_failed",
    "image_empty_result",
    "image_upstream_error",
    "image_internal_error",
    "image_unknown_error",
}
_ASYNC_RETRYABLE_ERROR_CATEGORIES = {
    "rate_limit",
    "upstream_busy",
    "timeout",
    "storage",
    "upstream_error",
    "internal_error",
    "unknown",
}
_ASYNC_NON_RETRYABLE_ERROR_CODES = {
    "image_invalid_size",
    "image_payload_too_large",
    "image_invalid_mask",
    "image_invalid_parameter",
    "image_safety_blocked",
    "image_provider_quota_exceeded",
    "image_provider_permission_required",
    "image_model_unavailable",
    "image_reference_download_failed",
}

REQUEST_LOG_ENABLED = False
POLL_LOG_ENABLED = False


class _AsyncImageTaskFailure(RuntimeError):
    def __init__(self, message: str, error_detail: Optional[dict] = None):
        super().__init__(message)
        self.error_detail = error_detail or {}


class GptImageClient:
    """
    GPT Image API 客户端

    接口说明：
      async generateImage：JSON 提交，返回 task_id
      tasks/{task_id}：轮询任务状态，成功后读取 data.images[].url

    旧 generations / edits 同步接口保留为兼容代码。
    """

    def __init__(self):
        self.api_key  = get_api_key_or_raise("O1KEY_API_KEY")
        self.base_url = get_api_base_url()

    # ── 认证头 ────────────────────────────────────────────────────────────────

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _json_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _new_multipart_form() -> aiohttp.FormData:
        try:
            return aiohttp.FormData(default_to_multipart=True)
        except TypeError:
            form = aiohttp.FormData()
            form._is_multipart = True
            return form

    @staticmethod
    def _add_form_fields(form: aiohttp.FormData, fields: dict) -> None:
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, bool):
                value = "true" if value else "false"
            elif isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            form.add_field(key, str(value))

    # ── 图像转换工具 ──────────────────────────────────────────────────────────

    # ── 请求体大小限制 ────────────────────────────────────────────────────────
    _MAX_BODY_BYTES = 20 * 1024 * 1024   # 20 MB

    @staticmethod
    def _shrink_png_to_limit(png_bytes: bytes, max_bytes: int, label: str = "") -> bytes:
        """
        若 PNG bytes 超过 max_bytes，按等比缩放反复压缩直到满足限制。
        每次将面积缩小至约 80%（线性尺寸缩小至约 89.4%）。
        """
        if len(png_bytes) <= max_bytes:
            return png_bytes

        img = Image.open(BytesIO(png_bytes))
        w, h = img.size
        original_size = len(png_bytes)
        step = 0

        while len(png_bytes) > max_bytes:
            scale = 0.894   # sqrt(0.8)，面积缩小 20%
            w = max(1, int(w * scale))
            h = max(1, int(h * scale))
            img = img.resize((w, h), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            step += 1

        tag = f" ({label})" if label else ""
        print(
            f"[o1key GPT Image] 图像{tag}超出 {max_bytes // (1024*1024)}MB 限制，"
            f"已等比缩放 {step} 次：{original_size // 1024}KB → {len(png_bytes) // 1024}KB "
            f"（{w}×{h}）"
        )
        return png_bytes

    @staticmethod
    def _png_bytes_to_data_url(png_bytes: bytes) -> str:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"

    @staticmethod
    def _json_body_size(body: dict) -> int:
        return len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    @staticmethod
    def _format_body_size(size: int) -> str:
        if size < 1024 * 1024:
            return f"{size / 1024:.2f}KB"
        return f"{size / 1024 / 1024:.2f}MB"

    @staticmethod
    def _shorten_data_urls_for_log(obj, max_len: int = 200):
        if isinstance(obj, dict):
            return {
                key: GptImageClient._shorten_data_urls_for_log(value, max_len)
                for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [GptImageClient._shorten_data_urls_for_log(item, max_len) for item in obj]
        if isinstance(obj, str) and obj.startswith("data:image") and len(obj) > max_len:
            header, _, data = obj.partition(",")
            return f"{header},<base64 data, {len(data)} chars>"
        return obj

    def _log_original_request_body(self, label: str, body: dict) -> None:
        if not REQUEST_LOG_ENABLED:
            return
        body_size = self._json_body_size(body)
        print(
            f"\n{'=' * 60}\n"
            f"[o1key GPT Image] 原始请求体日志 | {label} | "
            f"请求体积: {self._format_body_size(body_size)} "
            f"(data URL 已折叠显示 base64 长度)\n"
            f"{json.dumps(self._shorten_data_urls_for_log(body), ensure_ascii=False, indent=2)}\n"
            f"{'=' * 60}\n"
        )

    def _log_original_response_body(self, label: str, text: str) -> None:
        if not REQUEST_LOG_ENABLED:
            return
        size = len(text.encode("utf-8"))
        print(
            f"\n{'=' * 60}\n"
            f"[o1key GPT Image] 原始返回响应体日志 | {label} | "
            f"响应体积: {self._format_body_size(size)}\n"
            f"{text}\n"
            f"{'=' * 60}\n"
        )

    @staticmethod
    def _extract_error_message(payload_or_text, status_code: int = 0) -> str:
        payload = payload_or_text
        if isinstance(payload_or_text, str):
            try:
                payload = json.loads(payload_or_text)
            except Exception:
                return get_friendly_message(status_code, payload_or_text)

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, str) and error.strip():
                return error.strip()
            if isinstance(error, dict):
                msg = error.get("message") or error.get("msg") or error.get("error")
                if msg:
                    return str(msg)
                return json.dumps(error, ensure_ascii=False)

            msg = payload.get("message") or payload.get("msg")
            if msg:
                return str(msg)

        return get_friendly_message(status_code, str(payload_or_text))

    @staticmethod
    def _extract_error_detail(payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {}
        detail = payload.get("error_detail")
        return detail if isinstance(detail, dict) else {}

    @staticmethod
    def _should_retry_async_failure(error_detail: dict, retry_index: int) -> bool:
        if not isinstance(error_detail, dict) or not error_detail:
            return False

        code = error_detail.get("code")
        category = error_detail.get("category")
        retryable = error_detail.get("retryable")

        if code in _ASYNC_NON_RETRYABLE_ERROR_CODES:
            return False
        if code == "image_unknown_error":
            return retry_index == 0
        if retryable is True:
            return True
        if retryable is False:
            return False

        return (
            code in _ASYNC_RETRYABLE_ERROR_CODES
            or category in _ASYNC_RETRYABLE_ERROR_CATEGORIES
        )

    @staticmethod
    def _coerce_progress_percent(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip().rstrip("%")
            if not text:
                return None
            try:
                value = float(text)
            except ValueError:
                return None
        elif isinstance(value, (int, float)):
            value = float(value)
        else:
            return None

        if 0 <= value <= 1:
            value *= 100
        return max(0, min(100, int(round(value))))

    @staticmethod
    def _emit_progress(progress_callback: Optional[Callable[[int], None]], pct: int) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(pct)
        except Exception as error:
            print(f"[o1key GPT Image] progress callback failed: {error}")

    @staticmethod
    def _resize_png_bytes(source_image: Image.Image, scale: float) -> bytes:
        if scale < 0.999:
            width, height = source_image.size
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            image = source_image.resize((new_width, new_height), Image.LANCZOS)
        else:
            image = source_image

        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _pil_to_png_bytes(image: Image.Image) -> bytes:
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def _fit_png_assets_to_body_limit(self, assets: List[Dict[str, Any]], build_body) -> Dict[str, bytes]:
        """
        根据完整 JSON 请求体大小压缩图片资产，保证最终 body 不超过 20MB。
        使用同一个缩放比例二分搜索，让压缩结果尽量贴近上限而不是过度压缩。
        """
        asset_bytes = {asset["key"]: asset["bytes"] for asset in assets}
        initial_size = self._json_body_size(build_body(asset_bytes))
        if initial_size <= self._MAX_BODY_BYTES:
            return asset_bytes

        if not assets:
            raise RuntimeError(
                f"请求体大小 {initial_size // 1024}KB 超过 20MB，且没有可压缩图片"
            )

        originals = []
        for asset in assets:
            image = Image.open(BytesIO(asset["bytes"]))
            image.load()
            originals.append((asset, image.copy()))

        low = 0.001
        high = 1.0
        best_bytes = None
        best_size = 0
        best_scale = 0.0

        for _ in range(16):
            scale = (low + high) / 2
            candidate = {}
            for asset, image in originals:
                candidate[asset["key"]] = self._resize_png_bytes(image, scale)

            body_size = self._json_body_size(build_body(candidate))
            if body_size <= self._MAX_BODY_BYTES:
                best_bytes = candidate
                best_size = body_size
                best_scale = scale
                low = scale
            else:
                high = scale

        if best_bytes is None:
            candidate = {}
            for asset, image in originals:
                candidate[asset["key"]] = self._resize_png_bytes(image, low)
            body_size = self._json_body_size(build_body(candidate))
            if body_size > self._MAX_BODY_BYTES:
                raise RuntimeError(
                    f"图片已压缩到最小比例，但请求体仍超过 20MB：{body_size // 1024}KB"
                )
            best_bytes = candidate
            best_size = body_size
            best_scale = low

        print(
            f"[o1key GPT Image] 请求体超过 20MB，已等比压缩图片："
            f"{initial_size // 1024}KB → {best_size // 1024}KB，scale={best_scale:.3f}"
        )
        return best_bytes

    @staticmethod
    def _tensor_to_png_bytes(tensor: torch.Tensor) -> bytes:
        """
        单张 ComfyUI IMAGE tensor [1, H, W, C] 或 [H, W, C] → PNG bytes
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)              # [H, W, C]
        arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _mask_tensor_to_rgba_png_bytes(mask: torch.Tensor, image_size: tuple) -> bytes:
        """
        ComfyUI MASK tensor [1, H, W] 或 [H, W] → RGBA PNG bytes
        白色区域（mask=1）→ 透明（alpha=0），即 API 将在此处生成新内容。
        """
        if mask.dim() == 3:
            mask = mask.squeeze(0)                  # [H, W]

        h, w = mask.shape
        ih, iw = image_size

        # 尺寸不一致时给出提示（API 侧也会报错）
        if (h, w) != (ih, iw):
            raise ValueError(
                f"蒙版尺寸 ({h}×{w}) 与图像尺寸 ({ih}×{iw}) 不一致，请保持相同尺寸"
            )

        alpha = ((1.0 - mask.cpu().numpy()) * 255).clip(0, 255).astype(np.uint8)
        rgba  = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 3] = alpha                       # 只设 alpha，RGB 全 0

        buf = BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _pil_list_to_tensor(images: List[Image.Image]) -> torch.Tensor:
        """
        PIL Image 列表 → ComfyUI IMAGE tensor [B, H, W, C]，值域 [0, 1]
        RGBA 自动转换为 RGBA（保留透明通道）
        """
        if not images:
            placeholder = Image.new("RGBA", (512, 512), (128, 128, 128, 255))
            images = [placeholder]

        tensors = []
        for img in images:
            arr = np.array(img.convert("RGBA")).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))

        # 批量模式下 API 可能返回不同尺寸，统一 resize 到最大尺寸
        max_h = max(t.shape[0] for t in tensors)
        max_w = max(t.shape[1] for t in tensors)
        aligned = []
        for t in tensors:
            if t.shape[0] != max_h or t.shape[1] != max_w:
                t = t.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                t = torch.nn.functional.interpolate(
                    t, size=(max_h, max_w), mode="bilinear", align_corners=False
                )
                t = t.squeeze(0).permute(1, 2, 0)  # [H, W, C]
            aligned.append(t)

        return torch.stack(aligned, dim=0)          # [B, H, W, 4]

    # ── 响应解析（通用） ─────────────────────────────────────────────────────

    async def _parse_response(
        self,
        resp_json: dict,
        session: aiohttp.ClientSession,
    ) -> List[Image.Image]:
        """
        解析 data 列表，优先取 b64_json，回退到 url 下载
        """
        if "error" in resp_json:
            err = resp_json["error"]
            msg = (
                err.get("message") or err.get("msg") or json.dumps(err, ensure_ascii=False)
                if isinstance(err, dict)
                else str(err)
            )
            raise RuntimeError(f"API 返回错误: {msg}")

        data_list = resp_json.get("data")
        if data_list is None:
            data_list = resp_json.get("images")
        if data_list is None and (resp_json.get("b64_json") or resp_json.get("url")):
            data_list = [resp_json]
        if not data_list:
            raise RuntimeError(
                f"API 响应中未找到 data 字段，完整响应：\n"
                f"{json.dumps(resp_json, ensure_ascii=False, indent=2)}"
            )

        images: List[Image.Image] = []
        for idx, item in enumerate(data_list):
            b64 = item.get("b64_json", "")
            url = item.get("url", "")

            if b64:
                # 优先 base64（无需二次下载）
                try:
                    img_bytes = base64.b64decode(b64)
                    img = Image.open(BytesIO(img_bytes))
                    images.append(img)
                    print(f"[o1key GPT Image] 第 {idx + 1} 张 base64 解码完成 "
                          f"({img.size[0]}×{img.size[1]})")
                except Exception as e:
                    raise RuntimeError(f"第 {idx + 1} 张 base64 解码失败: {e}")

            elif url and url.startswith("http"):
                # 回退：下载 URL
                async with session.get(url, allow_redirects=True) as r:
                    if r.status != 200:
                        raise RuntimeError(
                            f"图像下载失败 HTTP {r.status}，URL: {url}"
                        )
                    img_bytes = await r.read()
                img = Image.open(BytesIO(img_bytes))
                images.append(img)
                print(f"[o1key GPT Image] 第 {idx + 1} 张下载完成 "
                      f"({img.size[0]}×{img.size[1]})")
            else:
                print(f"[o1key GPT Image] 警告：第 {idx + 1} 条数据既无 b64_json 也无 url，已跳过")

        return images

    @staticmethod
    def _decode_b64_image(b64: str, label: str) -> Image.Image:
        try:
            img_bytes = base64.b64decode(b64)
            img = Image.open(BytesIO(img_bytes))
            print(f"[o1key GPT Image] {label} base64 解码完成 ({img.size[0]}×{img.size[1]})")
            return img
        except Exception as e:
            raise RuntimeError(f"{label} base64 解码失败: {e}") from None

    async def _append_images_from_payload(
        self,
        payload: dict,
        session: aiohttp.ClientSession,
        images: List[Image.Image],
        event_name: str = "",
    ) -> bool:
        if isinstance(payload, dict) and "error" in payload:
            err = payload["error"]
            msg = (
                err.get("message") or err.get("msg") or json.dumps(err, ensure_ascii=False)
                if isinstance(err, dict)
                else str(err)
            )
            raise RuntimeError(get_friendly_message(500, msg)) from None

        if not isinstance(payload, dict):
            return False

        event_type = payload.get("type") or event_name
        if "partial_image" in event_type:
            return False

        for key in ("data", "images"):
            data_list = payload.get(key)
            if isinstance(data_list, list):
                parsed = await self._parse_response({"data": data_list}, session)
                images.extend(parsed)
                return True

        if payload.get("b64_json") or payload.get("url"):
            parsed = await self._parse_response({"data": [payload]}, session)
            images.extend(parsed)
            return True

        image_obj = payload.get("image")
        if isinstance(image_obj, dict) and (image_obj.get("b64_json") or image_obj.get("url")):
            parsed = await self._parse_response({"data": [image_obj]}, session)
            images.extend(parsed)
            return True

        return False

    async def _parse_edit_stream_response(
        self,
        resp: aiohttp.ClientResponse,
        session: aiohttp.ClientSession,
    ) -> List[Image.Image]:
        """
        Parse /v1/images/edits SSE events and return final completed images.
        Partial images are intentionally ignored so the node output stays unchanged.
        """
        images: List[Image.Image] = []
        buffer = ""
        event_name = ""
        data_lines = []
        partial_count = 0
        debug_body_parts = []

        async def _handle_event():
            nonlocal event_name, data_lines, partial_count, images
            if not data_lines:
                event_name = ""
                return

            data_str = "\n".join(data_lines).strip()
            event_name = event_name.strip()
            data_lines = []

            if not data_str or data_str == "[DONE]":
                return

            try:
                payload = json.loads(data_str)
            except Exception:
                raise RuntimeError(get_friendly_message(500, data_str)) from None

            if isinstance(payload, dict):
                event_type = payload.get("type") or event_name
            else:
                event_type = event_name
            if "partial_image" in event_type:
                partial_count += 1
                return

            await self._append_images_from_payload(payload, session, images, event_name)

        async for raw_chunk in resp.content.iter_any():
            chunk_text = raw_chunk.decode("utf-8", errors="ignore")
            debug_body_parts.append(chunk_text)
            buffer += chunk_text
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line == "":
                    await _handle_event()
                    event_name = ""
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].lstrip())

        if buffer.strip():
            data_lines.append(buffer.strip())
        await _handle_event()

        if partial_count:
            print(f"[o1key GPT Image] 流式中间图 {partial_count} 张（已忽略，仅输出最终图）")
        if not images:
            raise RuntimeError("流式响应结束，但未收到最终图片")

        return images

    async def _parse_stream_text_response(
        self,
        text: str,
        session: aiohttp.ClientSession,
    ) -> List[Image.Image]:
        images: List[Image.Image] = []
        partial_count = 0
        event_name = ""
        data_lines = []

        async def _handle_event():
            nonlocal event_name, data_lines, partial_count, images
            if not data_lines:
                event_name = ""
                return
            data_str = "\n".join(data_lines).strip()
            event_name = event_name.strip()
            data_lines = []
            if not data_str or data_str == "[DONE]":
                return
            payload = json.loads(data_str)
            if isinstance(payload, dict):
                event_type = payload.get("type") or event_name
            else:
                event_type = event_name
            if "partial_image" in event_type:
                partial_count += 1
                return
            await self._append_images_from_payload(payload, session, images, event_name)

        for raw_line in text.splitlines():
            line = raw_line.rstrip("\r")
            if line == "":
                await _handle_event()
                event_name = ""
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip())
        await _handle_event()

        if partial_count:
            print(f"[o1key GPT Image] 流式中间图 {partial_count} 张（已忽略，仅输出最终图）")
        if not images:
            raise RuntimeError("流式响应结束，但未收到最终图片")
        return images

    async def _parse_success_response(
        self,
        resp: aiohttp.ClientResponse,
        session: aiohttp.ClientSession,
        label: str = "",
    ) -> List[Image.Image]:
        content_type = resp.headers.get("Content-Type", "").lower()
        if "event-stream" in content_type:
            return await self._parse_edit_stream_response(resp, session)

        text = await resp.text()
        stripped = text.lstrip()
        if stripped.startswith("data:") or stripped.startswith("event:"):
            return await self._parse_stream_text_response(text, session)

        try:
            resp_json = json.loads(text)
        except Exception:
            raise RuntimeError(f"响应 JSON 解析失败，原始内容：{text[:500]}")
        return await self._parse_response(resp_json, session)

    # ── 中断轮询 ──────────────────────────────────────────────────────────────

    @staticmethod
    async def _poll_interrupt():
        """每 0.5s 轮询一次 ComfyUI 中断标志"""
        while True:
            await asyncio.sleep(0.5)
            if _INTERRUPT_AVAILABLE and processing_interrupted():
                return

    @staticmethod
    async def _run_with_interrupt(coro):
        """
        将异步任务与中断轮询并发执行。
        如果用户点击取消，cancel 掉 coro 并抛出 InterruptProcessingException。
        """
        if not _INTERRUPT_AVAILABLE:
            return await coro

        request_task = asyncio.ensure_future(coro)
        interrupt_task = asyncio.ensure_future(GptImageClient._poll_interrupt())

        done, pending = await asyncio.wait(
            [request_task, interrupt_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        if interrupt_task in done and request_task not in done:
            raise InterruptProcessingException()

        return request_task.result()

    # ── 新版异步 GPT Image 接口 ──────────────────────────────────────────────

    def _build_async_generate_body(
        self,
        prompt: str,
        model: str,
        quality: str,
        size: str,
        n: int,
        image_list: Optional[List[torch.Tensor]] = None,
        mask_tensor: Optional[torch.Tensor] = None,
        output_format: str = "png",
    ) -> dict:
        api_model = _MODEL_NAME_MAP.get(model, model)
        assets: List[Dict[str, Any]] = []
        image_keys: List[str] = []
        mask_key = None

        if image_list:
            for idx_img, img_tensor in enumerate(image_list):
                pil_images = tensor_to_pil(img_tensor)
                if not pil_images:
                    continue
                key = f"image_{idx_img}"
                image_keys.append(key)
                assets.append({
                    "key": key,
                    "label": f"参考图{idx_img + 1}",
                    "bytes": self._pil_to_png_bytes(pil_images[0]),
                })

        if mask_tensor is not None:
            if not image_list:
                raise ValueError("提供了蒙版但未提供图片，请同时提供图片和蒙版")

            first_tensor = image_list[0]
            if first_tensor.dim() == 3:
                first_tensor = first_tensor.unsqueeze(0)
            image_size = (first_tensor.shape[1], first_tensor.shape[2])
            mask_key = "mask"
            assets.append({
                "key": mask_key,
                "label": "蒙版",
                "bytes": self._mask_tensor_to_rgba_png_bytes(mask_tensor, image_size),
            })

        def _make_body(asset_bytes: Dict[str, bytes]) -> dict:
            body = {
                "model": api_model,
                "prompt": prompt,
                "images": [
                    self._png_bytes_to_data_url(asset_bytes[key])
                    for key in image_keys
                    if key in asset_bytes
                ],
                "size": size if size else "auto",
                "quality": quality,
                "n": int(n),
                "output_format": output_format or "png",
            }
            if mask_key and mask_key in asset_bytes:
                body["mask"] = {
                    "image_url": self._png_bytes_to_data_url(asset_bytes[mask_key])
                }
            return body

        original_asset_bytes = {asset["key"]: asset["bytes"] for asset in assets}
        original_body = _make_body(original_asset_bytes)
        self._log_original_request_body("async generateImage", original_body)

        asset_bytes = self._fit_png_assets_to_body_limit(assets, _make_body)
        body = _make_body(asset_bytes)
        body_size = self._json_body_size(body)
        if body_size > self._MAX_BODY_BYTES:
            raise RuntimeError(f"请求体超过 20MB：{body_size // 1024}KB")

        return body

    async def _submit_generate_image_task(
        self,
        session: aiohttp.ClientSession,
        payload: dict,
    ) -> str:
        url = f"{self.base_url}{_ENDPOINT_ASYNC_GENERATE}"
        last_status = None

        for attempt in range(DEFAULT_MAX_RETRIES + 1):
            t0 = time.time()
            async with session.post(
                url,
                json=payload,
                headers=self._json_headers(),
            ) as resp:
                elapsed = time.time() - t0
                text = await resp.text()
                self._log_original_response_body(
                    f"submit generateImage status={resp.status}",
                    text,
                )

                if resp.status not in (200, 201, 202):
                    last_status = resp.status
                    if resp.status in RETRYABLE_STATUS_CODES and attempt < DEFAULT_MAX_RETRIES:
                        friendly = get_friendly_message(resp.status)
                        delay = _compute_delay(
                            attempt,
                            DEFAULT_BASE_DELAY,
                            DEFAULT_MAX_DELAY,
                            DEFAULT_BACKOFF_FACTOR,
                        )
                        print(f"[o1key GPT Image] {friendly} {delay:.1f}s 后重试提交 ({attempt+1}/{DEFAULT_MAX_RETRIES})...")
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(self._extract_error_message(text, resp.status))

                try:
                    data = json.loads(text)
                except Exception:
                    raise RuntimeError(f"提交响应 JSON 解析失败，原始内容：{text[:500]}") from None

                task_id = data.get("task_id")
                if not task_id:
                    raise RuntimeError(f"提交响应中未找到 task_id: {data}")

                status = data.get("status", "")
                print(f"[o1key GPT Image] 异步任务已提交 | task_id={task_id} | status={status} | 耗时 {elapsed:.1f}s")
                return task_id

        if last_status and last_status in HTTP_ERROR_MESSAGES:
            raise RuntimeError(HTTP_ERROR_MESSAGES[last_status])
        raise RuntimeError(f"异步任务提交失败: 重试 {DEFAULT_MAX_RETRIES} 次后仍然失败")

    async def _poll_generate_image_task(
        self,
        session: aiohttp.ClientSession,
        task_id: str,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> dict:
        url = f"{self.base_url}{_ENDPOINT_ASYNC_TASK.format(task_id=task_id)}"
        start_time = time.time()

        poll_count = 0
        last_poll_at = start_time

        while True:
            if poll_count < len(_ASYNC_POLL_SCHEDULE):
                next_poll_at = start_time + _ASYNC_POLL_SCHEDULE[poll_count]
            else:
                next_poll_at = last_poll_at + _ASYNC_POLL_INTERVAL

            sleep_time = next_poll_at - time.time()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            last_poll_at = time.time()
            elapsed = last_poll_at - start_time
            if elapsed > _ASYNC_MAX_WAIT:
                raise RuntimeError(f"任务 {task_id} 超时（>{int(_ASYNC_MAX_WAIT)}秒），请稍后用 task_id 查询结果")

            poll_count += 1
            async with session.get(url, headers=self._auth_headers()) as resp:
                text = await resp.text()
                self._log_original_response_body(
                    f"poll task status={resp.status}",
                    text,
                )
                if resp.status != 200:
                    raise RuntimeError(self._extract_error_message(text, resp.status))

                try:
                    task = json.loads(text)
                except Exception:
                    raise RuntimeError(f"任务查询响应 JSON 解析失败，原始内容：{text[:500]}") from None

            status = task.get("status", "UNKNOWN")
            progress = task.get("progress")
            progress_pct = self._coerce_progress_percent(progress)
            if POLL_LOG_ENABLED:
                progress_text = f" | progress={progress}" if progress is not None else ""
                print(f"[o1key GPT Image] 查询任务 #{poll_count} | task_id={task_id} | status={status}{progress_text}")

            if status == "SUCCESS":
                self._emit_progress(progress_callback, 100)
                return task
            if progress_pct is not None and progress_pct < 100:
                self._emit_progress(progress_callback, progress_pct)
            if status == "FAILURE":
                error_message = self._extract_error_message(task, 500) or "生成失败"
                raise _AsyncImageTaskFailure(
                    error_message,
                    self._extract_error_detail(task),
                )
            if status not in ("SUBMITTED", "IN_PROGRESS"):
                raise RuntimeError(f"未知任务状态 {status}: {task}")

    async def _parse_async_task_images(
        self,
        task: dict,
        session: aiohttp.ClientSession,
    ) -> List[Image.Image]:
        data = task.get("data", {})
        image_items = data.get("images") if isinstance(data, dict) else None

        if not isinstance(image_items, list) or not image_items:
            raise RuntimeError(f"任务结果中未找到 data.images: {task}")

        images: List[Image.Image] = []
        for idx, item in enumerate(image_items, 1):
            if not isinstance(item, dict):
                continue

            url = item.get("url") or item.get("image_url")
            b64 = item.get("b64_json", "")

            if url and isinstance(url, str) and url.startswith("data:image"):
                try:
                    _, b64_data = url.split(",", 1)
                    img = Image.open(BytesIO(base64.b64decode(b64_data)))
                    images.append(img)
                    print(f"[o1key GPT Image] 第 {idx} 张 data URL 解码完成 ({img.size[0]}×{img.size[1]})")
                except Exception as e:
                    raise RuntimeError(f"第 {idx} 张 data URL 解码失败: {e}") from None
            elif url and isinstance(url, str) and url.startswith("http"):
                async with session.get(url, allow_redirects=True) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"图像下载失败 HTTP {resp.status}，URL: {url}")
                    img_bytes = await resp.read()
                img = Image.open(BytesIO(img_bytes))
                images.append(img)
                print(f"[o1key GPT Image] 第 {idx} 张 URL 下载完成 ({img.size[0]}×{img.size[1]}) | {url}")
            elif b64:
                img = self._decode_b64_image(b64, f"第 {idx} 张")
                images.append(img)
            else:
                print(f"[o1key GPT Image] 警告：第 {idx} 条结果既无 url 也无 b64_json，已跳过")

        if not images:
            raise RuntimeError("任务成功但没有可用图片结果")

        return images

    async def _generate_image_task_async(
        self,
        prompt: str,
        model: str,
        quality: str,
        size: str,
        n: int,
        seed: int,
        image_tensor: Optional[List[torch.Tensor]] = None,
        mask_tensor: Optional[torch.Tensor] = None,
        output_format: str = "png",
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Image.Image]:
        body = self._build_async_generate_body(
            prompt=prompt,
            model=model,
            quality=quality,
            size=size,
            n=n,
            image_list=image_tensor,
            mask_tensor=mask_tensor,
            output_format=output_format,
        )

        mode = "图像编辑" if mask_tensor is not None else ("图生图" if image_tensor else "文生图")
        body_size = self._json_body_size(body)
        print(
            f"[o1key GPT Image] {mode} | 新异步接口 | 模型={model} | "
            f"quality={quality} | size={size} | n={n} | body={body_size // 1024}KB"
        )

        connector = aiohttp.TCPConnector(ssl=False, force_close=True)
        timeout = aiohttp.ClientTimeout(total=_ASYNC_MAX_WAIT + 120)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            last_error = None
            for retry_index in range(len(_ASYNC_RETRY_DELAYS) + 1):
                try:
                    task_id = await self._submit_generate_image_task(session, body)
                    task = await self._poll_generate_image_task(
                        session,
                        task_id,
                        progress_callback=progress_callback,
                    )
                    return await self._parse_async_task_images(task, session)
                except _AsyncImageTaskFailure as error:
                    last_error = error
                    detail = error.error_detail
                    if (
                        retry_index < len(_ASYNC_RETRY_DELAYS)
                        and self._should_retry_async_failure(detail, retry_index)
                    ):
                        delay = _ASYNC_RETRY_DELAYS[retry_index]
                        code = detail.get("code", "unknown")
                        category = detail.get("category", "unknown")
                        failed_task_id = detail.get("task_id", "")
                        task_text = f" | failed_task_id={failed_task_id}" if failed_task_id else ""
                        print(
                            f"[o1key GPT Image] 任务失败但可重试 | code={code} | "
                            f"category={category}{task_text} | {delay:.0f}s 后重试 "
                            f"({retry_index + 1}/{len(_ASYNC_RETRY_DELAYS)})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(str(error)) from None

            if last_error is not None:
                raise RuntimeError(str(last_error)) from None
            raise RuntimeError("生成失败")

    # ── 文生图 / 图生图（generations 接口）───────────────────────────────────

    async def _generate_async(
        self,
        prompt: str,
        model: str,
        quality: str,
        size: str,
        n: int,
        seed: int,
        image_list: Optional[List[torch.Tensor]] = None,
    ) -> List[Image.Image]:
        """
        调用 /v1/images/generations/ 接口。
        当传入 image_list 时，以 data URI 格式内联图像（图生图）。
        """
        # 模型名映射：UI 显示名 → API 参数名
        api_model = _MODEL_NAME_MAP.get(model, model)

        body: dict = {
            "model":      api_model,
            "prompt":     prompt,
            "quality":    quality,
            "n":          n,
            "moderation": "low",
            "partial_images": 0,
        }

        body["size"] = size if size else "auto"

        image_files = []

        # 图生图：multipart 方式上传参考图
        if image_list is not None:
            for idx_img, img_tensor in enumerate(image_list):
                pil_images = tensor_to_pil(img_tensor)
                img = pil_images[0]
                buf = BytesIO()
                img.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                # 单张图像预算：20MB 按图数平摊，至少保留 1MB 给其他字段
                per_image_budget = max(
                    1024 * 1024,
                    (self._MAX_BODY_BYTES - 1024 * 1024) // len(image_list),
                )
                # base64 膨胀约 4/3，所以 PNG 目标上限 = budget * 3/4
                png_budget = int(per_image_budget * 3 / 4)
                label = f"第{idx_img + 1}张" if len(image_list) > 1 else ""
                png_bytes = self._shrink_png_to_limit(png_bytes, png_budget, label)
                image_files.append(png_bytes)
            mode = f"图生图（参考图 {len(image_files)} 张）"
        else:
            mode = "文生图"

        url = f"{self.base_url}{_ENDPOINT_GENERATIONS}"
        print(f"[o1key GPT Image] {mode} | 模型={model} | quality={quality} | "
              f"size={size} | n={n}")

        connector = aiohttp.TCPConnector(ssl=False, force_close=True)
        timeout   = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)

        def _build_multipart_form() -> aiohttp.FormData:
            form = self._new_multipart_form()
            self._add_form_fields(form, body)
            image_field = "image[]" if len(image_files) > 1 else "image"
            for idx_img, png_bytes in enumerate(image_files):
                form.add_field(
                    image_field,
                    png_bytes,
                    filename=f"image_{idx_img + 1}.png",
                    content_type="image/png",
                )
            return form

        async def _do_request():
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                last_status = None
                for attempt in range(DEFAULT_MAX_RETRIES + 1):
                    t0 = time.time()
                    async with session.post(
                        url,
                        data=_build_multipart_form(),
                        headers=self._auth_headers(),
                    ) as resp:
                        elapsed = time.time() - t0

                        if resp.status != 200:
                            last_status = resp.status
                            text = await resp.text()
                            if resp.status in RETRYABLE_STATUS_CODES and attempt < DEFAULT_MAX_RETRIES:
                                friendly = get_friendly_message(resp.status)
                                delay = _compute_delay(attempt, DEFAULT_BASE_DELAY, DEFAULT_MAX_DELAY, DEFAULT_BACKOFF_FACTOR)
                                print(f"[o1key GPT Image] {friendly} {delay:.1f}s 后重试 ({attempt+1}/{DEFAULT_MAX_RETRIES})...")
                                await asyncio.sleep(delay)
                                continue
                            if resp.status in HTTP_ERROR_MESSAGES:
                                raise RuntimeError(HTTP_ERROR_MESSAGES[resp.status])
                            try:
                                err_json = json.loads(text)
                                err_obj  = err_json.get("error", {})
                                msg = (
                                    err_obj.get("message") or err_obj.get("msg") or text
                                    if isinstance(err_obj, dict)
                                    else str(err_obj) or text
                                )
                            except Exception:
                                msg = text
                            raise RuntimeError(get_friendly_message(resp.status, msg))

                        print(f"[o1key GPT Image] API 响应耗时 {elapsed:.1f}s")
                        return await self._parse_success_response(resp, session, "GENERATIONS")

                if last_status and last_status in HTTP_ERROR_MESSAGES:
                    raise RuntimeError(HTTP_ERROR_MESSAGES[last_status])
                raise RuntimeError(f"请求失败: 重试 {DEFAULT_MAX_RETRIES} 次后仍然失败")

        return await self._run_with_interrupt(_do_request())

    # ── 图像编辑（edits 接口，multipart/form-data）──────────────────────

    async def _edit_async(
        self,
        prompt: str,
        model: str,
        quality: str,
        size: str,
        n: int,
        seed: int,
        image_list: List[torch.Tensor],
        mask_tensor: Optional[torch.Tensor] = None,
    ) -> List[Image.Image]:
        """
        调用 /v1/images/edits 接口（multipart/form-data）。
        """
        # 模型名映射：UI 显示名 → API 参数名
        api_model = _MODEL_NAME_MAP.get(model, model)

        # 统一 tensors 为 [1,H,W,C] 格式，支持不同尺寸
        normalized_tensors = []
        for t in image_list:
            if t.dim() == 3:
                t = t.unsqueeze(0)   # [H,W,C] → [1,H,W,C]
            normalized_tensors.append(t)
        num_images = len(normalized_tensors)

        image_files = []

        # 多图：用 multipart image/image[] 字段逐张上传
        # 预算：20MB 按图数平摊，蒙版预留 1MB
        mask_reserve = 1024 * 1024 if mask_tensor is not None else 0
        per_image_budget = max(
            1024 * 1024,
            (self._MAX_BODY_BYTES - mask_reserve) // num_images,
        )
        for i, frame in enumerate(normalized_tensors):
            img_bytes = self._tensor_to_png_bytes(frame)
            label = f"第{i + 1}张" if num_images > 1 else ""
            img_bytes = self._shrink_png_to_limit(img_bytes, per_image_budget, label)
            image_files.append(img_bytes)

        # 蒙版尺寸校验以第一张图为基准
        first_tensor = normalized_tensors[0]
        ih, iw = first_tensor.shape[1], first_tensor.shape[2]

        mask_png = None
        if mask_tensor is not None:
            mask_png = self._mask_tensor_to_rgba_png_bytes(mask_tensor, (ih, iw))
            mode = "图像编辑（带蒙版）"
        else:
            mode = "图像编辑（无蒙版）"

        form_fields = {
            "model": api_model,
            "prompt": prompt,
            "partial_images": 0,
            "n": n,
            "quality": quality,
            "size": size if size else "auto",
            "output_format": "png",
            "background": "opaque",
            "moderation": "low",
        }

        def _build_multipart_form() -> aiohttp.FormData:
            form = self._new_multipart_form()
            self._add_form_fields(form, form_fields)

            for idx_img, img_bytes in enumerate(image_files):
                form.add_field(
                    "image[]",
                    img_bytes,
                    filename=f"image_{idx_img + 1}.png",
                    content_type="image/png",
                )
            if mask_png is not None:
                form.add_field(
                    "mask",
                    mask_png,
                    filename="mask.png",
                    content_type="image/png",
                )
            return form

        url = f"{self.base_url}{_ENDPOINT_EDITS}"
        print(f"[o1key GPT Image] {mode} | 模型={model} | 参考图={num_images}张 | "
              f"quality={quality} | size={size} | n={n}")

        connector = aiohttp.TCPConnector(ssl=False, force_close=True)
        timeout   = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)

        async def _do_request():
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                last_status = None
                for attempt in range(DEFAULT_MAX_RETRIES + 1):
                    t0 = time.time()
                    async with session.post(
                        url,
                        data=_build_multipart_form(),
                        headers=self._auth_headers(),
                    ) as resp:
                        elapsed = time.time() - t0

                        if resp.status != 200:
                            text = await resp.text()
                            last_status = resp.status
                            if resp.status in RETRYABLE_STATUS_CODES and attempt < DEFAULT_MAX_RETRIES:
                                friendly = get_friendly_message(resp.status)
                                delay = _compute_delay(attempt, DEFAULT_BASE_DELAY, DEFAULT_MAX_DELAY, DEFAULT_BACKOFF_FACTOR)
                                print(f"[o1key GPT Image] {friendly} retrying in {delay:.1f}s ({attempt+1}/{DEFAULT_MAX_RETRIES})...")
                                await asyncio.sleep(delay)
                                continue
                            if resp.status in HTTP_ERROR_MESSAGES:
                                raise RuntimeError(HTTP_ERROR_MESSAGES[resp.status])
                            try:
                                err_json = json.loads(text)
                                err_obj  = err_json.get("error", {})
                                msg = (
                                    err_obj.get("message") or err_obj.get("msg") or text
                                    if isinstance(err_obj, dict)
                                    else str(err_obj) or text
                                )
                            except Exception:
                                msg = text
                            raise RuntimeError(get_friendly_message(resp.status, msg))

                        print(f"[o1key GPT Image] API 响应耗时 {elapsed:.1f}s")
                        return await self._parse_success_response(resp, session, "EDITS")

                if last_status and last_status in HTTP_ERROR_MESSAGES:
                    raise RuntimeError(HTTP_ERROR_MESSAGES[last_status])
                raise RuntimeError(f"Request failed after {DEFAULT_MAX_RETRIES} retries")

        return await self._run_with_interrupt(_do_request())

    # ── 同步统一入口（供节点调用）────────────────────────────────────────────

    def run_sync(
        self,
        prompt: str,
        model: str,
        quality: str,
        size: str,
        n: int,
        seed: int,
        image_tensor: Optional[List[torch.Tensor]] = None,
        mask_tensor: Optional[torch.Tensor] = None,
    ) -> List[Image.Image]:
        """
        同步入口，在独立线程中运行事件循环，避免与 ComfyUI 主循环冲突。

        路由逻辑：
          - 无 image_tensor  → generations 接口（文生图，multipart/form-data）
          - 有 image_tensor  → edits 接口（图生图/编辑，multipart/form-data）
        """
        use_edits = (image_tensor is not None)

        if use_edits:
            coro = self._edit_async(
                prompt=prompt, model=model, quality=quality,
                size=size, n=n, seed=seed,
                image_list=image_tensor, mask_tensor=mask_tensor,
            )
        else:
            coro = self._generate_async(
                prompt=prompt, model=model, quality=quality,
                size=size, n=n, seed=seed,
                image_list=image_tensor,
            )

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            try:
                return future.result(timeout=_REQUEST_TIMEOUT + 30)
            except TimeoutError:
                raise RuntimeError(
                    f"o1key GPT Image 请求超时（>{_REQUEST_TIMEOUT}s），请检查网络或稍后重试"
                )

    def generate_image_async_sync(
        self,
        prompt: str,
        model: str,
        quality: str,
        size: str,
        n: int,
        seed: int,
        image_tensor: Optional[List[torch.Tensor]] = None,
        mask_tensor: Optional[torch.Tensor] = None,
        output_format: str = "png",
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Image.Image]:
        """
        新版异步任务入口，供节点调用。
        旧 run_sync 保留兼容，但 GPT Image 节点不再使用旧同步接口。
        """
        coro = self._generate_image_task_async(
            prompt=prompt,
            model=model,
            quality=quality,
            size=size,
            n=n,
            seed=seed,
            image_tensor=image_tensor,
            mask_tensor=mask_tensor,
            output_format=output_format,
            progress_callback=progress_callback,
        )

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._run_with_interrupt(coro))
            finally:
                loop.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            try:
                return future.result(timeout=_ASYNC_MAX_WAIT + 150)
            except TimeoutError:
                raise RuntimeError(
                    f"o1key GPT Image 异步任务超时（>{int(_ASYNC_MAX_WAIT)}秒），请稍后用 task_id 查询结果"
                )

    # ── 余额查询 ──────────────────────────────────────────────────────────────

    async def _query_balance_async(self) -> dict:
        url = f"{self.base_url}/api/usage/token"
        connector = aiohttp.TCPConnector(ssl=False, force_close=True)
        timeout   = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.get(url, headers=self._auth_headers()) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"余额查询失败 HTTP {resp.status}")
                return await resp.json()

    def query_balance_sync(self) -> dict:
        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._query_balance_async())
            finally:
                loop.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_run).result(timeout=15)

    @staticmethod
    def format_balance_info(balance_data: dict) -> str:
        data             = balance_data.get("data", {})
        api_name         = data.get("name", "未知")
        total_available  = data.get("total_available", 0)
        balance_in_dollars = total_available / 500000
        return f"当前余额：{balance_in_dollars:.2f} | API：{api_name}"
