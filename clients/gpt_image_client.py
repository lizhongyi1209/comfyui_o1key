"""
GPT Image API 客户端
支持两个接口：
  - POST /v1/images/generations/  文生图 / 图生图（gpt-image-1 / gpt-image-1.5）
  - POST /v1/images/edits/        图像编辑（带蒙版 inpainting）

设计原则：
  - 与 doubao_image_client.py 保持相同的异步 + 同步双入口模式
  - generations / edits 接口均使用 multipart/form-data
  - 响应兼容 SSE 流式、JSON、url 和 b64_json
"""

import asyncio
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, Optional

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

# ── 模型名映射（UI 显示名 → API 实际参数名）─────────────────────────────────
_MODEL_NAME_MAP = {
    "gpt-image-2-按量": "gpt-image-2",
    "gpt-image-2-次卡": "gpt-image-2-special",
}

# ── 超时 ──────────────────────────────────────────────────────────────────────
_REQUEST_TIMEOUT = 900   # 秒


class GptImageClient:
    """
    GPT Image API 客户端

    接口说明：
      generations：multipart/form-data，支持 quality / size / n / model
      edits：multipart/form-data，图片和 mask 使用 PNG 文件上传

    响应支持 JSON 和 SSE 流式格式。
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
