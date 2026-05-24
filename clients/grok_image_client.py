"""
Grok Image API 客户端
支持两个接口：
  - POST /v1/images/generations  文生图
  - POST /v1/images/edits        图生图（带参考图）

上游 API 格式与 OpenAI Images API 兼容。
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

from ..utils.config import get_api_key_or_raise, get_base_url_by_route
from ..utils.image_utils import tensor_to_pil

try:
    from comfy.model_management import processing_interrupted, InterruptProcessingException
    _INTERRUPT_AVAILABLE = True
except ImportError:
    _INTERRUPT_AVAILABLE = False
    InterruptProcessingException = RuntimeError
    processing_interrupted = lambda: False

_ENDPOINT_GENERATIONS = "/v1/images/generations"
_ENDPOINT_EDITS = "/v1/images/edits"

_MODEL_NAME_MAP = {
    "Grok Image": "grok-imagine-image",
    "Grok Image Pro": "grok-imagine-image-quality",
}

_REQUEST_TIMEOUT = 900
_MAX_BODY_BYTES = 20 * 1024 * 1024

_MAX_RETRIES = 3
_RETRY_DELAY = 5


class GrokImageClient:

    def __init__(self, route: str = "全球加速"):
        self.api_key = get_api_key_or_raise("O1KEY_API_KEY")
        self.base_url = get_base_url_by_route(route)

    def _json_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    # ── 图像工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _shrink_png_to_limit(png_bytes: bytes, max_bytes: int, label: str = "") -> bytes:
        if len(png_bytes) <= max_bytes:
            return png_bytes
        img = Image.open(BytesIO(png_bytes))
        w, h = img.size
        original_size = len(png_bytes)
        step = 0
        while len(png_bytes) > max_bytes:
            scale = 0.894
            w = max(1, int(w * scale))
            h = max(1, int(h * scale))
            img = img.resize((w, h), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            step += 1
        tag = f" ({label})" if label else ""
        print(
            f"[o1key Grok Image] 图像{tag}超出 {max_bytes // (1024*1024)}MB 限制，"
            f"已等比缩放 {step} 次：{original_size // 1024}KB → {len(png_bytes) // 1024}KB "
            f"（{w}×{h}）"
        )
        return png_bytes

    @staticmethod
    def _pil_list_to_tensor(images: List[Image.Image]) -> torch.Tensor:
        if not images:
            placeholder = Image.new("RGB", (512, 512), (128, 128, 128))
            images = [placeholder]
        tensors = []
        for img in images:
            arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))
        return torch.stack(tensors, dim=0)

    # ── 中断轮询 ──────────────────────────────────────────────────────────────

    @staticmethod
    async def _poll_interrupt():
        while True:
            await asyncio.sleep(0.5)
            if _INTERRUPT_AVAILABLE and processing_interrupted():
                return

    @staticmethod
    async def _run_with_interrupt(coro):
        if not _INTERRUPT_AVAILABLE:
            return await coro
        request_task = asyncio.ensure_future(coro)
        interrupt_task = asyncio.ensure_future(GrokImageClient._poll_interrupt())
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

    # ── 响应解析 ──────────────────────────────────────────────────────────────

    async def _parse_response(self, resp_json: dict, session: aiohttp.ClientSession) -> List[Image.Image]:
        if "error" in resp_json:
            err = resp_json["error"]
            msg = (
                err.get("message") or err.get("msg") or json.dumps(err, ensure_ascii=False)
                if isinstance(err, dict) else str(err)
            )
            raise RuntimeError(f"API 返回错误: {msg}")
        data_list = resp_json.get("data")
        if not data_list:
            raise RuntimeError(f"API 响应中未找到 data 字段")
        images: List[Image.Image] = []
        for idx, item in enumerate(data_list):
            b64 = item.get("b64_json", "")
            url = item.get("url", "")
            if b64:
                img_bytes = base64.b64decode(b64)
                img = Image.open(BytesIO(img_bytes))
                images.append(img)
            elif url and url.startswith("http"):
                async with session.get(url, allow_redirects=True) as r:
                    if r.status != 200:
                        raise RuntimeError(f"图像下载失败 HTTP {r.status}")
                    img_bytes = await r.read()
                images.append(Image.open(BytesIO(img_bytes)))
            else:
                print(f"[o1key Grok Image] 警告：第 {idx + 1} 条数据无有效图像，已跳过")
        return images

    # ── 文生图（generations 接口）─────────────────────────────────────────────

    async def _generate_async(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str,
        resolution: str,
        n: int,
    ) -> List[Image.Image]:
        api_model = _MODEL_NAME_MAP.get(model, model)
        body: dict = {
            "model": api_model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio if aspect_ratio else "auto",
            "resolution": resolution if resolution else "1k",
            "response_format": "b64_json",
        }

        url = f"{self.base_url}{_ENDPOINT_GENERATIONS}"
        log_body = {k: v for k, v in body.items()}
        print(f"[o1key Grok Image] 请求 URL: {url}")
        print(f"[o1key Grok Image] 请求体: {json.dumps(log_body, ensure_ascii=False)}")

        results = []
        for i in range(n):
            images = await self._do_request_with_retry(url, body)
            results.extend(images)
            if n > 1:
                print(f"[o1key Grok Image] 第 {i+1}/{n} 张完成")
        return results

    # ── 图生图（edits 接口）───────────────────────────────────────────────────

    async def _edit_async(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str,
        resolution: str,
        n: int,
        image_list: List[torch.Tensor],
    ) -> List[Image.Image]:
        api_model = _MODEL_NAME_MAP.get(model, model)
        body: dict = {
            "model": api_model,
            "prompt": prompt,
            "response_format": "b64_json",
        }
        if aspect_ratio and aspect_ratio != "auto":
            body["aspect_ratio"] = aspect_ratio
        if resolution:
            body["resolution"] = resolution

        # 参考图转 base64 字符串
        pil_images = tensor_to_pil(image_list[0])
        img = pil_images[0]
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        png_bytes = self._shrink_png_to_limit(png_bytes, _MAX_BODY_BYTES // 2)
        body["image"] = base64.b64encode(png_bytes).decode("utf-8")

        url = f"{self.base_url}{_ENDPOINT_EDITS}"
        log_body = {k: (v[:50] + "..." if k == "image" and len(v) > 50 else v) for k, v in body.items()}
        print(f"[o1key Grok Image] 请求 URL: {url}")
        print(f"[o1key Grok Image] 请求体: {json.dumps(log_body, ensure_ascii=False)}")

        results = []
        for i in range(n):
            images = await self._do_request_with_retry(url, body)
            results.extend(images)
            if n > 1:
                print(f"[o1key Grok Image] 第 {i+1}/{n} 张完成")
        return results

    # ── 带重试的请求 ────────────────────────────────────────────────────────

    async def _do_request_with_retry(self, url: str, body: dict) -> List[Image.Image]:
        connector = aiohttp.TCPConnector(ssl=False, force_close=True)
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)

        async def _do_request():
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                last_error = None
                for attempt in range(1, _MAX_RETRIES + 1):
                    t0 = time.time()
                    async with session.post(url, json=body, headers=self._json_headers()) as resp:
                        elapsed = time.time() - t0
                        text = await resp.text()

                        if resp.status == 429 or resp.status in (502, 503, 504):
                            last_error = f"HTTP {resp.status}"
                            print(f"[o1key Grok Image] 重试 {attempt}/{_MAX_RETRIES}（{last_error}）")
                            await asyncio.sleep(_RETRY_DELAY * attempt)
                            continue

                        if resp.status == 400 and "high load" in text.lower():
                            last_error = "high load"
                            print(f"[o1key Grok Image] 重试 {attempt}/{_MAX_RETRIES}（服务繁忙）")
                            await asyncio.sleep(_RETRY_DELAY * attempt)
                            continue

                        if resp.status != 200:
                            try:
                                err_json = json.loads(text)
                                err_obj = err_json.get("error", {})
                                msg = (
                                    err_obj.get("message") or err_obj.get("msg") or text
                                    if isinstance(err_obj, dict) else str(err_obj) or text
                                )
                            except Exception:
                                msg = text
                            raise RuntimeError(f"请求失败 HTTP {resp.status}: {msg}")

                        try:
                            resp_json = json.loads(text)
                        except Exception:
                            raise RuntimeError(f"响应 JSON 解析失败，原始内容：{text[:500]}")

                    print(f"[o1key Grok Image] API 响应耗时 {elapsed:.1f}s")
                    return await self._parse_response(resp_json, session)

                raise RuntimeError(f"重试 {_MAX_RETRIES} 次后仍失败: {last_error}")

        return await self._run_with_interrupt(_do_request())

    # ── 同步入口 ──────────────────────────────────────────────────────────────

    def run_sync(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str,
        resolution: str,
        n: int,
        image_list: Optional[List[torch.Tensor]] = None,
    ) -> List[Image.Image]:
        if image_list:
            coro = self._edit_async(
                prompt=prompt, model=model, aspect_ratio=aspect_ratio,
                resolution=resolution, n=n, image_list=image_list,
            )
        else:
            coro = self._generate_async(
                prompt=prompt, model=model, aspect_ratio=aspect_ratio,
                resolution=resolution, n=n,
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
                raise RuntimeError("Grok Image 请求超时，请检查网络或稍后重试")

    # ── 余额查询 ──────────────────────────────────────────────────────────────

    async def _query_balance_async(self) -> dict:
        url = f"{self.base_url}/api/usage/token"
        connector = aiohttp.TCPConnector(ssl=False, force_close=True)
        timeout = aiohttp.ClientTimeout(total=10)
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
        data = balance_data.get("data", {})
        api_name = data.get("name", "未知")
        total_available = data.get("total_available", 0)
        balance_in_dollars = total_available / 500000
        return f"当前余额：{balance_in_dollars:.2f} | API：{api_name}"
