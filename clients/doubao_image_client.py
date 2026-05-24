"""
豆包生图 API 客户端
端点：POST /v1/images/generations/
兼容 new-api 透传格式（OpenAI images/generations 兼容）

设计原则：
  - 发送完整正确的请求体，new-api 丢弃字段是其侧问题
  - 响应永远是同步 JSON（new-api 强制 stream=false）
  - 图像输入以 data:image/png;base64,... 格式内联传递
"""

import asyncio
import json
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import aiohttp
from PIL import Image

from ..utils.config import get_api_key_or_raise, get_api_base_url
from ..utils.image_utils import tensor_to_pil, encode_image_to_base64
from ..utils.http_error import RETRYABLE_STATUS_CODES, HTTP_ERROR_MESSAGES, _compute_delay, DEFAULT_MAX_RETRIES, DEFAULT_BASE_DELAY, DEFAULT_MAX_DELAY, DEFAULT_BACKOFF_FACTOR


# ── 固定端点 ──────────────────────────────────────────────────────────────────
_ENDPOINT = "/v1/images/generations/"

# ── 轮询 / 请求超时 ───────────────────────────────────────────────────────────
_REQUEST_TIMEOUT = 300   # 单次请求超时秒数（豆包图像生成最长约 60s）


class DoubaoImageClient:
    """
    豆包生图客户端（new-api 原生 OpenAI 兼容格式）

    new-api 兼容性说明（基于源码分析）：
      ✅ 透传：model / prompt / size / response_format / watermark / image
      ❌ 丢弃：seed / sequential_image_generation / sequential_image_generation_options
              （进入 Extra map，但 MarshalJSON 中合并代码被注释）
      ❌ 强制：stream 硬编码 false，图像接口无流式处理
      ❌ 未实现：/v1/files 文件上传（501）

    节点仍发送完整字段，待 new-api 修复后自动生效。
    """

    def __init__(self):
        self.api_key = get_api_key_or_raise("O1KEY_API_KEY")
        self.base_url = get_api_base_url()

    # ── 认证头 ────────────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ── 图像字段构建 ──────────────────────────────────────────────────────────

    def _tensor_to_image_field(self, tensor) -> Union[str, List[str]]:
        """
        ComfyUI IMAGE tensor → API image 字段值

        单张返回字符串，多张返回字符串列表，格式：
            data:image/png;base64,<base64数据>
        """
        pil_images = tensor_to_pil(tensor)
        data_urls = []
        for img in pil_images:
            b64 = encode_image_to_base64(img, format="PNG")
            data_urls.append(f"data:image/png;base64,{b64}")

        return data_urls[0] if len(data_urls) == 1 else data_urls

    # ── 请求体构建 ────────────────────────────────────────────────────────────

    def _build_body(
        self,
        model: str,
        prompt: str,
        size: str,
        seed: int,
        sequential_image_generation: str,
        max_images: int,
        image_field=None,          # str | list[str] | None
    ) -> dict:
        """
        构建完整请求体。

        字段说明（对照官方示例）：
          - response_format: 固定 "url"（new-api 原样透传给豆包）
          - watermark: 固定 False（UI 已移除该参数）
          - stream: 固定 False（new-api 强制非流式，此字段不被读取，仅显式注明）
          - sequential_image_generation_options: 仅 sequential=auto 时发送
        """
        body = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "response_format": "url",
            "watermark": False,
            "seed": seed,
            "sequential_image_generation": sequential_image_generation,
        }

        # 仅 auto 模式才发送 max_images 选项
        if sequential_image_generation == "auto":
            body["sequential_image_generation_options"] = {
                "max_images": max_images
            }

        # 图像输入（图生图）
        if image_field is not None:
            body["image"] = image_field

        return body

    # ── 图像下载 ──────────────────────────────────────────────────────────────

    async def _download_image(
        self,
        url: str,
        session: aiohttp.ClientSession,
    ) -> Image.Image:
        """从 URL 下载图像，返回 PIL.Image。"""
        async with session.get(url, allow_redirects=True) as resp:
            if resp.status != 200:
                raise RuntimeError(
                    f"图像下载失败，HTTP {resp.status}，URL: {url}"
                )
            data = await resp.read()

        try:
            img = Image.open(BytesIO(data)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"图像解码失败: {e}")
        return img

    # ── 响应解析 ──────────────────────────────────────────────────────────────

    async def _parse_response(
        self,
        resp_json: dict,
        session: aiohttp.ClientSession,
    ) -> List[Image.Image]:
        """
        解析 /v1/images/generations 响应，返回 PIL.Image 列表。

        期望格式（new-api 原样透传豆包响应）：
        {
            "created": 1234567890,
            "data": [
                {"url": "https://..."},
                {"url": "https://..."}
            ]
        }

        兼容 b64_json 字段（豆包理论上也支持）。
        """
        # 检查 API 层级错误
        if "error" in resp_json:
            err = resp_json["error"]
            if isinstance(err, dict):
                msg = err.get("message") or err.get("msg") or json.dumps(err, ensure_ascii=False)
            else:
                msg = str(err)
            raise RuntimeError(f"API 返回错误: {msg}")

        data_list = resp_json.get("data")
        if not data_list:
            raise RuntimeError(
                f"API 响应中未找到 data 字段，完整响应：\n"
                f"{json.dumps(resp_json, ensure_ascii=False, indent=2)}"
            )

        images: List[Image.Image] = []

        for idx, item in enumerate(data_list):
            url = item.get("url", "")
            b64 = item.get("b64_json", "")

            if url and url.startswith("http"):
                # 优先使用 URL 模式
                img = await self._download_image(url, session)
                images.append(img)
                print(f"[豆包生图] 第 {idx + 1} 张下载完成 ({img.size[0]}×{img.size[1]})")

            elif b64:
                # 回退到 base64 模式
                import base64 as _b64
                try:
                    img_data = _b64.b64decode(b64)
                    img = Image.open(BytesIO(img_data)).convert("RGB")
                    images.append(img)
                    print(f"[豆包生图] 第 {idx + 1} 张 base64 解码完成 ({img.size[0]}×{img.size[1]})")
                except Exception as e:
                    raise RuntimeError(f"第 {idx + 1} 张 base64 解码失败: {e}")

            else:
                print(f"[豆包生图] 警告：第 {idx + 1} 条数据既无 url 也无 b64_json，已跳过")

        return images

    # ── 核心异步生成方法 ──────────────────────────────────────────────────────

    async def _generate_async(
        self,
        model: str,
        prompt: str,
        size: str,
        seed: int,
        sequential_image_generation: str,
        max_images: int,
        image_tensor=None,
    ) -> List[Image.Image]:
        """
        异步完整流程：构建请求 → POST → 解析 → 下载图像。
        """
        # 1. 构建 image 字段
        image_field = None
        if image_tensor is not None:
            image_field = self._tensor_to_image_field(image_tensor)
            n_imgs = len(image_field) if isinstance(image_field, list) else 1
            print(f"[豆包生图] 图生图模式，参考图 {n_imgs} 张")
        else:
            print(f"[豆包生图] 文生图模式")

        # 2. 构建请求体
        body = self._build_body(
            model=model,
            prompt=prompt,
            size=size,
            seed=seed,
            sequential_image_generation=sequential_image_generation,
            max_images=max_images,
            image_field=image_field,
        )

        url = f"{self.base_url}{_ENDPOINT}"
        print(f"[豆包生图] 提交请求 → {model} | {size}")

        connector = aiohttp.TCPConnector(ssl=False, force_close=True)
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

            # 3. 发送 POST 请求（带退避重试）
            last_status = None
            for attempt in range(DEFAULT_MAX_RETRIES + 1):
                t0 = time.time()
                async with session.post(
                    url,
                    json=body,
                    headers=self._headers(),
                ) as resp:
                    elapsed_req = time.time() - t0
                    text = await resp.text()

                    if resp.status != 200:
                        last_status = resp.status
                        if resp.status in RETRYABLE_STATUS_CODES and attempt < DEFAULT_MAX_RETRIES:
                            friendly = HTTP_ERROR_MESSAGES.get(resp.status)
                            delay = _compute_delay(attempt, DEFAULT_BASE_DELAY, DEFAULT_MAX_DELAY, DEFAULT_BACKOFF_FACTOR)
                            print(f"[豆包生图] {friendly} {delay:.1f}s 后重试 ({attempt+1}/{DEFAULT_MAX_RETRIES})...")
                            await asyncio.sleep(delay)
                            continue
                        if resp.status in HTTP_ERROR_MESSAGES:
                            raise RuntimeError(HTTP_ERROR_MESSAGES[resp.status])
                        try:
                            err_json = json.loads(text)
                            err_obj = err_json.get("error", {})
                            if isinstance(err_obj, dict):
                                msg = (
                                    err_obj.get("message")
                                    or err_obj.get("msg")
                                    or text
                                )
                            else:
                                msg = str(err_obj) or text
                        except Exception:
                            msg = text
                        raise RuntimeError(
                            f"请求失败 HTTP {resp.status}: {msg}"
                        )

                    try:
                        resp_json = json.loads(text)
                    except Exception:
                        raise RuntimeError(f"响应 JSON 解析失败，原始内容：{text[:500]}")

                break
            else:
                if last_status and last_status in HTTP_ERROR_MESSAGES:
                    raise RuntimeError(HTTP_ERROR_MESSAGES[last_status])
                raise RuntimeError(f"请求失败: 重试 {DEFAULT_MAX_RETRIES} 次后仍然失败")

            print(f"[豆包生图] API 响应耗时 {elapsed_req:.1f}s，开始下载图像...")

            # 4. 解析响应 & 下载图像（session 复用）
            images = await self._parse_response(resp_json, session)

        return images

    # ── 同步入口（供 ComfyUI 节点调用）──────────────────────────────────────

    def generate_sync(
        self,
        model: str,
        prompt: str,
        size: str,
        seed: int,
        sequential_image_generation: str,
        max_images: int,
        image_tensor=None,
    ) -> List[Image.Image]:
        """
        同步生成接口（在独立线程中运行事件循环，避免与 ComfyUI 主循环冲突）。

        Args:
            model:                       模型 ID
            prompt:                      提示词
            size:                        尺寸字符串，如 "2048x2048"
            seed:                        随机种子
            sequential_image_generation: "disabled" | "auto"
            max_images:                  最大图片数（auto 模式生效）
            image_tensor:                ComfyUI IMAGE tensor（可选，图生图用）

        Returns:
            List[PIL.Image]
        """
        coro = self._generate_async(
            model=model,
            prompt=prompt,
            size=size,
            seed=seed,
            sequential_image_generation=sequential_image_generation,
            max_images=max_images,
            image_tensor=image_tensor,
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
                    f"豆包生图超时（>{_REQUEST_TIMEOUT}s），请检查网络或稍后重试"
                )
