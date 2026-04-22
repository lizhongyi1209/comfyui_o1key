"""
GPT Image API 客户端
支持两个接口：
  - POST /v1/images/generations/  文生图 / 图生图（gpt-image-1 / gpt-image-1.5）
  - POST /v1/images/edits/        图像编辑（带蒙版 inpainting）

设计原则：
  - 与 doubao_image_client.py 保持相同的异步 + 同步双入口模式
  - 图像以 multipart/form-data 方式上传（edits 接口）
  - generations 接口使用 JSON 请求体，图像以 data URI base64 内联传递
  - 响应支持 url 和 b64_json 两种格式，优先处理 b64_json（避免二次下载）
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

# ── 接口端点 ──────────────────────────────────────────────────────────────────
_ENDPOINT_GENERATIONS = "/v1/images/generations/"
_ENDPOINT_EDITS       = "/v1/images/edits/"

# ── 模型名映射（UI 显示名 → API 实际参数名）─────────────────────────────────
_MODEL_NAME_MAP = {
    "gpt-image-1-特价":   "gpt-image-1-special",
    "gpt-image-1.5-特价": "gpt-image-1.5-special",
    "gpt-image-2-特价":   "gpt-image-2-special",
}

# ── 超时 ──────────────────────────────────────────────────────────────────────
_REQUEST_TIMEOUT = 900   # 秒


class GptImageClient:
    """
    GPT Image API 客户端

    接口说明：
      generations：JSON body，支持 background / quality / size / n / model
      edits：multipart/form-data，必须包含 image（PNG），可选 mask（PNG）

    两个接口的响应格式相同：
      { "data": [ {"url": "..."} | {"b64_json": "..."} ] }
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

    # ── 图像转换工具 ──────────────────────────────────────────────────────────

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

        return torch.stack(tensors, dim=0)          # [B, H, W, 4]

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

    # ── 文生图 / 图生图（generations 接口）───────────────────────────────────

    async def _generate_async(
        self,
        prompt: str,
        model: str,
        quality: str,
        background: str,
        size: str,
        n: int,
        seed: int,
        image_tensor: Optional[torch.Tensor] = None,
    ) -> List[Image.Image]:
        """
        调用 /v1/images/generations/ 接口。
        当传入 image_tensor 时，以 data URI 格式内联图像（图生图）。
        """
        # 模型名映射：UI 显示名 → API 参数名
        api_model = _MODEL_NAME_MAP.get(model, model)

        body: dict = {
            "model":      api_model,
            "prompt":     prompt,
            "quality":    quality,
            "background": background,
            "n":          n,
            "moderation": "low",
        }

        # size = "auto" 时不传该字段，让 API 自行决定
        if size and size != "auto":
            body["size"] = size

        # seed > 0 时才传递（0 视为不指定）
        if seed > 0:
            body["seed"] = seed

        # 图生图：将 tensor 转成 data URI 内联
        if image_tensor is not None:
            pil_images = tensor_to_pil(image_tensor)
            data_urls = []
            for img in pil_images:
                b64 = encode_image_to_base64(img, format="PNG")
                data_urls.append(f"data:image/png;base64,{b64}")
            body["image"] = data_urls[0] if len(data_urls) == 1 else data_urls
            mode = f"图生图（参考图 {len(data_urls)} 张）"
        else:
            mode = "文生图"

        url = f"{self.base_url}{_ENDPOINT_GENERATIONS}"
        print(f"[o1key GPT Image] {mode} | 模型={model} | quality={quality} | "
              f"background={background} | size={size} | n={n}")

        connector = aiohttp.TCPConnector(ssl=False, force_close=True)
        timeout   = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            t0 = time.time()
            async with session.post(url, json=body, headers=self._json_headers()) as resp:
                elapsed = time.time() - t0
                text = await resp.text()

                if resp.status != 200:
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
                    raise RuntimeError(f"请求失败 HTTP {resp.status}: {msg}")

                try:
                    resp_json = json.loads(text)
                except Exception:
                    raise RuntimeError(f"响应 JSON 解析失败，原始内容：{text[:500]}")

            print(f"[o1key GPT Image] API 响应耗时 {elapsed:.1f}s")
            return await self._parse_response(resp_json, session)

    # ── 图像编辑（edits 接口，multipart/form-data）──────────────────────────
    # 注意：o1key 中转服务的 edits 接口暂不支持 quality / background / moderation 参数，
    # 这些字段暂时不传递，待服务方更新后可恢复。

    async def _edit_async(
        self,
        prompt: str,
        model: str,
        quality: str,
        background: str,
        size: str,
        n: int,
        seed: int,
        image_tensor: torch.Tensor,
        mask_tensor: Optional[torch.Tensor] = None,
    ) -> List[Image.Image]:
        """
        调用 /v1/images/edits/ 接口（multipart/form-data）。
        当前仅传递 model / prompt / n / size / image / mask，
        quality / background / moderation 暂不支持（o1key 服务端限制）。
        """
        # 模型名映射：UI 显示名 → API 参数名
        api_model = _MODEL_NAME_MAP.get(model, model)

        # 将 batch tensor 拆成逐帧列表
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)   # [H,W,C] → [1,H,W,C]
        num_images = image_tensor.shape[0]

        # o1key 中转服务的 edits 接口暂不支持 quality / background / moderation / seed，
        # 待服务方更新后可重新加入。
        form = aiohttp.FormData()
        form.add_field("model",  api_model)
        form.add_field("prompt", prompt)
        form.add_field("n",      str(n))

        if size and size != "auto":
            form.add_field("size", size)

        # 多图：用 image[] 数组字段逐张附加，支持 gpt-image-1.5 最多 16 张
        for i in range(num_images):
            frame = image_tensor[i:i+1]               # [1,H,W,C]
            img_bytes = self._tensor_to_png_bytes(frame)
            form.add_field(
                "image[]",
                img_bytes,
                filename=f"image_{i}.png",
                content_type="image/png",
            )

        ih, iw = image_tensor.shape[1], image_tensor.shape[2]

        if mask_tensor is not None:
            mask_png = self._mask_tensor_to_rgba_png_bytes(mask_tensor, (ih, iw))
            form.add_field(
                "mask",
                mask_png,
                filename="mask.png",
                content_type="image/png",
            )
            mode = "图像编辑（带蒙版）"
        else:
            mode = "图像编辑（无蒙版）"

        url = f"{self.base_url}{_ENDPOINT_EDITS}"
        print(f"[o1key GPT Image] {mode} | 模型={model} | 参考图={num_images}张 | size={size} | n={n}")

        connector = aiohttp.TCPConnector(ssl=False, force_close=True)
        timeout   = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            t0 = time.time()
            async with session.post(
                url,
                data=form,
                headers=self._auth_headers(),
            ) as resp:
                elapsed = time.time() - t0
                text = await resp.text()

                if resp.status != 200:
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
                    raise RuntimeError(f"请求失败 HTTP {resp.status}: {msg}")

                try:
                    resp_json = json.loads(text)
                except Exception:
                    raise RuntimeError(f"响应 JSON 解析失败，原始内容：{text[:500]}")

            print(f"[o1key GPT Image] API 响应耗时 {elapsed:.1f}s")
            return await self._parse_response(resp_json, session)

    # ── 同步统一入口（供节点调用）────────────────────────────────────────────

    def run_sync(
        self,
        prompt: str,
        model: str,
        quality: str,
        background: str,
        size: str,
        n: int,
        seed: int,
        image_tensor: Optional[torch.Tensor] = None,
        mask_tensor: Optional[torch.Tensor] = None,
    ) -> List[Image.Image]:
        """
        同步入口，在独立线程中运行事件循环，避免与 ComfyUI 主循环冲突。

        路由逻辑：
          - 无 image_tensor  → generations 接口（文生图，JSON body）
          - 有 image_tensor  → edits 接口（图生图/编辑，multipart/form-data）
            所有模型统一走 multipart，quality/background 通过表单字段传递，
            new-api 开启"透传请求体"后原样转发给上游。
        """
        use_edits = (image_tensor is not None)

        if use_edits:
            coro = self._edit_async(
                prompt=prompt, model=model, quality=quality,
                background=background, size=size, n=n, seed=seed,
                image_tensor=image_tensor, mask_tensor=mask_tensor,
            )
        else:
            coro = self._generate_async(
                prompt=prompt, model=model, quality=quality,
                background=background, size=size, n=n, seed=seed,
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
                    f"o1key GPT Image 请求超时（>{_REQUEST_TIMEOUT}s），请检查网络或稍后重试"
                )
