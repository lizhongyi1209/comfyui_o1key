"""
全能LLM对话助手节点
ComfyUI 自定义节点，通过 OpenAI 兼容协议调用市面上主流的 AI 对话大模型
支持多模态（图片输入），单轮对话，非流式输出

API 密钥和地址通过插件统一配置（环境变量或 .config 文件），与 Google Gemini 节点一致
"""

import time
import base64
import json
from io import BytesIO
from typing import Optional, Tuple

import torch
from PIL import Image

from ..utils.image_utils import tensor_to_pil
from ..utils.config import get_api_key_or_raise, get_api_base_url

# ============================================================================
# 模型配置
# ============================================================================

SUPPORTED_MODELS = [
    "gpt-5.4",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
    "deepseek-v3.2",
    "kimi-k2.5",
    "doubao-seed-2-0-pro-260215",
    "qwen3.5-plus-2026-02-15",
    "qwen3.5-plus",
]

# 图片缩放最大尺寸
MAX_IMAGE_DIMENSION = 1568

# 图片最大文件大小（20MB）
MAX_IMAGE_SIZE = 20 * 1024 * 1024


class UniversalLLMChat:
    """
    全能LLM对话助手

    功能：
    - 通过 OpenAI 兼容协议调用主流大模型
    - 支持多模态（图片输入）
    - 单轮对话，非流式输出
    - API 密钥和地址继承插件统一配置
    """

    def __init__(self):
        self._api_key = None
        self._base_url = None

    def _ensure_config(self):
        """延迟加载配置，首次调用时初始化"""
        if self._api_key is None:
            self._api_key = get_api_key_or_raise("O1KEY_API_KEY")
            self._base_url = get_api_base_url()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": (SUPPORTED_MODELS, {
                    "default": SUPPORTED_MODELS[0]
                }),
                "提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            },
            "optional": {
                "图片": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("回复",)
    FUNCTION = "generate"
    CATEGORY = "text/generation"
    OUTPUT_NODE = True

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """如果图片过长边超过限制，等比缩放"""
        w, h = img.size
        max_dim = max(w, h)
        if max_dim > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            print(f"全能LLM: 图片缩放 {w}x{h} -> {new_w}x{new_h}")
            return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img

    def _image_to_data_url(self, img: Image.Image) -> str:
        """将 PIL Image 转为 data URL（JPEG base64）"""
        img = self._resize_image(img)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        for quality in [92, 82, 72, 60, 45]:
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=quality, optimize=True)
            data = buf.getvalue()
            if len(data) <= MAX_IMAGE_SIZE:
                b64 = base64.b64encode(data).decode('utf-8')
                return f"data:image/jpeg;base64,{b64}"

        b64 = base64.b64encode(data).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"

    def _build_messages(
        self,
        prompt: str,
        images: Optional[torch.Tensor] = None,
    ) -> list:
        """构建 OpenAI 格式的 messages 数组"""
        image_data_urls = []
        pil_images_cache = []  # 保留 PIL Image 用于总体积重新编码
        
        if images is not None:
            pil_images = tensor_to_pil(images)
            for img in pil_images:
                img_resized = self._resize_image(img)
                if img_resized.mode in ('RGBA', 'P'):
                    img_resized = img_resized.convert('RGB')
                pil_images_cache.append(img_resized)
                image_data_urls.append(self._image_to_data_url(img_resized))

        # 多图总体积控制
        if pil_images_cache and len(pil_images_cache) > 1:
            total_bytes = sum(
                len(base64.b64decode(url.split(',', 1)[1])) for url in image_data_urls
            )
            if total_bytes > MAX_IMAGE_SIZE:
                print(f"全能LLM: 图片总体积 {total_bytes / 1024 / 1024:.2f}MB 超过 {MAX_IMAGE_SIZE // 1024 // 1024}MB 限制，正在压缩...")
                
                # 降质量
                compressed = False
                for quality in [80, 70, 60, 50, 40, 30, 20]:
                    new_urls = []
                    for img in pil_images_cache:
                        buf = BytesIO()
                        img.save(buf, format='JPEG', quality=quality, optimize=True)
                        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                        new_urls.append(f"data:image/jpeg;base64,{b64}")
                    total_bytes = sum(len(base64.b64decode(u.split(',', 1)[1])) for u in new_urls)
                    if total_bytes <= MAX_IMAGE_SIZE:
                        image_data_urls = new_urls
                        print(f"全能LLM: 图片压缩完成，总体积 {total_bytes / 1024 / 1024:.2f}MB ({len(pil_images_cache)}张图片，质量{quality})")
                        compressed = True
                        break
                
                # 降分辨率
                if not compressed:
                    for scale in [0.75, 0.5, 0.35]:
                        new_urls = []
                        for img in pil_images_cache:
                            w, h = img.size
                            resized = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                            buf = BytesIO()
                            resized.save(buf, format='JPEG', quality=20, optimize=True)
                            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                            new_urls.append(f"data:image/jpeg;base64,{b64}")
                        total_bytes = sum(len(base64.b64decode(u.split(',', 1)[1])) for u in new_urls)
                        if total_bytes <= MAX_IMAGE_SIZE:
                            image_data_urls = new_urls
                            print(f"全能LLM: 图片压缩完成，总体积 {total_bytes / 1024 / 1024:.2f}MB ({len(pil_images_cache)}张图片，缩放{int(scale*100)}%)")
                            compressed = True
                            break
                
                if not compressed:
                    print(f"全能LLM: 无法将 {len(pil_images_cache)} 张图片压缩到 {MAX_IMAGE_SIZE // 1024 // 1024}MB 以内，请减少图片数量或降低分辨率")
                    raise ValueError(f"图片总体积 {total_bytes / 1024 / 1024:.2f}MB 超过限制，无法压缩到 {MAX_IMAGE_SIZE // 1024 // 1024}MB 以内")

        if not image_data_urls:
            return [{"role": "user", "content": prompt}]

        content_parts = []
        for url in image_data_urls:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": url}
            })
        content_parts.append({
            "type": "text",
            "text": prompt
        })

        return [{"role": "user", "content": content_parts}]

    def generate(
        self,
        模型: str,
        提示词: str,
        图片: Optional[torch.Tensor] = None,
    ) -> Tuple[str]:
        start_time = time.time()

        try:
            self._ensure_config()

            # 构建 messages
            messages = self._build_messages(提示词, 图片)

            img_count = len(tensor_to_pil(图片)) if 图片 is not None else 0
            input_desc = "文本" + (f" + {img_count}张图片" if img_count > 0 else "")

            print(f"全能LLM: 模型 = {模型}")
            print(f"全能LLM: 输入 = {input_desc}")

            # 构建请求体
            request_body = {
                "model": 模型,
                "messages": messages,
                "stream": False,
            }

            # 发送请求（在独立线程中运行异步请求，避免与 ComfyUI 事件循环冲突）
            import aiohttp
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            async def _do_request():
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                }
                url = f"{self._base_url}/v1/chat/completions"
                timeout = aiohttp.ClientTimeout(total=120)

                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=headers, json=request_body) as resp:
                        status = resp.status
                        body = await resp.text()

                        if status != 200:
                            try:
                                err_data = json.loads(body)
                                err_msg = err_data.get("error", {}).get("message", body[:200])
                            except Exception:
                                err_msg = body[:200]

                            if status == 401:
                                raise ValueError(f"认证失败：API Key 无效或已过期")
                            elif status == 403:
                                raise ValueError(f"无权访问模型 {模型}")
                            elif status == 429:
                                raise ValueError(f"请求频率超限，请稍后重试")
                            elif status == 404:
                                raise ValueError(f"模型 {模型} 不存在或 API 地址错误")
                            else:
                                raise RuntimeError(f"API 错误 ({status}): {err_msg}")

                        return json.loads(body)

            def _run_in_thread():
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(_do_request())
                finally:
                    loop.close()

            with ThreadPoolExecutor(max_workers=1) as pool:
                response_data = pool.submit(_run_in_thread).result()

            # 解析响应
            choices = response_data.get("choices", [])
            if not choices:
                raise RuntimeError("API 返回了空响应（无 choices）")

            reply = choices[0].get("message", {}).get("content", "")

            # Token 用量
            usage = response_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            elapsed = time.time() - start_time
            print(f"全能LLM: 生成完成 (耗时: {elapsed:.2f}s)")
            print(f"全能LLM: Token 用量 — 输入: {prompt_tokens}, 输出: {completion_tokens}, 合计: {total_tokens}")
            if reply:
                preview = reply[:100] + "..." if len(reply) > 100 else reply
                print(f"全能LLM: 回复预览: {preview}")

            return (reply,)

        except ValueError as e:
            if str(e) == "未授权！":
                print("全能LLM: 请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            error_msg = str(e).split('\n')[0]
            print(f"全能LLM: ❌ {error_msg}")
            raise

        except Exception as e:
            error_msg = str(e).split('\n')[0]
            print(f"全能LLM: ❌ {error_msg}")
            raise RuntimeError(error_msg) from None
