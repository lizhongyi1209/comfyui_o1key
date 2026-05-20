"""
全能LLM对话助手节点
ComfyUI 自定义节点，通过 OpenAI 兼容协议调用市面上主流的 AI 对话大模型
支持多模态（图片输入），单轮对话，非流式输出

API 密钥和地址通过插件统一配置（环境变量或 .config 文件），与 Google Gemini 节点一致
"""

import os
import time
import base64
import json
from io import BytesIO
from typing import Optional, Tuple, List

import torch
from PIL import Image

from ..utils.image_utils import tensor_to_pil
from ..utils.config import get_api_key_or_raise, get_api_base_url
from ..utils.file_types import FileList

# ============================================================================
# 模型配置
# ============================================================================

SUPPORTED_MODELS = [
    "gpt-5.5",
    "gemini-3.1-pro-preview",
    "deepseek-v4-pro",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "gemini-3.5-flash",
    "doubao-seed-2.0-pro",
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
                "视频": ("VIDEO",),
                "文件": ("FILE_LIST",),
                "令牌": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "留空则使用默认 API Key",
                }),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            },
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

    # 文件大小限制
    MAX_FILE_SIZE = 50 * 1024 * 1024       # 单文件 50MB
    MAX_TOTAL_FILE_SIZE = 50 * 1024 * 1024  # 所有文件总计 50MB

    # 常见 MIME 类型映射
    MIME_MAP = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".json": "application/json",
        ".py": "text/x-python",
        ".js": "text/javascript",
        ".html": "text/html",
        ".xml": "application/xml",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".zip": "application/zip",
    }

    # 纯文本类型，直接读取内容
    TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".py", ".js", ".ts", ".html",
                 ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log",
                 ".sh", ".bat", ".sql", ".css", ".scss", ".jsx", ".tsx"}

    def _load_files(self, file_paths_str: str) -> List[dict]:
        """读取文件列表，返回 content part 数组"""
        if not file_paths_str or not file_paths_str.strip():
            return []

        paths = [p.strip() for p in file_paths_str.split(",") if p.strip()]
        parts = []
        total_size = 0

        for path in paths:
            if not os.path.isfile(path):
                raise ValueError(f"文件不存在: {path}")

            file_size = os.path.getsize(path)
            if file_size > self.MAX_FILE_SIZE:
                raise ValueError(f"文件 {os.path.basename(path)} 大小 {file_size / 1024 / 1024:.1f}MB 超过单文件 50MB 限制")

            total_size += file_size
            if total_size > self.MAX_TOTAL_FILE_SIZE:
                raise ValueError(f"所有文件总大小超过 50MB 限制")

            ext = os.path.splitext(path)[1].lower()
            mime = self.MIME_MAP.get(ext, "application/octet-stream")
            filename = os.path.basename(path)

            if ext in self.TEXT_EXTS:
                # 文本文件直接读取内容
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    text_content = f.read()
                parts.append({
                    "type": "text",
                    "text": f"[文件: {filename}]\n```\n{text_content}\n```",
                })
            else:
                # 二进制文件转 base64，使用 file 格式（OpenAI 兼容协议）
                with open(path, "rb") as f:
                    file_data = base64.b64encode(f.read()).decode("utf-8")
                parts.append({
                    "type": "file",
                    "file": {
                        "filename": filename,
                        "file_data": f"data:{mime};base64,{file_data}",
                    },
                })

            print(f"全能LLM: 加载文件 {filename} ({file_size / 1024:.1f}KB, {mime})")

        return parts

    def _build_input(
        self,
        prompt: str,
        images: Optional[torch.Tensor] = None,
        file_paths: str = "",
        file_list: Optional[FileList] = None,
        video=None,
    ) -> list:
        """构建 chat/completions 格式的 messages 数组"""
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

        # 处理视频输入（ComfyUI VIDEO 类型）
        video_url_str = ""
        if video is not None:
            # 从 VIDEO 对象中提取文件路径
            vp = None
            if isinstance(video, dict):
                vp = video.get("video") or video.get("path") or video.get("file") or video.get("filename")
                if not vp:
                    for val in video.values():
                        if isinstance(val, str) and os.path.exists(val):
                            vp = val
                            break
            elif isinstance(video, str):
                vp = video
            else:
                for attr in ("video", "path", "filename"):
                    if hasattr(video, attr):
                        vp = getattr(video, attr)
                        break
                if not vp and hasattr(video, "__dict__"):
                    for attr_val in video.__dict__.values():
                        if isinstance(attr_val, str) and os.path.isfile(attr_val):
                            vp = attr_val
                            break

            if not vp or not os.path.isfile(vp):
                raise ValueError(f"视频文件不存在或路径无效: {vp}")

            mime_map = {
                ".mp4": "video/mp4", ".mpeg": "video/mpeg", ".mpg": "video/mpg",
                ".mov": "video/quicktime", ".avi": "video/x-msvideo",
                ".flv": "video/x-flv", ".webm": "video/webm",
                ".wmv": "video/x-ms-wmv", ".mkv": "video/x-matroska",
            }
            ext = os.path.splitext(vp)[1].lower()
            mime = mime_map.get(ext, "video/mp4")
            file_size = os.path.getsize(vp)
            print(f"全能LLM: 加载视频 {os.path.basename(vp)} ({file_size / 1024 / 1024:.1f}MB, {mime})")
            with open(vp, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            video_url_str = f"data:{mime};base64,{b64}"

        # 加载文件：优先使用 FILE_LIST，其次使用字符串路径
        file_parts = []
        if file_list:
            for fd in file_list:
                print(f"全能LLM: 使用文件 {fd.filename}{fd.extension} ({fd.size / 1024:.1f}KB)")
                file_parts.append({
                    "type": "file",
                    "file": {
                        "filename": fd.filename + fd.extension,
                        "file_data": f"data:{fd.mime_type};base64,{fd.data}",
                    },
                })
        elif file_paths:
            file_parts = self._load_files(file_paths)

        # 纯文本，无图片无文件无视频
        if not image_data_urls and not file_parts and not video_url_str:
            return [{"role": "user", "content": prompt}]

        content_parts = []

        # 图片
        for url in image_data_urls:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": url},
            })

        # 视频：用 image_url 类型传 data URL（Gemini OpenAI 兼容层支持此格式）
        # 同时保留 video_url 类型作为备用（其他支持 video_url 的模型）
        if video_url_str:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": video_url_str},
            })

        # 文件
        for fp in file_parts:
            content_parts.append(fp)

        content_parts.append({
            "type": "text",
            "text": prompt,
        })

        return [{"role": "user", "content": content_parts}]

    @staticmethod
    def _send_stream_token(node_id, token, done=False):
        """通过 PromptServer 向前端推送流式 token"""
        try:
            from server import PromptServer
            PromptServer.instance.send_sync(
                "o1key.stream_token",
                {"node_id": str(node_id), "token": token, "done": done},
            )
        except Exception:
            pass

    def generate(
        self,
        模型: str,
        提示词: str,
        图片: Optional[torch.Tensor] = None,
        视频=None,
        文件: Optional[FileList] = None,
        令牌: str = "",
        node_id: str = "",
    ) -> Tuple[str]:
        start_time = time.time()

        try:
            self._ensure_config()

            # 如果用户传入了自定义令牌，则覆盖默认 API Key
            effective_api_key = 令牌.strip() if 令牌 and 令牌.strip() else self._api_key

            # 构建 input
            input_data = self._build_input(提示词, 图片, "", 文件, 视频)

            img_count = len(tensor_to_pil(图片)) if 图片 is not None else 0
            file_count = len(文件) if 文件 else 0
            input_desc = "文本"
            if img_count: input_desc += f" + {img_count}张图片"
            if 视频 is not None: input_desc += " + 视频"
            if file_count: input_desc += f" + {file_count}个文件"

            print(f"全能LLM: 模型 = {模型}")
            print(f"全能LLM: 输入 = {input_desc}")

            # 构建请求体（chat/completions 格式）
            request_body = {
                "model": 模型,
                "messages": input_data,
                "stream": True,
            }

            # 打印请求体，base64 截断显示
            def _truncate_for_log(obj):
                if isinstance(obj, dict):
                    return {k: _truncate_for_log(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_truncate_for_log(i) for i in obj]
                if isinstance(obj, str) and (obj.startswith("data:image") or obj.startswith("data:application") or obj.startswith("data:text")):
                    return obj[:60] + f"...[{len(obj)}chars]"
                return obj
            print(f"全能LLM: 请求原始内容 = {json.dumps(_truncate_for_log(request_body), ensure_ascii=False)}")

            # 发送请求（在独立线程中运行异步请求，避免与 ComfyUI 事件循环冲突）
            import aiohttp
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            async def _do_request():
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {effective_api_key}",
                }
                url = f"{self._base_url}/v1/chat/completions"
                timeout = aiohttp.ClientTimeout(total=120)

                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=headers, json=request_body) as resp:
                        status = resp.status

                        if status != 200:
                            body = await resp.text()
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

                        # 流式读取，拼接 delta content
                        reply_parts = []
                        async for raw_line in resp.content:
                            line = raw_line.decode("utf-8").strip()
                            if not line or not line.startswith("data:"):
                                continue
                            data_str = line[len("data:"):].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except Exception:
                                continue
                            choices = chunk.get("choices")
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                reply_parts.append(content)
                                UniversalLLMChat._send_stream_token(node_id, content)

                        UniversalLLMChat._send_stream_token(node_id, "", done=True)
                        return "".join(reply_parts)

            def _run_in_thread():
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(_do_request())
                finally:
                    loop.close()

            with ThreadPoolExecutor(max_workers=1) as pool:
                reply = pool.submit(_run_in_thread).result()

            elapsed = time.time() - start_time
            print(f"全能LLM: 生成完成 (耗时: {elapsed:.2f}s)")
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
