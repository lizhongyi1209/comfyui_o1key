"""
异步生图节点（通用）
ComfyUI 自定义节点，通过异步提交+轮询模式调用多种生图模型

架构：
  - 节点层（本文件）：批量调度、进度条、ComfyUI 集成，不关心具体 API 协议
  - Provider 层：封装每种 API 后端的通信协议（端点、请求体格式、响应解析）

新增第三方生图模型时，只需实现 BaseAsyncImageProvider 并注册即可。
"""

import os
import time
import random
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

try:
    from comfy.model_management import processing_interrupted, InterruptProcessingException
    INTERRUPT_AVAILABLE = True
except ImportError:
    INTERRUPT_AVAILABLE = False
    InterruptProcessingException = RuntimeError  # fallback

import torch
import numpy as np
from PIL import Image

from ..utils.image_utils import tensor_to_pil, pil_to_tensor, parse_batch_prompts
from ..utils.file_utils import load_images_from_folder, pair_images_by_name, pair_images_cartesian
from ..utils.config import get_api_key_or_raise, NETWORK_ROUTE_OPTIONS, get_base_url_by_route
from ..utils.http_error import async_request_with_retry
from ..models_config import (
    get_enabled_async_models,
    get_model_provider,
    get_model_supported_aspect_ratios,
    get_all_supported_aspect_ratios,
    get_model_supported_resolutions,
    get_all_supported_resolutions,
)
from ..clients.base_async_provider import BaseAsyncImageProvider

try:
    import folder_paths  # noqa: F401
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

try:
    from comfy.utils import ProgressBar
    PROGRESS_BAR_AVAILABLE = True
except ImportError:
    PROGRESS_BAR_AVAILABLE = False

try:
    import psutil
    MEMORY_MONITOR_AVAILABLE = True
except ImportError:
    MEMORY_MONITOR_AVAILABLE = False

DEBUG_LOG_ENABLED = False
REQUEST_LOG_ENABLED = False

_POLL_INTERVAL = 2           # 轮询间隔（秒）
_INTERRUPT_CHECK_INTERVAL = 0.1  # 取消检查间隔（秒）
_MAX_WAIT_TIME = 900         # 单任务最大等待时间（秒）


def _images_to_tensor_safe(images: List[Image.Image], node_label: str) -> torch.Tensor:
    """将图像列表转为 tensor，过滤不同尺寸的图"""
    if not images:
        placeholder = Image.new('RGB', (512, 512), color=(128, 128, 128))
        return pil_to_tensor([placeholder])

    base_size = max(images, key=lambda img: img.size[0] * img.size[1]).size
    matched = [img for img in images if img.size == base_size]
    skipped = [img for img in images if img.size != base_size]

    if skipped:
        sizes_str = ", ".join(f"{img.size[0]}x{img.size[1]}" for img in skipped)
        print(
            f"{node_label}: 丢弃 {len(skipped)} 张较小尺寸的图 ({sizes_str})，"
            f"仅输出最大尺寸 {base_size[0]}x{base_size[1]} 的 {len(matched)} 张"
        )

    return pil_to_tensor(matched)


class NanoBananaV2:
    """
    异步生图节点（通用）

    功能：
      - 异步提交 + 轮询模式，避免 ComfyUI 主线程阻塞
      - 支持多种生图模型后端（通过 Provider 扩展）
      - 支持批量提示词、多参考图、代理端口
    """

    NODE_LABEL = "Nano Banana V2"

    # Provider 注册表：provider 名称 → 类路径
    PROVIDER_CLASSES = {
        "gemini_async": "..clients.gemini_async_provider.GeminiAsyncImageProvider",
    }

    # Provider 专有输入参数声明（用于 INPUT_TYPES 合并）
    _PROVIDER_EXTRA_INPUTS: Dict[str, dict] = {
        "gemini_async": {
            "联网功能": (["关闭", "打开"], {"default": "关闭"}),
        },
    }

    def __init__(self):
        self._provider: Optional[BaseAsyncImageProvider] = None
        self._provider_name: Optional[str] = None

    @classmethod
    def INPUT_TYPES(cls):
        models = get_enabled_async_models()
        if not models:
            models = ["请在 models_config.py 中启用至少一个异步模型"]

        # 宽高比 / 分辨率（取所有模型的并集，运行时验证）
        all_aspect_ratios = get_all_supported_aspect_ratios()
        if not all_aspect_ratios:
            all_aspect_ratios = [
                "1:1", "4:3", "3:4", "16:9", "9:16",
                "2:3", "3:2", "4:5", "5:4", "21:9",
                "1:4", "4:1", "1:8", "8:1"
            ]
        all_resolutions = get_all_supported_resolutions()
        if not all_resolutions:
            all_resolutions = ["512px", "1K", "2K", "4K"]

        # 可选输入（按展示顺序）
        optional = {}

        # Provider 专有参数（紧接 required 参数下方）
        for provider_extra in cls._PROVIDER_EXTRA_INPUTS.values():
            optional.update(provider_extra)

        optional["图片质量"] = (["日常", "高清"], {"default": "日常"})

        for i in range(1, 10):
            optional[f"参考图{i}"] = ("IMAGE",)

        optional["seed"] = ("INT", {
            "default": 0,
            "min": 0,
            "max": 0xffffffffffffffff
        })

        optional["分组令牌"] = ("STRING", {
            "default": "",
            "multiline": False,
            "placeholder": "手动填写分组令牌将覆盖 .config 中的默认令牌"
        })

        optional["代理端口"] = ("STRING", {
            "default": "",
            "multiline": False,
            "placeholder": "本地代理端口，如 7897（Clash Verge）或 10808（v2rayN），留空不使用"
        })

        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "一个中国女子的OOTD",
                    "multiline": True
                }),
                "网络线路": (NETWORK_ROUTE_OPTIONS, {"default": "全球加速"}),
                "模型": (models, {"default": models[0]}),
                "宽高比": (["智能"] + all_aspect_ratios, {"default": "智能"}),
                "分辨率": (all_resolutions, {"default": "2K"}),
                "生图数量": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 9,
                    "step": 1
                }),
            },
            "optional": optional
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "generate"
    CATEGORY = "image/generation"
    OUTPUT_NODE = True

    # ========================================================================
    # Provider 工厂
    # ========================================================================

    def _get_provider(self, model_id: str, proxy_url: Optional[str] = None, api_key_override: Optional[str] = None) -> BaseAsyncImageProvider:
        """根据模型 ID 获取或创建对应的 Provider 实例

        api_key_override: 手动填写的分组令牌，非空时优先使用，空则回退到 .config
        """
        provider_name = get_model_provider(model_id)
        if not provider_name:
            raise ValueError(f"模型 \"{model_id}\" 不支持异步模式")

        # 同类型 Provider 复用，更新代理和 API 密钥
        if self._provider is not None and self._provider_name == provider_name:
            self._provider.proxy_url = proxy_url
            # 根据当前 api_key_override 重新计算应使用的密钥
            api_key = api_key_override.strip() if api_key_override else ""
            if not api_key:
                api_key = get_api_key_or_raise("O1KEY_API_KEY")
            self._provider.api_key = api_key
            return self._provider

        # 创建新 Provider
        class_path = self.PROVIDER_CLASSES.get(provider_name)
        if not class_path:
            raise ValueError(f"未注册的 Provider: {provider_name}")

        module_path, class_name = class_path.rsplit(".", 1)
        if module_path.startswith(".."):
            import importlib
            module = importlib.import_module(module_path, package=__package__)
        else:
            import importlib
            module = importlib.import_module(module_path)
        provider_class = getattr(module, class_name)

        api_key = api_key_override.strip() if api_key_override else ""
        if not api_key:
            api_key = get_api_key_or_raise("O1KEY_API_KEY")
        self._provider = provider_class(api_key=api_key, proxy_url=proxy_url)
        self._provider_name = provider_name
        return self._provider

    # ========================================================================
    # 工具方法
    # ========================================================================

    @staticmethod
    def _friendly_error(error_msg: str) -> str:
        """将上游错误转化为用户友好的提示"""
        if "No available channel for model" in error_msg:
            return (
                "当前分组下模型不可用，请检查分组是否正确。"
                "若是正常出图过程中遇到该报错，说明该错误只是暂时的，稍后重试即可。或切换其他模型。"
                f"\n（原始错误: {error_msg}）"
            )
        if "401" in error_msg and "Invalid token" in error_msg:
            return "API令牌不正确或余额不足，请检查！\n（原始错误: " + error_msg + "）"
        if "Invalid token" in error_msg:
            return "API令牌不正确或余额不足，请检查！\n（原始错误: " + error_msg + "）"
        return error_msg

    @staticmethod
    def _apply_image_format(img: Image.Image, image_format: str) -> Image.Image:
        """按指定格式处理图像：JPEG 时转 RGB（质量100不压缩），PNG 不做处理"""
        if image_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
            return img.convert("RGB")
        return img

    @staticmethod
    def _check_interrupt():
        """检查 ComfyUI 是否点击了取消按钮，是则抛出 InterruptProcessingException"""
        if INTERRUPT_AVAILABLE and processing_interrupted():
            raise InterruptProcessingException()

    # ========================================================================
    # 核心异步逻辑
    # ========================================================================

    async def _submit_one(
        self,
        session: aiohttp.ClientSession,
        provider: BaseAsyncImageProvider,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        input_images: List[Image.Image],
        **extra_kwargs,
    ) -> str:
        """提交单个异步任务，返回 task_id"""
        endpoint = provider.get_submit_endpoint(model, resolution)
        request_body = provider.build_submit_body(
            prompt=prompt,
            model=model,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            images=input_images if input_images else None,
            **extra_kwargs,
        )

        url = f"{provider.api_base_url}{endpoint}"
        headers = provider.get_headers()

        if REQUEST_LOG_ENABLED:
            import json
            _log_body = {k: v for k, v in request_body.items()}
            print(f"[异步提交] URL: {url}")
            print(f"[异步提交] 请求体: {json.dumps(_log_body, ensure_ascii=False)[:500]}")

        resp = await async_request_with_retry(
            session, "POST", url, json=request_body, headers=headers,
            proxy=provider.proxy_url, prefix="异步提交: "
        )
        data = await resp.json()

        if DEBUG_LOG_ENABLED:
            import json
            print(f"[异步提交] 响应: {json.dumps(data, ensure_ascii=False)[:500]}")

        return provider.extract_task_id(data)

    async def _poll_one(
        self,
        session: aiohttp.ClientSession,
        provider: BaseAsyncImageProvider,
        task_id: str,
        on_progress=None,
    ) -> dict:
        """轮询单个任务直到完成，返回 result data；on_progress(delta) 可选，用于驱动进度条"""
        poll_endpoint = provider.get_poll_endpoint(task_id)
        url = f"{provider.api_base_url}{poll_endpoint}"
        headers = provider.get_headers()

        start_time = time.time()
        poll_count = 0
        last_progress = 0.0

        while True:
            self._check_interrupt()

            elapsed = time.time() - start_time
            if elapsed > _MAX_WAIT_TIME:
                raise RuntimeError(f"任务 {task_id} 超时（{_MAX_WAIT_TIME}秒）")

            poll_count += 1

            async with session.get(url, headers=headers, proxy=provider.proxy_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    if not error_text.strip():
                        error_text = "(服务器未返回错误详情)"
                    raise RuntimeError(f"查询任务失败 ({response.status}): {error_text}")

                result = await response.json()
                status = provider.extract_status(result)

                # 提取进度并回调（封顶 1.0 防止异常值导致进度条溢出）
                if on_progress and status in ("SUBMITTED", "IN_PROGRESS"):
                    p = provider.extract_progress(result)
                    if p is not None:
                        p = min(p, 1.0)
                        if p > last_progress:
                            on_progress(p - last_progress)
                            last_progress = p
                            print(f"{self.NODE_LABEL}: 任务{task_id[:8]}... 进度 {p * 100:.0f}%")

                if DEBUG_LOG_ENABLED:
                    import json
                    print(f"[轮询 #{poll_count}] {task_id}: status={status}")

                if status == "SUCCESS":
                    # 补足剩余进度
                    if on_progress and last_progress < 1.0:
                        on_progress(1.0 - last_progress)
                    return result.get("data", {})
                elif status == "FAILURE":
                    error_msg = result.get("error") or "未知错误"
                    if error_msg == "未知错误":
                        import json
                        print(f"{self.NODE_LABEL}: [轮询] FAILURE 但无错误信息，原始响应: {json.dumps(result, ensure_ascii=False)[:500]}")
                    friendly_msg = self._friendly_error(error_msg)
                    raise RuntimeError(f"任务失败: {friendly_msg}")
                elif status in ("SUBMITTED", "IN_PROGRESS"):
                    # 分段 sleep，每 0.1 秒检查一次取消信号
                    sleep_iterations = int(_POLL_INTERVAL / _INTERRUPT_CHECK_INTERVAL)
                    for _ in range(sleep_iterations):
                        self._check_interrupt()
                        await asyncio.sleep(_INTERRUPT_CHECK_INTERVAL)
                else:
                    raise RuntimeError(f"未知任务状态: {status}")

    async def _execute_one(
        self,
        session: aiohttp.ClientSession,
        provider: BaseAsyncImageProvider,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        input_images: List[Image.Image],
        global_task_index: int,
        on_progress=None,
        **extra_kwargs,
    ) -> dict:
        """执行单个异步生成任务（提交 + 轮询 + 解析）；on_progress(delta) 可选"""
        result = {
            "global_task_index": global_task_index,
            "prompt": prompt,
            "success": False,
            "generated_count": 0,
            "output_images": [],
            "error": None,
        }

        contributed = [0.0]  # mutable container，追踪本任务已贡献的 pbar 进度

        def _track_progress(delta):
            contributed[0] += delta
            if on_progress:
                on_progress(delta)

        try:
            t_req_start = time.time()
            task_id = await self._submit_one(
                session, provider, prompt, model,
                resolution, aspect_ratio, input_images,
                **extra_kwargs,
            )
            response_data = await self._poll_one(
                session, provider, task_id, on_progress=_track_progress,
            )
            request_time = time.time() - t_req_start

            t_dl_start = time.time()
            images_list = await provider.parse_result(response_data, session)
            download_time = time.time() - t_dl_start

            result["success"] = True
            result["generated_count"] = len(images_list)
            result["output_images"] = images_list
            result["request_time"] = request_time
            result["download_time"] = download_time
        except InterruptProcessingException:
            # 用户取消：补齐进度后向上传播，不吞掉
            if contributed[0] < 1.0 and on_progress:
                on_progress(1.0 - contributed[0])
            raise
        except Exception as e:
            result["error"] = str(e) or f"{type(e).__name__}(无错误详情)"
            # 失败也补齐 1.0 进度，保证进度条总数正确
            if contributed[0] < 1.0 and on_progress:
                on_progress(1.0 - contributed[0])

        return result

    async def _process_batch(
        self,
        provider: BaseAsyncImageProvider,
        prompts: List[str],
        model: str,
        resolution: str,
        aspect_ratio: str,
        images_per_prompt: int,
        input_images: List[Image.Image],
        pbar=None,
        **extra_kwargs,
    ) -> List[dict]:
        """全并发处理（V2 节点最多 9 个任务，无需分批）"""
        # 构建任务定义
        tasks_def = []
        for p_idx, prompt in enumerate(prompts):
            for sub_idx in range(images_per_prompt):
                tasks_def.append((p_idx, sub_idx, prompt))

        total_tasks = len(tasks_def)
        all_results: List[dict] = []
        completed = 0

        # 所有任务共享同一个进度回调，驱动同一个进度条
        _on_progress = (lambda delta: pbar.update(delta)) if pbar is not None else None

        connector = aiohttp.TCPConnector(ssl=False, limit=0, limit_per_host=0)

        async with aiohttp.ClientSession(connector=connector) as session:
            # 一次性全并发提交
            batch_tasks = []
            for i, (_, _, prompt) in enumerate(tasks_def):
                task = asyncio.create_task(
                    self._execute_one(
                        session=session,
                        provider=provider,
                        prompt=prompt,
                        model=model,
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                        input_images=input_images,
                        global_task_index=i,
                        on_progress=_on_progress,
                        **extra_kwargs,
                    )
                )
                batch_tasks.append(task)

            for coro in asyncio.as_completed(batch_tasks):
                result_data = None
                try:
                    result_data = await coro
                except InterruptProcessingException:
                    # 用户取消：终止所有未完成任务
                    for t in batch_tasks:
                        t.cancel()
                    raise
                except Exception as e2:
                    # 意外错误（不应发生，_execute_one 内部已捕获常规异常）
                    result_data = {
                        "success": False,
                        "error": str(e2),
                        "generated_count": 0,
                        "output_images": [],
                        "prompt": "",
                    }

                all_results.append(result_data)
                completed += 1

                prompt_snippet = (result_data.get("prompt", "") or "").replace("\n", " ")[:30]
                if result_data and result_data.get("success"):
                    count = result_data.get("generated_count", 1)
                    print(f"{self.NODE_LABEL}: [{completed}/{total_tasks}] {prompt_snippet}{'...' if len(prompt_snippet) >= 30 else ''} -> OK({count}张)")
                else:
                    error_msg = (result_data.get("error") or "未知错误") if result_data else "未知错误"
                    print(f"{self.NODE_LABEL}: [{completed}/{total_tasks}] {prompt_snippet}{'...' if len(prompt_snippet) >= 30 else ''} -> FAIL: {error_msg}")

            import gc
            gc.collect()

        return all_results

    # ========================================================================
    # ComfyUI 入口
    # ========================================================================

    def generate(
        self,
        prompt: str,
        模型: str,
        宽高比: str,
        分辨率: str,
        生图数量: int,
        网络线路: str = "全球加速",
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """生成图像（异步模式）"""
        start_time = time.time()

        # 提取通用可选参数
        seed: int = kwargs.pop("seed", 0)
        proxy_port: str = kwargs.pop("代理端口", "")
        api_key_override: str = kwargs.pop("分组令牌", "")
        图片质量: str = kwargs.pop("图片质量", "日常")
        image_format = "JPEG" if 图片质量 == "日常" else "PNG"

        # 初始化 Provider
        proxy_url = BaseAsyncImageProvider.build_proxy_url(proxy_port)
        provider = self._get_provider(模型, proxy_url=proxy_url, api_key_override=api_key_override)
        provider.image_compression = "webp" if 图片质量 == "日常" else None
        provider._route_base_url = get_base_url_by_route(网络线路)

        if proxy_url:
            print(f"{self.NODE_LABEL}: 已启用代理加速 -> {proxy_url}")

        # 提取 Provider 专有参数
        extra_kwargs = provider.get_extra_kwargs(**kwargs)

        # 进度条
        pbar = None
        if PROGRESS_BAR_AVAILABLE:
            pbar = ProgressBar(生图数量)

        try:
            # 初始化随机种子
            random.seed(seed)
            np.random.seed(seed % (2**32))

            # 内存监控
            if MEMORY_MONITOR_AVAILABLE and 生图数量 > 50:
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024
                print(f"{self.NODE_LABEL}: 初始内存使用: {initial_memory:.1f} MB")

            # 运行时验证分辨率
            supported_resolutions = provider.get_model_resolutions(模型)
            if supported_resolutions and 分辨率 not in supported_resolutions:
                raise ValueError(
                    f"分辨率 \"{分辨率}\" 与模型 \"{模型}\" 不兼容！\n"
                    f"该模型支持的分辨率：{', '.join(supported_resolutions)}"
                )

            # 运行时验证宽高比
            supported_ratios = provider.get_model_aspect_ratios(模型)
            if 宽高比 != "智能" and supported_ratios and 宽高比 not in supported_ratios:
                raise ValueError(
                    f"宽高比 \"{宽高比}\" 与模型 \"{模型}\" 不兼容！\n"
                    f"该模型支持的宽高比：{', '.join(supported_ratios)}"
                )

            # 收集参考图
            input_images = []
            for i in range(1, 10):
                key = f"参考图{i}"
                if key in kwargs and kwargs[key] is not None:
                    pil_imgs = tensor_to_pil(kwargs[key])
                    input_images.extend(pil_imgs)

            if input_images and len(input_images) > 14:
                raise ValueError(
                    f"输入图像数量 {len(input_images)} 超过限制 14 张"
                )

            # 单提示词模式
            mode_str = f"图生图模式 (输入{len(input_images)}张)" if input_images else "文生图模式"
            print(f"{self.NODE_LABEL}: {mode_str} | {分辨率} {宽高比} | {生图数量}张")
            prompts_list = [prompt]
            images_per_prompt = 生图数量
            total_tasks = 生图数量

            if pbar is not None:
                pbar = ProgressBar(total_tasks)

            # 在独立线程中运行异步全并发处理
            def run_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self._process_batch(
                            provider=provider,
                            prompts=prompts_list,
                            model=模型,
                            resolution=分辨率,
                            aspect_ratio=宽高比,
                            images_per_prompt=images_per_prompt,
                            input_images=input_images,
                            pbar=pbar,
                            **extra_kwargs,
                        )
                    )
                finally:
                    loop.close()

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_async)
                try:
                    results = future.result(timeout=_MAX_WAIT_TIME)
                except TimeoutError:
                    raise RuntimeError(f"任务执行超时（{_MAX_WAIT_TIME}秒），请减少数量或检查网络")

            # 统计结果
            elapsed = time.time() - start_time
            time_str = f"{elapsed:.3f}s" if elapsed < 1 else f"{elapsed:.2f}s"

            failed = [r for r in results if not r.get("success")]
            if failed:
                reason = failed[0].get("error") or "未知错误"
                print(f"{self.NODE_LABEL}: 失败！原因：{reason}")
                for fr in failed:
                    idx = fr.get("global_task_index", -1) + 1
                    prompt_snippet = (fr.get("prompt", "") or "").replace("\n", " ")[:30]
                    error_msg = fr.get("error") or "未知错误"
                    print(f"  FAIL #{idx}: {prompt_snippet}{'...' if len(prompt_snippet) >= 30 else ''} -> {error_msg}")
            else:
                total_request = sum(r.get("request_time", 0) for r in results)
                total_download = sum(r.get("download_time", 0) for r in results)
                req_str = f"{total_request:.2f}s" if total_request >= 1 else f"{total_request:.3f}s"
                dl_str = f"{total_download:.2f}s" if total_download >= 1 else f"{total_download:.3f}s"
                print(f"{self.NODE_LABEL}: 完成！请求耗时 {req_str} | 下载耗时 {dl_str} | 总耗时 {time_str}")

            # 收集输出图像
            output_images = []
            for r in results:
                output_images.extend(r.get("output_images", []))

            if not output_images:
                # 收集所有错误原因
                error_details = "\n".join(
                    f"  - {r.get('prompt', '未知提示词')[:40]}: {r.get('error') or '未知错误'}"
                    for r in failed
                )
                raise RuntimeError(
                    f"所有任务均失败 ({len(failed)}/{len(results)})：\n{error_details}"
                )

            output_images = [self._apply_image_format(img, image_format) for img in output_images]
            output_tensor = _images_to_tensor_safe(output_images, self.NODE_LABEL)

            import gc
            gc.collect()
            return (output_tensor,)

        except InterruptProcessingException:
            print(f"{self.NODE_LABEL}: 用户取消")
            raise
        except ValueError as e:
            if str(e) == "未授权！":
                print(f"{self.NODE_LABEL}: 请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            raise ValueError(str(e)) from None

        except RuntimeError as e:
            raise RuntimeError(str(e)) from None

        except Exception as e:
            raise type(e)(str(e)) from None

        finally:
            # 查询并打印余额
            try:
                balance_data = provider.query_balance_sync()
                if balance_data:
                    print(f"{self.NODE_LABEL}: {provider.format_balance_info(balance_data)}")
            except Exception:
                pass

            import gc
            gc.collect()


class NanoBananaV2Batch(NanoBananaV2):
    """
    Nano Banana V2（批量）- 全并发提交 + 即时落盘

    与原版区别：
      - 所有任务一次性全并发提交，不分批次
      - 每完成一个任务立即将图像保存到磁盘，不会因中途失败丢失已完成图片
      - 最终从磁盘加载所有已保存的图像输出
    """

    NODE_LABEL = "Nano Banana V2（批量）"
    RETURN_NAMES = ("输出图像",)

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()

        # 按指定顺序重建 optional
        new_optional = {}

        # 1. 联网功能（Provider 专有参数）
        for provider_extra in cls._PROVIDER_EXTRA_INPUTS.values():
            for k, v in provider_extra.items():
                if k in types["optional"]:
                    new_optional[k] = types["optional"][k]

        # 2. 图片质量
        new_optional["图片质量"] = (["日常", "高清"], {"default": "日常"})

        # 2.5 图片格式
        new_optional["图片格式"] = (["PNG", "JPEG"], {"default": "JPEG"})

        # 3. seed
        if "seed" in types["optional"]:
            new_optional["seed"] = types["optional"]["seed"]

        # 4. 图片配对模式
        new_optional["图片配对模式"] = (["不配对", "按相同图片命名", "1*N"], {
            "default": "不配对"
        })

        # 5. 图片命名规则
        new_optional["图片命名规则"] = (["和图片同名", "1,2,3,4..."], {
            "default": "和图片同名"
        })

        # 6. 图片保存路径（可选）
        new_optional["图片保存路径（可选）"] = ("STRING", {
            "default": "",
            "multiline": False,
            "placeholder": "留空默认保存到 ComfyUI/output；填写则保存到指定路径"
        })

        # 7. 图片路径（图1-5）
        for i in range(1, 6):
            new_optional[f"图片路径（图{i}）"] = ("STRING", {
                "default": "",
                "multiline": False,
                "placeholder": f"填写后将加载文件夹中的图片作为第{i}组参考图"
            })

        # 8. 分组令牌
        if "分组令牌" in types["optional"]:
            new_optional["分组令牌"] = types["optional"]["分组令牌"]

        # 9. 代理端口
        if "代理端口" in types["optional"]:
            new_optional["代理端口"] = types["optional"]["代理端口"]

        # 保留参考图1-9
        for i in range(1, 10):
            key = f"参考图{i}"
            if key in types["optional"]:
                new_optional[key] = types["optional"][key]

        types["optional"] = new_optional

        # 批量版隐藏生图数量参数，固定为1
        if "生图数量" in types["required"]:
            del types["required"]["生图数量"]
        if "hidden" not in types:
            types["hidden"] = {}
        types["hidden"]["生图数量"] = ("INT", {
            "default": 1,
            "min": 1,
            "max": 1000,
            "step": 1
        })

        return types

    def __init__(self):
        super().__init__()
        self._output_file_paths: List[str] = []
        self._output_dir: str = ""

    def _get_output_dir(self) -> str:
        """获取本次运行的输出目录（带时间戳）"""
        if FOLDER_PATHS_AVAILABLE:
            base = folder_paths.get_output_directory()
        else:
            base = os.path.join(os.path.dirname(__file__), "..", "output")
        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base, f"batch_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    async def _process_batch(
        self,
        provider: BaseAsyncImageProvider,
        prompts: List[str],
        model: str,
        resolution: str,
        aspect_ratio: str,
        images_per_prompt: int,
        input_images: List[Image.Image],
        image_format: str = "JPEG",
        save_path: str = "",
        save_naming: str = "{src}_{sub}.png",
        pbar=None,
        per_task_images: Optional[List[List[Image.Image]]] = None,
        per_task_names: Optional[List[str]] = None,
        **extra_kwargs,
    ) -> List[dict]:
        """全并发处理：所有任务一次性提交，谁先完成谁先落盘

        per_task_images: 可选，每个任务专属的图片列表，与 input_images 合并
        per_task_names: 可选，每个任务的源文件名列表，用于命名时保持与输入图片一致
        """
        # 构建任务定义
        tasks_def = []
        for p_idx, prompt in enumerate(prompts):
            for sub_idx in range(images_per_prompt):
                tasks_def.append((p_idx, sub_idx, prompt))

        total_tasks = len(tasks_def)
        all_results: List[dict] = []
        completed = 0

        # 初始化输出目录
        self._output_file_paths = []
        if save_path:
            self._output_dir = save_path
        else:
            self._output_dir = self._get_output_dir()
        os.makedirs(self._output_dir, exist_ok=True)
        print(f"{self.NODE_LABEL}: 输出目录: {self._output_dir}")

        _on_progress = (lambda delta: pbar.update(delta)) if pbar is not None else None

        connector = aiohttp.TCPConnector(ssl=False, limit=0, limit_per_host=0)

        async with aiohttp.ClientSession(connector=connector) as session:
            # 一次性提交所有任务（全并发）
            batch_tasks = []
            for i, (_, _, prompt) in enumerate(tasks_def):
                task_imgs = list(input_images)
                if per_task_images and i < len(per_task_images) and per_task_images[i]:
                    task_imgs = per_task_images[i] + task_imgs
                task = asyncio.create_task(
                    self._execute_one(
                        session=session,
                        provider=provider,
                        prompt=prompt,
                        model=model,
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                        input_images=task_imgs,
                        global_task_index=i,
                        on_progress=_on_progress,
                        **extra_kwargs,
                    )
                )
                batch_tasks.append(task)

            # 谁先完成先处理谁
            for coro in asyncio.as_completed(batch_tasks):
                result_data = None
                try:
                    result_data = await coro
                except InterruptProcessingException:
                    for t in batch_tasks:
                        t.cancel()
                    raise
                except Exception as e2:
                    result_data = {
                        "success": False,
                        "error": str(e2),
                        "generated_count": 0,
                        "output_images": [],
                        "prompt": "",
                    }

                # 即时落盘（应用图片格式转换 + 自定义命名规则）
                if result_data and result_data.get("success"):
                    ext = "jpg" if image_format == "JPEG" else "png"
                    # 源文件名（优先使用输入图片名，保持命名一致）
                    task_idx = result_data.get("global_task_index", completed)
                    source_name = ""
                    if per_task_names and task_idx < len(per_task_names):
                        source_name = per_task_names[task_idx]
                    prompt_snippet = (result_data.get("prompt", "") or "").replace("\n", " ")[:20]
                    prompt_sanitized = "".join(c for c in prompt_snippet if c.isalnum() or c in "._- ")
                    time_str = time.strftime("%Y%m%d_%H%M%S")
                    for img_idx, img in enumerate(result_data.get("output_images", [])):
                        img = self._apply_image_format(img, image_format)
                        # 构建文件名
                        filename = save_naming
                        filename = filename.replace("{src}", source_name)
                        filename = filename.replace("{index}", str(task_idx))
                        filename = filename.replace("{i}", str(task_idx))
                        filename = filename.replace("{num}", str(task_idx + 1))
                        filename = filename.replace("{n}", str(task_idx + 1))
                        filename = filename.replace("{sub}", str(img_idx))
                        filename = filename.replace("{prompt}", prompt_sanitized)
                        filename = filename.replace("{time}", time_str)
                        # 确保扩展名正确
                        if not filename.lower().endswith(f".{ext}"):
                            base, _ = os.path.splitext(filename)
                            filename = f"{base}.{ext}"
                        # 防覆盖
                        filepath = os.path.join(self._output_dir, filename)
                        if os.path.exists(filepath):
                            base, file_ext = os.path.splitext(filename)
                            counter = 1
                            while os.path.exists(filepath):
                                filepath = os.path.join(self._output_dir, f"{base}_{counter}{file_ext}")
                                counter += 1
                        if image_format == "JPEG":
                            img.save(filepath, "JPEG", quality=100)
                        else:
                            img.save(filepath)
                        self._output_file_paths.append(filepath)
                    # 释放内存中的图像对象
                    result_data["saved_count"] = len(result_data.get("output_images", []))
                    result_data["output_images"] = []

                all_results.append(result_data)
                completed += 1

                prompt_snippet = (result_data.get("prompt", "") or "").replace("\n", " ")[:30]
                if result_data and result_data.get("success"):
                    count = result_data.get("saved_count", result_data.get("generated_count", 1))
                    print(f"{self.NODE_LABEL}: [{completed}/{total_tasks}] {prompt_snippet}{'...' if len(prompt_snippet) >= 30 else ''} -> OK({count}张) [已落盘]")
                else:
                    error_msg = (result_data.get("error") or "未知错误") if result_data else "未知错误"
                    print(f"{self.NODE_LABEL}: [{completed}/{total_tasks}] {prompt_snippet}{'...' if len(prompt_snippet) >= 30 else ''} -> FAIL: {error_msg}")

            import gc
            gc.collect()

        return all_results

    def generate(
        self,
        prompt: str,
        模型: str,
        宽高比: str,
        分辨率: str,
        生图数量: int = 1,
        网络线路: str = "全球加速",
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """生成图像（异步模式 - 批量版：全并发 + 即时落盘）"""
        start_time = time.time()

        seed: int = kwargs.pop("seed", 0)
        proxy_port: str = kwargs.pop("代理端口", "")
        api_key_override: str = kwargs.pop("分组令牌", "")
        图片质量: str = kwargs.pop("图片质量", "日常")
        image_format: str = kwargs.pop("图片格式", "JPEG")
        save_path: str = kwargs.pop("图片保存路径（可选）", "").strip()
        命名规则选择: str = kwargs.pop("图片命名规则", "和图片同名")
        if 命名规则选择 == "1,2,3,4...":
            save_naming = "{num}.png"
        else:
            save_naming = "{src}_{sub}.png"

        proxy_url = BaseAsyncImageProvider.build_proxy_url(proxy_port)
        provider = self._get_provider(模型, proxy_url=proxy_url, api_key_override=api_key_override)
        provider.image_compression = "webp" if 图片质量 == "日常" else None
        provider._route_base_url = get_base_url_by_route(网络线路)

        if proxy_url:
            print(f"{self.NODE_LABEL}: 已启用代理加速 -> {proxy_url}")

        extra_kwargs = provider.get_extra_kwargs(**kwargs)

        pbar = None
        if PROGRESS_BAR_AVAILABLE:
            pbar = ProgressBar(生图数量)

        try:
            random.seed(seed)
            np.random.seed(seed % (2**32))

            if MEMORY_MONITOR_AVAILABLE and 生图数量 > 50:
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024
                print(f"{self.NODE_LABEL}: 初始内存使用: {initial_memory:.1f} MB")

            supported_resolutions = provider.get_model_resolutions(模型)
            if supported_resolutions and 分辨率 not in supported_resolutions:
                raise ValueError(
                    f"分辨率 \"{分辨率}\" 与模型 \"{模型}\" 不兼容！\n"
                    f"该模型支持的分辨率：{', '.join(supported_resolutions)}"
                )

            supported_ratios = provider.get_model_aspect_ratios(模型)
            if supported_ratios and 宽高比 not in supported_ratios:
                raise ValueError(
                    f"宽高比 \"{宽高比}\" 与模型 \"{模型}\" 不兼容！\n"
                    f"该模型支持的宽高比：{', '.join(supported_ratios)}"
                )

            # 收集参考图
            input_images = []
            for i in range(1, 10):
                key = f"参考图{i}"
                if key in kwargs and kwargs[key] is not None:
                    pil_imgs = tensor_to_pil(kwargs[key])
                    input_images.extend(pil_imgs)

            if input_images and len(input_images) > 14:
                raise ValueError(f"输入图像数量 {len(input_images)} 超过限制 14 张")

            # 加载文件夹图片
            folder_paths = []
            for i in range(1, 6):
                fp = kwargs.pop(f"图片路径（图{i}）", "").strip()
                if fp:
                    folder_paths.append(fp)

            pairing_mode = kwargs.pop("图片配对模式", "不配对")
            per_task_images = None
            per_task_names = None
            if folder_paths:
                # 按文件夹分组加载
                folder_image_lists = []  # List[List[ImageInfo]]
                for fp in folder_paths:
                    try:
                        infos = load_images_from_folder(fp)
                        if infos:
                            folder_image_lists.append(infos)
                    except ValueError as e:
                        print(f"{self.NODE_LABEL}: {e}")

                if folder_image_lists:
                    # 根据配对模式生成配对
                    if pairing_mode == "不配对":
                        if len(folder_image_lists) > 1:
                            raise ValueError("「不配对」模式只支持单个文件夹，请清空其他文件夹路径")
                        pairs = [(info,) for info in folder_image_lists[0]]
                    elif pairing_mode == "按相同图片命名":
                        pairs = list(pair_images_by_name(*folder_image_lists))
                    else:  # 1*N
                        pairs = list(pair_images_cartesian(*folder_image_lists))

                    if pairs:
                        batch_prompts = parse_batch_prompts(prompt)
                        per_task_images = []
                        per_task_names = []
                        prompts_list = []
                        for pair in pairs:
                            task_imgs = [info.image for info in pair] + list(input_images)
                            # 取第一张输入图的文件名作为源名，保持命名一致
                            src_name = pair[0].filename if pair else ""
                            if batch_prompts:
                                for bp in batch_prompts:
                                    per_task_images.append(list(task_imgs))
                                    per_task_names.append(src_name)
                                    prompts_list.append(bp)
                            else:
                                per_task_images.append(list(task_imgs))
                                per_task_names.append(src_name)
                                prompts_list.append(prompt)

                        images_per_prompt = 1
                        total_tasks = len(prompts_list)
                        folder_count = len(folder_paths)
                        total_folder_imgs = sum(len(lst) for lst in folder_image_lists)
                        mode_str = f"文件夹批量模式 ({pairing_mode}, {folder_count}个文件夹, {total_folder_imgs}张图片→{len(pairs)}组)"
                        if batch_prompts:
                            mode_str += f" × {len(batch_prompts)}个提示词"
                        if input_images:
                            mode_str += f" (+{len(input_images)}张参考图)"
                        print(f"{self.NODE_LABEL}: {mode_str} | {分辨率} {宽高比} | 共{total_tasks}张")

            if per_task_images is None:
                batch_prompts = parse_batch_prompts(prompt)
                if batch_prompts:
                    num_prompts = len(batch_prompts)
                    total_images = num_prompts * 生图数量
                    mode_str = f"批量提示词模式 ({num_prompts}个提示词)"
                    if input_images:
                        mode_str += f" (输入{len(input_images)}张)"
                    print(f"{self.NODE_LABEL}: {mode_str} | {分辨率} {宽高比} | 共{total_images}张")
                    prompts_list = batch_prompts
                    images_per_prompt = 生图数量
                    total_tasks = total_images
                else:
                    mode_str = f"图生图模式 (输入{len(input_images)}张)" if input_images else "文生图模式"
                    print(f"{self.NODE_LABEL}: {mode_str} | {分辨率} {宽高比} | {生图数量}张")
                    prompts_list = [prompt]
                    images_per_prompt = 生图数量
                    total_tasks = 生图数量

            if total_tasks > 100:
                print(f"  {self.NODE_LABEL}: 全并发模式，{total_tasks} 张图片将同时提交")

            if pbar is not None:
                pbar = ProgressBar(total_tasks)

            def run_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self._process_batch(
                            provider=provider,
                            prompts=prompts_list,
                            model=模型,
                            resolution=分辨率,
                            aspect_ratio=宽高比,
                            images_per_prompt=images_per_prompt,
                            input_images=input_images,
                            image_format=image_format,
                            save_path=save_path,
                            save_naming=save_naming,
                            pbar=pbar,
                            per_task_images=per_task_images,
                            per_task_names=per_task_names,
                            **extra_kwargs,
                        )
                    )
                finally:
                    loop.close()

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_async)
                # 全并发：所有任务并行，总超时 = 单任务超时
                try:
                    results = future.result(timeout=_MAX_WAIT_TIME)
                except TimeoutError:
                    raise RuntimeError(f"任务执行超时（{_MAX_WAIT_TIME}秒），请减少数量或检查网络")

            success_count = sum(1 for r in results if r.get("success"))
            fail_count = len(results) - success_count

            elapsed = time.time() - start_time
            time_str = f"{elapsed:.3f}s" if elapsed < 1 else f"{elapsed:.2f}s"
            avg_time = elapsed / success_count if success_count > 0 else 0
            avg_str = f"{avg_time:.2f}s/张" if avg_time >= 1 else f"{avg_time:.3f}s/张"
            print(f"{self.NODE_LABEL}: 完成！总耗时 {time_str} ({avg_str}) | 成功: {success_count}/{total_tasks} | 失败: {fail_count} | 已落盘: {len(self._output_file_paths)} 张")

            failed = [r for r in results if not r.get("success")]
            if failed:
                for fr in failed:
                    idx = fr.get("global_task_index", -1) + 1
                    prompt_snippet = (fr.get("prompt", "") or "")[:30]
                    error_msg = fr.get("error") or "未知错误"
                    print(f"  FAIL #{idx}: {prompt_snippet}{'...' if len(prompt_snippet) >= 30 else ''} -> {error_msg}")

            # 从磁盘加载已保存的图像
            output_images = []
            for fp in self._output_file_paths:
                try:
                    img = Image.open(fp)
                    output_images.append(img)
                except Exception as e:
                    print(f"{self.NODE_LABEL}: 加载图像失败 {fp}: {e}")

            if not output_images:
                error_details = "\n".join(
                    f"  - {r.get('prompt', '未知提示词')[:40]}: {r.get('error') or '未知错误'}"
                    for r in results if not r.get("success")
                )
                raise RuntimeError(
                    f"所有任务均失败 ({fail_count}/{total_tasks})：\n{error_details}"
                )

            output_tensor = _images_to_tensor_safe(output_images, self.NODE_LABEL)

            import gc
            gc.collect()
            return (output_tensor,)

        except InterruptProcessingException:
            # 即使被取消，已落盘的图片路径仍然保留
            if self._output_file_paths:
                print(f"{self.NODE_LABEL}: 用户取消，但 {len(self._output_file_paths)} 张已完成的图片已保存至: {self._output_dir}")
            else:
                print(f"{self.NODE_LABEL}: 用户取消")
            raise
        except ValueError as e:
            if str(e) == "未授权！":
                print(f"{self.NODE_LABEL}: 请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            raise ValueError(str(e)) from None

        except RuntimeError as e:
            raise RuntimeError(str(e)) from None

        except Exception as e:
            raise type(e)(str(e)) from None

        finally:
            try:
                balance_data = provider.query_balance_sync()
                if balance_data:
                    print(f"{self.NODE_LABEL}: {provider.format_balance_info(balance_data)}")
            except Exception:
                pass

            import gc
            gc.collect()


# 向后兼容别名（旧工作流中使用旧类名仍可正常加载）
AsyncImageGenerator = NanoBananaV2
BatchAsyncImageGenerator = NanoBananaV2Batch
