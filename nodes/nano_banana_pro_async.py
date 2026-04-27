"""
Nano Banana Pro（异步）节点
ComfyUI 自定义节点，用于调用 Gemini 模型生成图像（异步提交+轮询模式）
"""

import time
import math
import random
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List

import torch
import numpy as np
from PIL import Image

from ..utils.image_utils import tensor_to_pil, pil_to_tensor, parse_batch_prompts
from ..utils.file_utils import ImageInfo, generate_timestamp_filename, save_image
from ..utils.config import get_async_api_base_url
from ..clients.gemini_client import GeminiAPIClient
from ..models_config import (
    get_enabled_models, get_model_description,
    get_model_supported_aspect_ratios, get_all_supported_aspect_ratios,
    get_model_supported_resolutions, get_all_supported_resolutions
)

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

try:
    from comfy.utils import ProgressBar
    PROGRESS_BAR_AVAILABLE = True
except ImportError:
    PROGRESS_BAR_AVAILABLE = False
    print("⚠️ NanoBananaProAsync: comfy.utils.ProgressBar 不可用，将只使用终端进度显示")

try:
    import psutil
    MEMORY_MONITOR_AVAILABLE = True
except ImportError:
    MEMORY_MONITOR_AVAILABLE = False
    print("⚠️ NanoBananaProAsync: psutil 不可用，内存监控功能禁用")

DEBUG_LOG_ENABLED = True
REQUEST_LOG_ENABLED = True

_NODE = "Nano Banana Pro（异步）"
_POLL_INTERVAL = 4  # 轮询间隔（秒）
_MAX_WAIT_TIME = 300  # 最大等待时间（秒）


def _images_to_tensor_safe(images: List[Image.Image], node_label: str) -> torch.Tensor:
    if not images:
        placeholder = Image.new('RGB', (512, 512), color=(128, 128, 128))
        return pil_to_tensor([placeholder])

    base_size = max(images, key=lambda img: img.size[0] * img.size[1]).size
    matched = [img for img in images if img.size == base_size]
    skipped = [img for img in images if img.size != base_size]

    if skipped:
        sizes_str = ", ".join(f"{img.size[0]}×{img.size[1]}" for img in skipped)
        print(
            f"{node_label}: 丢弃 {len(skipped)} 张较小尺寸的图 ({sizes_str})，"
            f"仅输出最大尺寸 {base_size[0]}×{base_size[1]} 的 {len(matched)} 张"
        )

    return pil_to_tensor(matched)


class NanoBananaProAsync:
    """
    Nano Banana Pro（异步）节点

    功能：
    - 异步提交任务到 cf-api.o1key.com
    - 轮询任务状态直到完成
    - 支持批量并发生成
    """

    MODELS = None
    ASPECT_RATIOS = [
        "1:1", "4:3", "3:4", "16:9", "9:16",
        "2:3", "3:2", "4:5", "5:4", "21:9",
        "1:4", "4:1", "1:8", "8:1"
    ]
    RESOLUTIONS = ["512px", "1K", "2K", "4K"]

    def __init__(self):
        self.client = None

    @classmethod
    def INPUT_TYPES(cls):
        enabled_models = get_enabled_models()
        if not enabled_models:
            enabled_models = ["请在 models_config.py 中启用至少一个模型"]

        all_aspect_ratios = get_all_supported_aspect_ratios()
        if not all_aspect_ratios:
            all_aspect_ratios = cls.ASPECT_RATIOS

        all_resolutions = get_all_supported_resolutions()
        if not all_resolutions:
            all_resolutions = cls.RESOLUTIONS

        optional_inputs = {}
        for i in range(1, 10):
            optional_inputs[f"参考图{i}"] = ("IMAGE",)

        optional_inputs["代理端口"] = ("STRING", {
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
                "模型": (enabled_models, {
                    "default": enabled_models[0]
                }),
                "宽高比": (all_aspect_ratios, {
                    "default": "1:1"
                }),
                "分辨率": (all_resolutions, {
                    "default": "2K"
                }),
                "生图数量": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "谷歌搜索（联网）": (["关闭", "打开"], {
                    "default": "关闭"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                })
            },
            "optional": optional_inputs
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)

    try:
        import folder_paths
        FOLDER_PATHS_AVAILABLE = True
    except ImportError:
        FOLDER_PATHS_AVAILABLE = False

    FUNCTION = "generate"
    CATEGORY = "image/generation"

    async def _submit_task_async(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        images: List[Image.Image],
        enable_grounding: bool = False,
    ) -> str:
        """提交异步任务，返回 task_id"""
        endpoint = self.client.get_endpoint(model=model, resolution=resolution, image_format="url")
        async_endpoint = f"/async{endpoint.split('?')[0]}"
        if "?" in endpoint:
            async_endpoint += "?" + endpoint.split("?")[1]

        request_body = self.client.build_request_body(
            prompt=prompt,
            images=images if images else None,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            enable_grounding=enable_grounding,
            enable_image_search=False,
        )

        url = f"{get_async_api_base_url()}{async_endpoint}"
        headers = {
            "Authorization": f"Bearer {self.client.api_key}",
            "Content-Type": "application/json"
        }

        if REQUEST_LOG_ENABLED:
            import json
            import copy
            debug_body = copy.deepcopy(request_body)
            for content in debug_body.get("contents", []):
                for part in content.get("parts", []):
                    if "inline_data" in part and "data" in part["inline_data"]:
                        data_str = part["inline_data"]["data"]
                        part["inline_data"]["data"] = f"{data_str[:50]}...[截断]" if len(data_str) > 50 else data_str
            print(f"\n{'='*60}")
            print(f"[异步提交] URL: {url}")
            print(f"[异步提交] 请求体:\n{json.dumps(debug_body, indent=2, ensure_ascii=False)}")
            print(f"{'='*60}\n")

        async with session.post(url, json=request_body, headers=headers, proxy=self.client.proxy_url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"提交任务失败 ({response.status}): {error_text}")

            data = await response.json()

            if DEBUG_LOG_ENABLED:
                import json
                print(f"\n{'='*60}")
                print(f"[异步提交] 响应:\n{json.dumps(data, indent=2, ensure_ascii=False)}")
                print(f"{'='*60}\n")

            task_id = data.get("task_id")
            if not task_id:
                raise RuntimeError(f"提交响应中未找到 task_id: {data}")

            return task_id

    async def _poll_task_async(
        self,
        session: aiohttp.ClientSession,
        task_id: str,
    ) -> dict:
        """轮询任务状态直到完成"""
        url = f"{get_async_api_base_url()}/async/v1/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {self.client.api_key}",
            "Content-Type": "application/json"
        }

        start_time = time.time()
        poll_count = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed > _MAX_WAIT_TIME:
                raise RuntimeError(f"任务 {task_id} 超时（{_MAX_WAIT_TIME}秒），请稍后手动查询")

            poll_count += 1

            async with session.get(url, headers=headers, proxy=self.client.proxy_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"查询任务失败 ({response.status}): {error_text}")

                result = await response.json()
                status = result.get("status")

                if DEBUG_LOG_ENABLED:
                    import json
                    print(f"\n{'='*60}")
                    print(f"[轮询 #{poll_count}] task_id: {task_id}")
                    print(f"[轮询 #{poll_count}] 响应:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
                    print(f"{'='*60}\n")

                if status == "SUCCESS":
                    return result.get("data", {})
                elif status == "FAILURE":
                    error_msg = result.get("error", "未知错误")
                    raise RuntimeError(f"任务失败: {error_msg}")
                elif status in ["SUBMITTED", "IN_PROGRESS"]:
                    await asyncio.sleep(_POLL_INTERVAL)
                else:
                    raise RuntimeError(f"未知任务状态: {status}")

    async def _generate_single_task(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        images: List[Image.Image],
        output_folder: str,
        global_task_index: int,
        enable_grounding: bool = False,
        save_to_disk: bool = True,
    ) -> dict:
        """执行单个异步生成任务"""
        result = {
            "global_task_index": global_task_index,
            "prompt": prompt,
            "success": False,
            "generated_count": 0,
            "saved_files": [],
            "output_images": [],
            "error": None
        }

        try:
            task_id = await self._submit_task_async(
                session=session,
                prompt=prompt,
                model=model,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                images=images,
                enable_grounding=enable_grounding,
            )

            response_data = await self._poll_task_async(session=session, task_id=task_id)

            images_list, _ = await self.client.parse_response_async(response_data, session=session)

            if save_to_disk:
                for gen_img in images_list:
                    output_path = generate_timestamp_filename(
                        output_folder=output_folder,
                        extension=".png"
                    )
                    save_image(gen_img, output_path)
                    result["saved_files"].append(output_path)
                    gen_img = None
            else:
                result["output_images"] = images_list

            result["success"] = True
            result["generated_count"] = len(images_list)
        except Exception as e:
            result["error"] = str(e)

        return result

    async def _process_batch_async(
        self,
        prompts: List[str],
        model: str,
        resolution: str,
        aspect_ratio: str,
        images_per_prompt: int,
        input_images: List[Image.Image],
        output_folder: str,
        pbar=None,
        enable_grounding: bool = False,
        save_to_disk: bool = True,
    ) -> List[dict]:
        """异步批量处理"""
        tasks_def = []
        for p_idx, prompt in enumerate(prompts):
            for sub_idx in range(images_per_prompt):
                tasks_def.append((p_idx, sub_idx, prompt))

        total_tasks = len(tasks_def)
        max_concurrent = 50
        num_batches = math.ceil(total_tasks / max_concurrent)

        all_results = []
        completed = 0
        success_count = 0
        fail_count = 0

        connector = aiohttp.TCPConnector(ssl=False, limit=0, limit_per_host=0)

        async with aiohttp.ClientSession(connector=connector) as session:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * max_concurrent
                end_idx = min(start_idx + max_concurrent, total_tasks)

                tasks = []
                for i in range(start_idx, end_idx):
                    _, _, prompt = tasks_def[i]
                    task = asyncio.create_task(
                        self._generate_single_task(
                            session=session,
                            prompt=prompt,
                            model=model,
                            resolution=resolution,
                            aspect_ratio=aspect_ratio,
                            images=input_images,
                            output_folder=output_folder,
                            global_task_index=i,
                            enable_grounding=enable_grounding,
                            save_to_disk=save_to_disk,
                        )
                    )
                    tasks.append(task)

                batch_results = []
                for coro in asyncio.as_completed(tasks):
                    result_data = None
                    try:
                        result = await coro
                        if isinstance(result, Exception):
                            result_data = {"success": False, "error": str(result), "generated_count": 0, "saved_files": [], "prompt": ""}
                        else:
                            result_data = result
                            batch_results.append(result_data)
                    except Exception as e:
                        result_data = {"success": False, "error": str(e), "generated_count": 0, "saved_files": [], "prompt": ""}
                        batch_results.append(result_data)

                    completed += 1
                    prompt_snippet = (result_data.get("prompt", "") or "")[:30]

                    if result_data and result_data.get("success", False):
                        success_count += 1
                        count = result_data.get("generated_count", 1)
                        print(f"{_NODE}: [{completed}/{total_tasks}] {prompt_snippet}{'...' if len(prompt_snippet) >= 30 else ''} → ✓成功({count}张)")
                    else:
                        fail_count += 1
                        error_msg = result_data.get("error", "未知错误") if result_data else "未知错误"
                        print(f"{_NODE}: [{completed}/{total_tasks}] {prompt_snippet}{'...' if len(prompt_snippet) >= 30 else ''} → ✗失败: {error_msg}")

                    if pbar is not None:
                        pbar.update(1)

                all_results.extend(batch_results)

                import gc
                gc.collect()

                await asyncio.sleep(0.1)

        return all_results

    def generate(
        self,
        prompt: str,
        模型: str,
        宽高比: str,
        分辨率: str,
        生图数量: int,
        seed: int,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """生成图像（异步模式）"""
        start_time = time.time()

        enable_grounding: bool = (kwargs.pop("谷歌搜索（联网）", "关闭") == "打开")
        proxy_port: str = kwargs.pop("代理端口", "")

        pbar = None
        if PROGRESS_BAR_AVAILABLE:
            pbar = ProgressBar(生图数量)

        try:
            random.seed(seed)
            np.random.seed(seed % (2**32))

            if MEMORY_MONITOR_AVAILABLE and 生图数量 > 50:
                import psutil
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024
                print(f"{_NODE}: 初始内存使用: {initial_memory:.1f} MB")

            if self.client is None:
                try:
                    self.client = GeminiAPIClient()
                except ValueError as e:
                    raise ValueError(f"初始化失败: {str(e)}")

            self.client.proxy_url = GeminiAPIClient.build_proxy_url(proxy_port)
            if self.client.proxy_url:
                print(f"{_NODE}: 已启用代理加速 → {self.client.proxy_url}")

            supported_resolutions = get_model_supported_resolutions(模型)
            if supported_resolutions and 分辨率 not in supported_resolutions:
                raise ValueError(
                    f"分辨率 \"{分辨率}\" 与模型 \"{模型}\" 不兼容！\n"
                    f"该模型支持的分辨率：{', '.join(supported_resolutions)}"
                )

            supported_ratios = get_model_supported_aspect_ratios(模型)
            if supported_ratios and 宽高比 not in supported_ratios:
                raise ValueError(
                    f"宽高比 \"{宽高比}\" 与模型 \"{模型}\" 不兼容！\n"
                    f"该模型支持的宽高比：{', '.join(supported_ratios)}"
                )

            input_images = []
            for i in range(1, 10):
                key = f"参考图{i}"
                if key in kwargs and kwargs[key] is not None:
                    pil_imgs = tensor_to_pil(kwargs[key])
                    input_images.extend(pil_imgs)

            if input_images:
                if len(input_images) > 14:
                    raise ValueError(
                        f"输入图像数量 {len(input_images)} 超过限制 14 张，请减少输入图像数量"
                    )

            batch_prompts = parse_batch_prompts(prompt)

            grounding_str = ""
            if enable_grounding:
                grounding_str = " | 谷歌搜索接地"

            if batch_prompts:
                num_prompts = len(batch_prompts)
                total_images = num_prompts * 生图数量
                mode_str = f"批量提示词模式 ({num_prompts}个提示词)"
                if input_images:
                    mode_str += f" (输入{len(input_images)}张)"
                print(f"{_NODE}: {mode_str} | {分辨率} {宽高比} | 共{total_images}张{grounding_str}")

                if total_images > 100:
                    print(f"⚠️ {_NODE}: 警告！批量生成 {total_images} 张图片，内存占用可能较高")
                    print(f"⚠️ 建议：分批执行或减少生图数量")
            else:
                mode_str = f"图生图模式 (输入{len(input_images)}张)" if input_images else "文生图模式"
                print(f"{_NODE}: {mode_str} | {分辨率} {宽高比} | {生图数量}张{grounding_str}")

                if 生图数量 > 100:
                    print(f"⚠️ {_NODE}: 警告！批量生成 {生图数量} 张图片，内存占用可能较高")
                    print(f"⚠️ 建议：分批执行或减少生图数量")

            success_count = 0
            fail_count = 0

            if batch_prompts:
                num_prompts = len(batch_prompts)
                total_images = num_prompts * 生图数量

                if pbar is not None:
                    pbar = ProgressBar(total_images)

                def run_async_in_thread():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            self._process_batch_async(
                                prompts=batch_prompts,
                                model=模型,
                                resolution=分辨率,
                                aspect_ratio=宽高比,
                                images_per_prompt=生图数量,
                                input_images=input_images,
                                output_folder="",
                                pbar=pbar,
                                enable_grounding=enable_grounding,
                                save_to_disk=False,
                            )
                        )
                    finally:
                        loop.close()

                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_async_in_thread)
                    try:
                        results = future.result(timeout=900)
                    except TimeoutError:
                        raise RuntimeError("任务执行超时（900秒），请减少提示词数量或检查网络连接")

                success_count = sum(1 for r in results if r.get("success", False))
                fail_count = len(results) - success_count
                total_generated = sum(r.get("generated_count", 0) for r in results)

                elapsed = time.time() - start_time
                time_str = f"{elapsed:.3f}s" if elapsed < 1 else f"{elapsed:.2f}s"

                print(f"完成！总耗时 {time_str} | 成功: {success_count}/{total_images} | 失败: {fail_count}")

                failed_results = [r for r in results if not r.get("success", False)]
                if failed_results:
                    for fr in failed_results:
                        idx = fr.get("global_task_index", -1) + 1
                        prompt_snippet = (fr.get("prompt", "") or "")[:30]
                        error_msg = fr.get("error", "未知错误")
                        print(f"  失败 #{idx}: {prompt_snippet}{'...' if len(prompt_snippet) >= 30 else ''} → {error_msg}")

                output_images = []
                for r in results:
                    output_images.extend(r.get("output_images", []))

                if not output_images:
                    placeholder = Image.new('RGB', (512, 512), color=(128, 128, 128))
                    output_images = [placeholder]

                output_tensor = _images_to_tensor_safe(output_images, _NODE)

                import gc
                gc.collect()
                return (output_tensor,)
            else:
                if pbar is not None:
                    pbar = ProgressBar(生图数量)

                def run_async_in_thread():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            self._process_batch_async(
                                prompts=[prompt],
                                model=模型,
                                resolution=分辨率,
                                aspect_ratio=宽高比,
                                images_per_prompt=生图数量,
                                input_images=input_images,
                                output_folder="",
                                pbar=pbar,
                                enable_grounding=enable_grounding,
                                save_to_disk=False,
                            )
                        )
                    finally:
                        loop.close()

                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_async_in_thread)
                    try:
                        results = future.result(timeout=900)
                    except TimeoutError:
                        raise RuntimeError("任务执行超时（900秒），请减少生图数量或检查网络连接")

                success_count = sum(1 for r in results if r.get("success", False))
                fail_count = len(results) - success_count
                total_generated = sum(r.get("generated_count", 0) for r in results)

                elapsed = time.time() - start_time
                time_str = f"{elapsed:.3f}s" if elapsed < 1 else f"{elapsed:.2f}s"
                print(f"完成！总耗时 {time_str} | 成功: {success_count}/{生图数量} | 失败: {fail_count}")

                failed_results = [r for r in results if not r.get("success", False)]
                if failed_results:
                    for fr in failed_results:
                        idx = fr.get("global_task_index", -1) + 1
                        error_msg = fr.get("error", "未知错误")
                        print(f"  失败 #{idx}: {prompt[:30]}{'...' if len(prompt) >= 30 else ''} → {error_msg}")

                output_images = []
                for r in results:
                    output_images.extend(r.get("output_images", []))

                if not output_images:
                    placeholder = Image.new('RGB', (512, 512), color=(128, 128, 128))
                    output_images = [placeholder]

                output_tensor = _images_to_tensor_safe(output_images, _NODE)

                import gc
                gc.collect()
                return (output_tensor,)

        except ValueError as e:
            if str(e) == "未授权！":
                print("请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            raise ValueError(str(e)) from None

        except RuntimeError as e:
            raise RuntimeError(str(e)) from None

        except Exception as e:
            raise type(e)(str(e)) from None

        finally:
            if self.client is not None:
                try:
                    balance_data = self.client.query_balance_sync()
                    balance_info = self.client.format_balance_info(balance_data)
                    print(f"{_NODE}: {balance_info}")
                except Exception:
                    pass

            import gc
            gc.collect()
