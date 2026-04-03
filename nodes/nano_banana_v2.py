"""
Nano Banana v2 节点
NanoBananaPro 的完全复刻，唯一改动：

  将原来 9 个独立「参考图1~9」输入端
  改为 1 个「参考图」输入端（可选），配合「加载图像（批量）」节点使用。

  「加载图像（批量）」输出 is_output_list=True（list[Tensor]），
  本节点声明 INPUT_IS_LIST = True 来整体接收该列表，
  然后在 generate() 开头对所有参数统一解包，其余业务逻辑与原节点完全一致。
"""

import os
import gc
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

try:
    import psutil
    MEMORY_MONITOR_AVAILABLE = True
except ImportError:
    MEMORY_MONITOR_AVAILABLE = False

DEBUG_LOG_ENABLED = False
REQUEST_LOG_ENABLED = False

_NODE = "Nano Banana v2"


def _images_to_tensor_safe(images: List[Image.Image], node_label: str) -> torch.Tensor:
    """
    将 PIL Image 列表转换为 ComfyUI tensor，安全处理多张不同尺寸的情况。

    ComfyUI 的 IMAGE tensor 格式为 [B, H, W, C]，要求 batch 内所有图尺寸相同。
    当 API 返回多张不同分辨率的图时（主图 + 附图），直接 stack 会崩溃。

    策略：
    - 所有图均已按原始分辨率保存到磁盘（调用此函数前已完成）
    - 以第一张图的尺寸为基准，只将尺寸相同的图纳入 tensor 输出
    - 尺寸不同的图跳过（不 resize、不丢弃磁盘文件），并打印日志提示
    - 若没有任何图与第一张尺寸相同（极罕见），则只输出第一张
    """
    if not images:
        placeholder = Image.new('RGB', (512, 512), color=(128, 128, 128))
        return pil_to_tensor([placeholder])

    base_size = images[0].size  # PIL size = (W, H)
    matched = [img for img in images if img.size == base_size]
    skipped = [img for img in images if img.size != base_size]

    if skipped:
        sizes_str = ", ".join(f"{img.size[0]}×{img.size[1]}" for img in skipped)
        print(
            f"{node_label}: API 额外返回了 {len(skipped)} 张不同尺寸的图 ({sizes_str})，"
            f"已按原始分辨率保存到磁盘，tensor 输出仅包含与主图尺寸相同的 {len(matched)} 张 "
            f"({base_size[0]}×{base_size[1]})"
        )

    return pil_to_tensor(matched if matched else [images[0]])


class NanaBananaV2:
    """
    Nano Banana v2

    与 NanoBananaPro 完全一致，参考图输入方式不同：
    - 原版：9 个独立可选端口（参考图1~9）
    - v2：1 个可选端口「参考图」，配合「加载图像（批量）」可传入任意数量图片
    """

    ASPECT_RATIOS = [
        "1:1", "4:3", "3:4", "16:9", "9:16",
        "2:3", "3:2", "4:5", "5:4", "21:9",
        "1:4", "4:1", "1:8", "8:1"
    ]
    RESOLUTIONS = ["512", "1K", "2K", "4K"]

    def __init__(self):
        self.client = None

    @classmethod
    def INPUT_TYPES(cls):
        enabled_models = get_enabled_models()
        if not enabled_models:
            enabled_models = ["请在 models_config.py 中启用至少一个模型"]

        all_aspect_ratios = get_all_supported_aspect_ratios() or cls.ASPECT_RATIOS
        all_resolutions = get_all_supported_resolutions() or cls.RESOLUTIONS

        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "一个中国女子的OOTD",
                    "multiline": True
                }),
                "模型": (enabled_models, {"default": enabled_models[0]}),
                "宽高比": (all_aspect_ratios, {"default": "1:1"}),
                "分辨率": (all_resolutions, {"default": "2K"}),
                "生图数量": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "像素缩放": ("BOOLEAN", {"default": True, "label_on": "打开", "label_off": "关闭"}),
                "分辨率像素": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1, "display": "number"}),
                "谷歌搜索（联网）": (["关闭", "打开"], {"default": "关闭"}),
                "图片搜索（联网）": (["关闭", "打开"], {"default": "关闭"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                # 单个参考图端口，接受普通 IMAGE 或「加载图像（批量）」输出的列表
                "参考图": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "generate"
    CATEGORY = "image/generation"

    # 声明 INPUT_IS_LIST，使 ComfyUI 将「加载图像（批量）」的 list[Tensor]
    # 整体传入而非逐张迭代执行，同时其余所有参数也会被包进 list，需解包。
    INPUT_IS_LIST = True

    # ------------------------------------------------------------------ #
    #  以下方法与 NanoBananaPro 完全相同，仅 generate() 开头增加了解包逻辑
    # ------------------------------------------------------------------ #

    def resize_to_megapixels(self, image: Image.Image, target_megapixels: float) -> Image.Image:
        current_pixels = image.width * image.height
        target_pixels = int(target_megapixels * 1_000_000)
        if abs(current_pixels - target_pixels) / target_pixels < 0.05:
            return image
        scale = (target_pixels / current_pixels) ** 0.5
        new_width = max(1, int(image.width * scale))
        new_height = max(1, int(image.height * scale))
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

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
        enable_image_search: bool = False,
    ) -> dict:
        result = {
            "global_task_index": global_task_index,
            "prompt": prompt,
            "success": False,
            "generated_count": 0,
            "saved_files": [],
            "error": None
        }
        try:
            gen_result = await self.client.generate_single_async(
                prompt=prompt,
                model=model,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                images=images if images else None,
                session=session,
                debug=DEBUG_LOG_ENABLED,
                debug_request=REQUEST_LOG_ENABLED,
                enable_grounding=enable_grounding,
                enable_image_search=enable_image_search,
            )
            if gen_result:
                images_list, _ = gen_result
                for gen_img in images_list:
                    output_path = generate_timestamp_filename(
                        output_folder=output_folder,
                        extension=".png"
                    )
                    save_image(gen_img, output_path)
                    result["saved_files"].append(output_path)
                    gen_img = None
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
        enable_image_search: bool = False,
    ) -> List[dict]:
        tasks_def = []
        for p_idx, prompt in enumerate(prompts):
            for sub_idx in range(images_per_prompt):
                tasks_def.append((p_idx, sub_idx, prompt))

        total_tasks = len(tasks_def)
        num_prompts = len(prompts)
        print(f"{_NODE}: 批量提示词模式 | {num_prompts}个提示词 × {images_per_prompt}张/提示词 | 共{total_tasks}任务")

        max_concurrent = 10
        num_batches = math.ceil(total_tasks / max_concurrent)
        all_results = []
        completed = 0
        success_count = 0
        fail_count = 0

        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
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
                            enable_image_search=enable_image_search,
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
                gc.collect()
                await asyncio.sleep(0.1)

        return all_results

    def generate(
        self,
        prompt,
        模型,
        宽高比,
        分辨率,
        生图数量,
        像素缩放,
        分辨率像素,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        # ----------------------------------------------------------------
        # INPUT_IS_LIST=True 时，所有参数均为 list，先统一解包为标量
        # ----------------------------------------------------------------
        prompt      = prompt[0]      if isinstance(prompt,      list) else prompt
        模型        = 模型[0]        if isinstance(模型,        list) else 模型
        宽高比      = 宽高比[0]      if isinstance(宽高比,      list) else 宽高比
        分辨率      = 分辨率[0]      if isinstance(分辨率,      list) else 分辨率
        生图数量    = 生图数量[0]    if isinstance(生图数量,    list) else 生图数量
        像素缩放    = 像素缩放[0]    if isinstance(像素缩放,    list) else 像素缩放
        分辨率像素  = 分辨率像素[0]  if isinstance(分辨率像素,  list) else 分辨率像素

        # seed 也在 kwargs 里（含全角括号的参数名无法作为形参）
        seed_raw = kwargs.pop("seed", [0])
        seed: int = seed_raw[0] if isinstance(seed_raw, list) else seed_raw

        # 搜索开关同理
        grounding_raw      = kwargs.pop("谷歌搜索（联网）", ["关闭"])
        image_search_raw   = kwargs.pop("图片搜索（联网）", ["关闭"])
        enable_grounding:    bool = (grounding_raw[0]    if isinstance(grounding_raw,    list) else grounding_raw)    == "打开"
        enable_image_search: bool = (image_search_raw[0] if isinstance(image_search_raw, list) else image_search_raw) == "打开"

        # ----------------------------------------------------------------
        # 收集参考图：兼容两种来源
        #   1. 「加载图像（批量）」→ is_output_list=True → list[Tensor]
        #      INPUT_IS_LIST 下传入的是 list[list[Tensor]] 或 list[Tensor]，需展平
        #   2. 普通 IMAGE 端口（单 tensor 或 batch tensor）→ list 中只有 1 个元素
        # ----------------------------------------------------------------
        ref_raw = kwargs.pop("参考图", None)
        input_images: List[Image.Image] = []

        if ref_raw is not None:
            # INPUT_IS_LIST 下，可选端口若连接则为 list；元素可能是 Tensor 或 list[Tensor]
            items = ref_raw if isinstance(ref_raw, list) else [ref_raw]
            for item in items:
                if item is None:
                    continue
                if isinstance(item, list):
                    # 来自 is_output_list 的嵌套 list，继续展平
                    for sub in item:
                        if sub is not None and isinstance(sub, torch.Tensor):
                            input_images.extend(tensor_to_pil(sub))
                elif isinstance(item, torch.Tensor):
                    input_images.extend(tensor_to_pil(item))

        # ----------------------------------------------------------------
        # 以下逻辑与 NanoBananaPro.generate() 完全一致
        # ----------------------------------------------------------------
        start_time = time.time()

        pbar = None
        if PROGRESS_BAR_AVAILABLE:
            pbar = ProgressBar(生图数量)

        try:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))

            if MEMORY_MONITOR_AVAILABLE and 生图数量 > 50:
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024
                print(f"{_NODE}: 初始内存使用: {initial_memory:.1f} MB")

            if self.client is None:
                try:
                    self.client = GeminiAPIClient()
                except ValueError as e:
                    raise ValueError(f"初始化失败: {str(e)}")

            # 校验分辨率
            supported_resolutions = get_model_supported_resolutions(模型)
            if supported_resolutions and 分辨率 not in supported_resolutions:
                raise ValueError(
                    f"分辨率 \"{分辨率}\" 与模型 \"{模型}\" 不兼容！\n"
                    f"该模型支持的分辨率：{', '.join(supported_resolutions)}"
                )

            # 校验宽高比
            supported_ratios = get_model_supported_aspect_ratios(模型)
            if supported_ratios and 宽高比 not in supported_ratios:
                raise ValueError(
                    f"宽高比 \"{宽高比}\" 与模型 \"{模型}\" 不兼容！\n"
                    f"该模型支持的宽高比：{', '.join(supported_ratios)}"
                )

            # 校验图片搜索与模型兼容性
            IMAGE_SEARCH_UNSUPPORTED_MODELS = [
                "nano-banana-pro-限时特价", "nano-banana-pro-官方计费", "gemini-3-pro-image-preview"
            ]
            if enable_image_search and 模型 in IMAGE_SEARCH_UNSUPPORTED_MODELS:
                raise ValueError(
                    f"模型 \"{模型}\" 不支持【图片搜索（联网）】功能！"
                    f"请切换到 nano-banana-2-限时特价 或 gemini-3.1-flash-image-preview 后再使用"
                )

            # 验证输入图像数量上限
            if len(input_images) > 14:
                raise ValueError(
                    f"输入图像数量 {len(input_images)} 超过限制 14 张，请减少输入图像数量"
                )

            # 像素缩放
            if input_images and 像素缩放:
                input_images = [self.resize_to_megapixels(img, 分辨率像素) for img in input_images]

            # 解析批量提示词
            batch_prompts = parse_batch_prompts(prompt)

            # 打印概览
            grounding_str = ""
            if enable_image_search:
                grounding_str = " | 谷歌图片搜索接地"
            elif enable_grounding:
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

            def progress_callback(current, total, success, error_msg=None):
                nonlocal success_count, fail_count
                if success:
                    success_count += 1
                    print(f"{_NODE}: 任务 {current}/{total} 成功 ✓")
                else:
                    fail_count += 1
                    if error_msg:
                        print(f"{_NODE}: 任务 {current}/{total} 失败 ✗")
                        print(f"原始错误详情:\n{error_msg}")
                    else:
                        print(f"{_NODE}: 任务 {current}/{total} 失败 ✗")
                if pbar is not None:
                    pbar.update(1)
                if MEMORY_MONITOR_AVAILABLE and total > 50 and current % 10 == 0:
                    gc.collect()
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    print(f"{_NODE}: 内存使用: {current_memory:.1f} MB (+{memory_increase:.1f} MB)")
                    if current_memory > 2000:
                        print(f"⚠️ {_NODE}: 内存使用过高！建议减少生图数量或分批执行")

            def _get_output_folder():
                if FOLDER_PATHS_AVAILABLE:
                    folder = folder_paths.get_output_directory()
                    return folder
                raise ValueError("无法获取 ComfyUI output 目录，请检查 folder_paths 是否可用")

            def run_async_in_thread(coro_fn):
                def _run():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(coro_fn())
                    finally:
                        loop.close()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_run)
                    try:
                        return future.result(timeout=3600)
                    except TimeoutError:
                        raise RuntimeError("任务执行超时（1小时），请减少数量或检查网络连接")

            # ── 批量提示词模式 ──────────────────────────────────────
            if batch_prompts:
                num_prompts = len(batch_prompts)
                total_images = num_prompts * 生图数量
                if pbar is not None:
                    pbar = ProgressBar(total_images)

                output_folder = _get_output_folder()
                os.makedirs(output_folder, exist_ok=True)

                results = run_async_in_thread(lambda: self._process_batch_async(
                    prompts=batch_prompts,
                    model=模型,
                    resolution=分辨率,
                    aspect_ratio=宽高比,
                    images_per_prompt=生图数量,
                    input_images=input_images,
                    output_folder=output_folder,
                    pbar=pbar,
                    enable_grounding=enable_grounding,
                    enable_image_search=enable_image_search,
                ))

                success_count = sum(1 for r in results if r.get("success", False))
                fail_count = len(results) - success_count
                all_saved_files = [f for r in results for f in r.get("saved_files", [])]

                elapsed = time.time() - start_time
                time_str = f"{elapsed:.3f}s" if elapsed < 1 else f"{elapsed:.2f}s"
                print(f"完成！总耗时 {time_str} | 成功: {success_count}/{total_images} | 失败: {fail_count}")

                failed_results = [r for r in results if not r.get("success", False)]
                for fr in failed_results:
                    idx = fr.get("global_task_index", -1) + 1
                    snippet = (fr.get("prompt", "") or "")[:30]
                    print(f"  失败 #{idx}: {snippet}{'...' if len(snippet) >= 30 else ''} → {fr.get('error', '未知错误')}")

                output_images = []
                for fp in all_saved_files[-min(10, len(all_saved_files)):]:
                    try:
                        output_images.append(Image.open(fp))
                    except Exception as e:
                        print(f"{_NODE}: 无法加载 {fp} - {e}")

                if not output_images:
                    output_images = [Image.new('RGB', (512, 512), color=(128, 128, 128))]

                output_tensor = _images_to_tensor_safe(output_images, _NODE)
                print(f"{_NODE}: 共保存 {len(all_saved_files)} 张图片到磁盘，节点输出最后 {len(output_images)} 张")
                gc.collect()
                return (output_tensor,)

            # ── 单提示词模式 ────────────────────────────────────────
            if 生图数量 == 1:
                generated_images = self.client.generate_sync(
                    prompt=prompt,
                    model=模型,
                    resolution=分辨率,
                    aspect_ratio=宽高比,
                    batch_size=1,
                    images=input_images,
                    progress_callback=progress_callback,
                    debug=DEBUG_LOG_ENABLED,
                    debug_request=REQUEST_LOG_ENABLED,
                    enable_grounding=enable_grounding,
                    enable_image_search=enable_image_search,
                )
                output_folder = _get_output_folder()
                os.makedirs(output_folder, exist_ok=True)
                for gen_img in generated_images:
                    output_path = generate_timestamp_filename(output_folder=output_folder)
                    save_image(gen_img, output_path)
            else:
                print(f"{_NODE}: 单提示词×{生图数量}张 → 异步并发模式")
                if pbar is not None:
                    pbar = ProgressBar(生图数量)

                output_folder = _get_output_folder()
                os.makedirs(output_folder, exist_ok=True)

                results = run_async_in_thread(lambda: self._process_batch_async(
                    prompts=[prompt],
                    model=模型,
                    resolution=分辨率,
                    aspect_ratio=宽高比,
                    images_per_prompt=生图数量,
                    input_images=input_images,
                    output_folder=output_folder,
                    pbar=pbar,
                    enable_grounding=enable_grounding,
                    enable_image_search=enable_image_search,
                ))

                success_count = sum(1 for r in results if r.get("success", False))
                fail_count = len(results) - success_count
                all_saved_files = [f for r in results for f in r.get("saved_files", [])]

                elapsed = time.time() - start_time
                time_str = f"{elapsed:.3f}s" if elapsed < 1 else f"{elapsed:.2f}s"
                print(f"完成！总耗时 {time_str} | 成功: {success_count}/{生图数量} | 失败: {fail_count}")

                failed_results = [r for r in results if not r.get("success", False)]
                for fr in failed_results:
                    idx = fr.get("global_task_index", -1) + 1
                    print(f"  失败 #{idx}: {prompt[:30]}{'...' if len(prompt) >= 30 else ''} → {fr.get('error', '未知错误')}")

                output_images = []
                for fp in all_saved_files[-min(10, len(all_saved_files)):]:
                    try:
                        output_images.append(Image.open(fp))
                    except Exception as e:
                        print(f"{_NODE}: 无法加载 {fp} - {e}")

                if not output_images:
                    output_images = [Image.new('RGB', (512, 512), color=(128, 128, 128))]

                output_tensor = _images_to_tensor_safe(output_images, _NODE)
                print(f"{_NODE}: 共保存 {len(all_saved_files)} 张图片到磁盘，节点输出最后 {len(output_images)} 张")
                gc.collect()
                return (output_tensor,)

            # 单张同步模式的输出路径（生图数量==1 走到这里）
            max_output_images = 20
            if len(generated_images) > max_output_images:
                print(f"{_NODE}: 生成 {len(generated_images)} 张图片，限制输出前 {max_output_images} 张到ComfyUI")
                output_images = generated_images[:max_output_images]
            else:
                output_images = generated_images

            output_tensor = _images_to_tensor_safe(output_images, _NODE)
            elapsed = time.time() - start_time
            time_str = f"{elapsed:.3f}s" if elapsed < 1 else f"{elapsed:.2f}s"
            if fail_count > 0:
                print(f"[4/4] 完成！总耗时 {time_str} | 成功 {success_count}张 | 失败 {fail_count}张")
            else:
                print(f"[4/4] 完成！总耗时 {time_str} | 成功 {len(generated_images)}张")

            gc.collect()
            return (output_tensor,)

        except ValueError as e:
            if str(e) == "未授权！":
                print("请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            error_msg = str(e)
            print(f"{_NODE}: ❌ {error_msg}")
            raise ValueError(error_msg) from None

        except RuntimeError as e:
            error_full = str(e)
            print(f"{_NODE}: ❌ {error_full}")
            raise RuntimeError(error_full) from None

        except Exception as e:
            error_msg = str(e)
            print(f"{_NODE}: ❌ {error_msg}")
            raise type(e)(error_msg) from None

        finally:
            if self.client is not None:
                try:
                    balance_data = self.client.query_balance_sync()
                    balance_info = self.client.format_balance_info(balance_data)
                    print(f"{_NODE}: {balance_info}")
                except Exception:
                    pass
            gc.collect()
