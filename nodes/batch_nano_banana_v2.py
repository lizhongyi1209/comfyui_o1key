"""
批量 Nano Banana v2 节点
BatchNanoBananaPro 的完全复刻，唯一改动：

  将原来 9 个独立「参考图1~9」输入端
  改为 1 个「参考图」输入端（可选），配合「加载图像（批量）」节点使用。

  「加载图像（批量）」输出 is_output_list=True（list[Tensor]），
  本节点声明 INPUT_IS_LIST = True 来整体接收该列表，
  然后在 process_batch() 开头对所有参数统一解包，其余业务逻辑与原节点完全一致。
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
from PIL import Image

import torch
import numpy as np

from ..utils.image_utils import tensor_to_pil, pil_to_tensor, parse_batch_prompts
from ..utils.file_utils import (
    ImageInfo,
    load_images_from_folder,
    pair_images_by_name,
    pair_images_cartesian,
    generate_timestamp_filename,
    save_image,
)
from ..clients.gemini_client import GeminiAPIClient
from ..models_config import (
    get_enabled_models,
    get_model_supported_aspect_ratios, get_all_supported_aspect_ratios,
    get_model_supported_resolutions, get_all_supported_resolutions
)

try:
    from comfy.utils import ProgressBar
    PROGRESS_BAR_AVAILABLE = True
except ImportError:
    PROGRESS_BAR_AVAILABLE = False

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

try:
    import psutil
    MEMORY_MONITOR_AVAILABLE = True
except ImportError:
    MEMORY_MONITOR_AVAILABLE = False

DEBUG_LOG_ENABLED = False
REQUEST_LOG_ENABLED = False

_NODE = "BatchNanoBananaV2"


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


class BatchNanaBananaV2:
    """
    批量 Nano Banana v2

    与 BatchNanoBananaPro 完全一致，参考图输入方式不同：
    - 原版：9 个独立可选端口（参考图1~9）
    - v2：1 个可选端口「参考图」，配合「加载图像（批量）」可传入任意数量图片
    """

    ASPECT_RATIOS = [
        "1:1", "4:3", "3:4", "16:9", "9:16",
        "2:3", "3:2", "4:5", "5:4", "21:9",
        "1:4", "4:1", "1:8", "8:1"
    ]
    RESOLUTIONS = ["512", "1K", "2K", "4K"]
    PAIRING_MODES = ["按相同图片命名", "1*N", "不配对"]

    def __init__(self):
        self.client = None

    def resize_to_megapixels(self, image: Image.Image, target_megapixels: float) -> Image.Image:
        current_pixels = image.width * image.height
        target_pixels = int(target_megapixels * 1_000_000)
        if abs(current_pixels - target_pixels) / target_pixels < 0.05:
            return image
        scale = (target_pixels / current_pixels) ** 0.5
        new_width = max(1, int(image.width * scale))
        new_height = max(1, int(image.height * scale))
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    @classmethod
    def INPUT_TYPES(cls):
        enabled_models = get_enabled_models()
        if not enabled_models:
            enabled_models = ["请在 models_config.py 中启用至少一个模型"]

        all_aspect_ratios = get_all_supported_aspect_ratios() or cls.ASPECT_RATIOS
        all_resolutions = get_all_supported_resolutions() or cls.RESOLUTIONS

        return {
            "required": {
                "prompt": ("STRING", {"default": "一个中国女子的OOTD", "multiline": True}),
                "模型": (enabled_models, {"default": enabled_models[0]}),
                "宽高比": (all_aspect_ratios, {"default": "1:1"}),
                "分辨率": (all_resolutions, {"default": "2K"}),
                "像素缩放": ("BOOLEAN", {"default": False, "label_on": "打开", "label_off": "关闭"}),
                "分辨率像素": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1, "display": "number"}),
                "谷歌搜索（联网）": (["关闭", "打开"], {"default": "关闭"}),
                "图片搜索（联网）": (["关闭", "打开"], {"default": "关闭"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "文件夹1": ("STRING", {"default": "", "multiline": False}),
                "文件夹2": ("STRING", {"default": "", "multiline": False}),
                "文件夹3": ("STRING", {"default": "", "multiline": False}),
                "文件夹4": ("STRING", {"default": "", "multiline": False}),
                "文件夹5": ("STRING", {"default": "", "multiline": False}),
                "文件夹6": ("STRING", {"default": "", "multiline": False}),
                "文件夹7": ("STRING", {"default": "", "multiline": False}),
                "文件夹8": ("STRING", {"default": "", "multiline": False}),
                "文件夹9": ("STRING", {"default": "", "multiline": False}),
                "保存路径": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                # 单个参考图端口，接受普通 IMAGE 或「加载图像（批量）」输出的列表
                "参考图": ("IMAGE",),
                "图片配对模式": (cls.PAIRING_MODES, {"default": "不配对"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "process_batch"
    CATEGORY = "image/batch"

    # 声明 INPUT_IS_LIST，使 ComfyUI 将「加载图像（批量）」的 list[Tensor]
    # 整体传入而非逐张迭代执行，同时其余所有参数也会被包进 list，需解包。
    INPUT_IS_LIST = True

    # ------------------------------------------------------------------ #
    #  以下方法与 BatchNanoBananaPro 完全相同
    # ------------------------------------------------------------------ #

    def _load_folders(
        self,
        folder1, folder2, folder3, folder4,
        enable_scaling, target_megapixels,
        folder5=None, folder6=None, folder7=None, folder8=None, folder9=None,
    ) -> List[List[ImageInfo]]:
        folders = [folder1, folder2, folder3, folder4,
                   folder5, folder6, folder7, folder8, folder9]
        all_images = []
        for i, folder in enumerate(folders, 1):
            if folder and folder.strip():
                try:
                    images = load_images_from_folder(folder)
                    if images:
                        if enable_scaling:
                            scaled = []
                            for info in images:
                                scaled_img = self.resize_to_megapixels(info.image, target_megapixels)
                                scaled.append(ImageInfo(
                                    image=scaled_img,
                                    filename=info.filename,
                                    extension=info.extension,
                                    source_path=info.source_path
                                ))
                            images = scaled
                        all_images.append(images)
                except ValueError as e:
                    print(f"{_NODE}: 文件夹{i} 加载失败 - {e}")
        return all_images

    def _create_pairs(
        self,
        image_lists: List[List[ImageInfo]],
        pairing_mode: str,
        manual_images: Optional[List[ImageInfo]] = None
    ) -> List[Tuple[ImageInfo, ...]]:
        if pairing_mode == "不配对":
            if len(image_lists) > 1:
                raise ValueError("「不配对」模式只支持单个文件夹，请清空其他文件夹路径")
            if image_lists and manual_images:
                return [(img,) + tuple(manual_images) for img in image_lists[0]]
            elif image_lists:
                return [(img,) for img in image_lists[0]]
            else:
                return []

        if not image_lists:
            return []

        if len(image_lists) == 1:
            base_pairs = [(img,) for img in image_lists[0]]
        elif pairing_mode == "按相同图片命名":
            base_pairs = list(pair_images_by_name(*image_lists))
        else:
            base_pairs = list(pair_images_cartesian(*image_lists))

        if manual_images:
            manual_tuple = tuple(manual_images)
            base_pairs = [pair + manual_tuple for pair in base_pairs]

        return base_pairs

    async def _generate_single_task(
        self,
        client: GeminiAPIClient,
        session: aiohttp.ClientSession,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        images: List[ImageInfo],
        output_folder: str,
        task_index: int,
        enable_grounding: bool = True,
        enable_image_search: bool = False,
        base_filename: str = None,
    ) -> dict:
        result = {
            "task_index": task_index,
            "prompt": prompt,
            "success": False,
            "generated_count": 0,
            "saved_files": [],
            "output_images": [],
            "error": None
        }
        try:
            input_pil_images = [info.image for info in images]
            generated_images = []
            try:
                gen_result = await client.generate_single_async(
                    prompt=prompt,
                    model=model,
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    images=input_pil_images,
                    session=session,
                    debug=DEBUG_LOG_ENABLED,
                    debug_request=REQUEST_LOG_ENABLED,
                    enable_grounding=enable_grounding,
                    enable_image_search=enable_image_search,
                )
                if gen_result:
                    images_list, timing_info = gen_result
                    generated_images.extend(images_list)
            except Exception as e:
                import traceback
                error_msg = str(e)
                error_traceback = traceback.format_exc()
                print(f"=" * 80)
                print(f"🔍 【原始报错信息展示】")
                print(f"=" * 80)
                print(f"任务编号: {task_index + 1}")
                print(f"失败时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"模型: {model}")
                print(f"分辨率: {resolution}")
                print(f"宽高比: {aspect_ratio}")
                print(f"-" * 80)
                print(f"错误信息: {error_msg}")
                print(f"-" * 80)
                print(f"完整堆栈追踪:")
                print(error_traceback)
                print(f"=" * 80)
                result["error"] = error_msg

            for i, gen_img in enumerate(generated_images):
                if base_filename:
                    base_name = base_filename
                    counter = 0
                    while True:
                        filename = f"{base_name}.png" if counter == 0 else f"{base_name}+{counter}.png"
                        output_path = os.path.join(output_folder, filename)
                        if not os.path.exists(output_path):
                            break
                        counter += 1
                else:
                    output_path = generate_timestamp_filename(
                        output_folder=output_folder, extension=".png"
                    )
                save_image(gen_img, output_path)
                result["saved_files"].append(output_path)
                gen_img = None

            if len(generated_images) > 0:
                result["success"] = True
            result["generated_count"] = len(generated_images)

        except Exception as e:
            result["error"] = str(e)

        return result

    async def _process_batch_async(
        self,
        pairs: List[Tuple[ImageInfo, ...]],
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        output_folder: str,
        pbar=None,
        prompts_per_task: Optional[List[str]] = None,
        enable_grounding: bool = True,
        enable_image_search: bool = False,
    ) -> List[dict]:
        if self.client is None:
            self.client = GeminiAPIClient()

        total_tasks = len(pairs)
        max_concurrent = 10

        print(f"{_NODE}: 检测到 {total_tasks} 个任务")

        all_results = []
        completed = 0
        success_count = 0
        fail_count = 0
        num_batches = math.ceil(total_tasks / max_concurrent)

        if MEMORY_MONITOR_AVAILABLE and total_tasks > 50:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            print(f"{_NODE}: 初始内存使用: {initial_memory:.1f} MB")

        show_milestone = total_tasks >= 50
        milestones = [0.2, 0.4, 0.6, 0.8, 1.0]
        milestone_index = 0

        if num_batches > 1:
            print(f"{_NODE}: 任务数 {total_tasks} 超过并发上限 {max_concurrent}，将分 {num_batches} 批执行")

        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        async with aiohttp.ClientSession(connector=connector) as session:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * max_concurrent
                end_idx = min(start_idx + max_concurrent, total_tasks)
                batch_pairs = pairs[start_idx:end_idx]

                if num_batches > 1:
                    print(f"{_NODE}: 执行第 {batch_idx + 1}/{num_batches} 批 ({start_idx + 1}-{end_idx})...")

                tasks = []
                for i, pair in enumerate(batch_pairs):
                    task_prompt = prompts_per_task[start_idx + i] if prompts_per_task else prompt
                    base_filename = None
                    if pair and len(pair) > 0:
                        first_image = pair[0]
                        if hasattr(first_image, 'filename'):
                            base_filename = first_image.filename

                    task = asyncio.create_task(
                        self._generate_single_task(
                            client=self.client,
                            session=session,
                            prompt=task_prompt,
                            model=model,
                            resolution=resolution,
                            aspect_ratio=aspect_ratio,
                            images=list(pair),
                            output_folder=output_folder,
                            task_index=start_idx + i,
                            enable_grounding=enable_grounding,
                            enable_image_search=enable_image_search,
                            base_filename=base_filename,
                        )
                    )
                    tasks.append(task)

                batch_results = []
                for coro in asyncio.as_completed(tasks):
                    result_data = None
                    try:
                        result = await coro
                        if isinstance(result, Exception):
                            result_data = {"success": False, "error": str(result), "generated_count": 0, "saved_files": []}
                            batch_results.append(result_data)
                        else:
                            result_data = result
                            batch_results.append(result)
                    except Exception as e:
                        result_data = {"success": False, "error": str(e), "generated_count": 0, "saved_files": []}
                        batch_results.append(result_data)

                    completed += 1
                    if result_data and result_data.get("success", False):
                        success_count += 1
                        print(f"{_NODE}: 任务 {completed}/{total_tasks} 成功 ✓")
                    else:
                        fail_count += 1
                        error_msg = result_data.get("error", "未知错误") if result_data else "未知错误"
                        print(f"{_NODE}: 任务 {completed}/{total_tasks} 失败 ✗")
                        print(f"=" * 80)
                        print(f"🔍 【原始报错信息展示】")
                        print(f"=" * 80)
                        print(f"任务编号: {completed}/{total_tasks}")
                        print(f"失败时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"-" * 80)
                        print(f"错误详情:")
                        print(error_msg)
                        print(f"=" * 80)

                    if pbar is not None:
                        pbar.update(1)

                    if show_milestone and milestone_index < len(milestones):
                        progress = completed / total_tasks
                        if progress >= milestones[milestone_index]:
                            percentage = int(milestones[milestone_index] * 100)
                            print(f"{_NODE}: >>> 进度 {percentage}% <<<")
                            milestone_index += 1

                all_results.extend(batch_results)
                print(f"{_NODE}: 第 {batch_idx + 1} 批完成，开始分批保存...")

                batch_success = sum(1 for r in batch_results if r.get("success", False))
                batch_fail = len(batch_results) - batch_success
                batch_generated = sum(r.get("generated_count", 0) for r in batch_results)
                print(f"{_NODE}: 本批结果 - 成功: {batch_success}/{len(batch_results)}，生成: {batch_generated} 张")

                gc.collect()

                if MEMORY_MONITOR_AVAILABLE and total_tasks > 50:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    print(f"{_NODE}: 内存使用: {current_memory:.1f} MB (+{memory_increase:.1f} MB)")
                    if current_memory > 2000:
                        print(f"⚠️ {_NODE}: 内存使用过高！但图片已分批保存，即使崩溃也不会丢失已完成的任务")

                await asyncio.sleep(0.5)

        return all_results

    def process_batch(
        self,
        prompt,
        文件夹1, 文件夹2, 文件夹3, 文件夹4,
        文件夹5, 文件夹6, 文件夹7, 文件夹8, 文件夹9,
        像素缩放,
        分辨率像素,
        seed,
        模型,
        宽高比,
        分辨率,
        保存路径,
        **kwargs
    ) -> Tuple[torch.Tensor]:

        # ----------------------------------------------------------------
        # INPUT_IS_LIST=True 时，所有参数均为 list，先统一解包为标量
        # ----------------------------------------------------------------
        def _unpack(v):
            return v[0] if isinstance(v, list) else v

        prompt       = _unpack(prompt)
        文件夹1      = _unpack(文件夹1)
        文件夹2      = _unpack(文件夹2)
        文件夹3      = _unpack(文件夹3)
        文件夹4      = _unpack(文件夹4)
        文件夹5      = _unpack(文件夹5)
        文件夹6      = _unpack(文件夹6)
        文件夹7      = _unpack(文件夹7)
        文件夹8      = _unpack(文件夹8)
        文件夹9      = _unpack(文件夹9)
        像素缩放     = _unpack(像素缩放)
        分辨率像素   = _unpack(分辨率像素)
        seed         = _unpack(seed)
        模型         = _unpack(模型)
        宽高比       = _unpack(宽高比)
        分辨率       = _unpack(分辨率)
        保存路径     = _unpack(保存路径)

        # 含全角括号的参数名无法作为形参，从 kwargs 中提取
        enable_grounding:    bool = (_unpack(kwargs.pop("谷歌搜索（联网）", "关闭"))) == "打开"
        enable_image_search: bool = (_unpack(kwargs.pop("图片搜索（联网）", "关闭"))) == "打开"

        # 图片配对模式（可选参数）
        图片配对模式 = _unpack(kwargs.pop("图片配对模式", "不配对"))

        # ----------------------------------------------------------------
        # 收集参考图：兼容两种来源
        #   1. 「加载图像（批量）」→ is_output_list=True → list[Tensor]
        #      INPUT_IS_LIST 下传入的是 list[list[Tensor]] 或 list[Tensor]，需展平
        #   2. 普通 IMAGE 端口（单 tensor 或 batch tensor）→ list 中只有 1 个元素
        # ----------------------------------------------------------------
        ref_raw = kwargs.pop("参考图", None)
        manual_images: List[ImageInfo] = []

        if ref_raw is not None:
            items = ref_raw if isinstance(ref_raw, list) else [ref_raw]
            idx = 0
            for item in items:
                if item is None:
                    continue
                if isinstance(item, list):
                    sub_tensors = item
                elif isinstance(item, torch.Tensor):
                    sub_tensors = [item]
                else:
                    continue
                for tensor in sub_tensors:
                    if tensor is None or not isinstance(tensor, torch.Tensor):
                        continue
                    pil_images = tensor_to_pil(tensor)
                    for j, img in enumerate(pil_images):
                        if 像素缩放:
                            img = self.resize_to_megapixels(img, 分辨率像素)
                        manual_images.append(ImageInfo(
                            image=img,
                            filename=f"manual_{idx}_{j}",
                            extension=".png",
                            source_path=""
                        ))
                    idx += 1

        # ----------------------------------------------------------------
        # 以下逻辑与 BatchNanoBananaPro.process_batch() 完全一致
        # ----------------------------------------------------------------
        start_time = time.time()

        try:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))

            has_any_folder = any(
                f and f.strip()
                for f in [文件夹1, 文件夹2, 文件夹3, 文件夹4,
                           文件夹5, 文件夹6, 文件夹7, 文件夹8, 文件夹9]
            )
            if not has_any_folder:
                raise ValueError("请至少填写一个文件夹路径，该节点专为批量文件夹处理设计")

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

            IMAGE_SEARCH_UNSUPPORTED_MODELS = [
                "nano-banana-pro-限时特价", "nano-banana-pro-官方计费", "gemini-3-pro-image-preview"
            ]
            if enable_image_search and 模型 in IMAGE_SEARCH_UNSUPPORTED_MODELS:
                raise ValueError(
                    f"模型 \"{模型}\" 不支持【图片搜索（联网）】功能！"
                    f"请切换到 nano-banana-2-限时特价 或 gemini-3.1-flash-image-preview 后再使用"
                )

            print(f"{_NODE}: 开始加载图片...")
            image_lists = self._load_folders(
                文件夹1, 文件夹2, 文件夹3, 文件夹4,
                像素缩放, 分辨率像素,
                文件夹5, 文件夹6, 文件夹7, 文件夹8, 文件夹9
            )

            total_folder_images = sum(len(lst) for lst in image_lists)
            if total_folder_images == 0:
                raise ValueError("文件夹中未找到任何图片，请检查文件夹路径是否正确")

            pairs = self._create_pairs(image_lists, 图片配对模式, manual_images if manual_images else None)

            if not pairs:
                raise ValueError("配对结果为空，请检查输入")

            batch_prompts = parse_batch_prompts(prompt)
            prompts_per_task = None
            if batch_prompts:
                expanded_pairs = []
                expanded_prompts = []
                for pair in pairs:
                    for bp in batch_prompts:
                        expanded_pairs.append(pair)
                        expanded_prompts.append(bp)
                pairs = expanded_pairs
                prompts_per_task = expanded_prompts

            total_tasks = len(pairs)

            grounding_str = ""
            if enable_image_search:
                grounding_str = " | 谷歌图片搜索接地"
            elif enable_grounding:
                grounding_str = " | 谷歌搜索接地"

            if batch_prompts:
                print(f"{_NODE}: 批量任务 | {图片配对模式} 配对模式 × {len(batch_prompts)}个提示词 | 共 {total_tasks} 任务{grounding_str}")
            else:
                print(f"{_NODE}: 批量任务 | {图片配对模式} 配对模式 | 共 {total_tasks} 任务{grounding_str}")

            pbar = None
            if PROGRESS_BAR_AVAILABLE:
                pbar = ProgressBar(total_tasks)

            has_save_path = bool(保存路径 and 保存路径.strip())
            if not has_save_path:
                if FOLDER_PATHS_AVAILABLE:
                    保存路径 = folder_paths.get_output_directory()
                    has_save_path = True
                    print(f"{_NODE}: 未设置保存路径，将使用 ComfyUI 默认 output 目录: {保存路径}")
                else:
                    print(f"{_NODE}: 未设置保存路径，图片将输出到节点")

            if has_save_path:
                try:
                    os.makedirs(保存路径, exist_ok=True)
                    test_file = os.path.join(保存路径, ".write_test")
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                    print(f"{_NODE}: 保存路径验证通过: {保存路径}")
                except Exception as e:
                    raise ValueError(f"保存路径无效或无写入权限: {保存路径} - {str(e)}")

            if self.client is None:
                try:
                    self.client = GeminiAPIClient()
                except ValueError as e:
                    raise ValueError(f"初始化 API 客户端失败: {str(e)}")

            def run_async_in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self._process_batch_async(
                            pairs=pairs,
                            prompt=prompt,
                            model=模型,
                            resolution=分辨率,
                            aspect_ratio=宽高比,
                            output_folder=保存路径,
                            pbar=pbar,
                            prompts_per_task=prompts_per_task,
                            enable_grounding=enable_grounding,
                            enable_image_search=enable_image_search,
                        )
                    )
                except Exception as e:
                    print(f"{_NODE}: 异步任务执行异常: {str(e)}")
                    raise
                finally:
                    loop.close()

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_async_in_thread)
                try:
                    results = future.result(timeout=3600)
                except TimeoutError:
                    print(f"{_NODE}: 任务执行超时（1小时）")
                    raise RuntimeError("任务执行超时，请减少任务数量或检查网络连接")
                except Exception as e:
                    print(f"{_NODE}: 任务执行失败: {str(e)}")
                    raise

            success_count = sum(1 for r in results if r.get("success", False))
            fail_count = len(results) - success_count
            total_generated = sum(r.get("generated_count", 0) for r in results)
            all_saved_files = [f for r in results for f in r.get("saved_files", [])]

            elapsed = time.time() - start_time
            time_str = f"{elapsed:.3f}s" if elapsed < 1 else f"{elapsed:.2f}s"
            avg_time = elapsed / success_count if success_count > 0 else 0
            avg_time_str = f"{avg_time:.1f}s/张" if success_count > 0 else "N/A"

            print("=" * 60)
            print(f"完成！总耗时 {time_str} | 成功: {success_count}/{total_tasks} | 生成 {total_generated} 张 | 平均 {avg_time_str}")
            if has_save_path:
                print(f"保存路径: {保存路径}")
            else:
                print("保存路径: 未设置（仅输出到节点）")

            failed_results = [r for r in results if not r.get("success", False)]
            if failed_results:
                print(f"-" * 60)
                print(f"❌ 失败任务汇总: {len(failed_results)} 个")
                print(f"-" * 60)
                for idx, failed in enumerate(failed_results[:3], 1):
                    task_num = failed.get('task_index', '?') + 1
                    error_msg = failed.get('error', '未知错误')
                    print(f"\n【失败任务 #{task_num}】")
                    print(f"错误信息: {error_msg}")
                if len(failed_results) > 3:
                    remaining = [str(r.get('task_index', '?') + 1) for r in failed_results[3:]]
                    print(f"\n其他失败任务编号: {', '.join(remaining)}")
                print(f"-" * 60)

            output_images = []
            if all_saved_files:
                for fp in all_saved_files[-min(10, len(all_saved_files)):]:
                    try:
                        output_images.append(Image.open(fp))
                    except Exception as e:
                        print(f"{_NODE}: 无法加载图片 {fp} - {e}")

            if not output_images:
                output_images = [Image.new('RGB', (512, 512), color=(128, 128, 128))]

            output_tensor = _images_to_tensor_safe(output_images, _NODE)
            gc.collect()

            total_saved = len(all_saved_files)
            print(f"{_NODE}: 任务完成！共保存 {total_saved} 张图片到磁盘")
            if total_saved > 0:
                print(f"{_NODE}: 最新保存的文件: {all_saved_files[-1]}")


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
                    print("=" * 60)
                except Exception:
                    pass
            gc.collect()
