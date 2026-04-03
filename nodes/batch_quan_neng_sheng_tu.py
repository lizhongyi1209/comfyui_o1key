"""
全能生图（批量）节点
ComfyUI 自定义节点，用于批量处理图像生成任务
支持多文件夹加载、1:1/笛卡尔积配对、智能命名保存
"""

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
from ..clients.openai_client import OpenAIAPIClient
from ..models_config import (
    get_enabled_models,
    get_model_supported_aspect_ratios, get_all_supported_aspect_ratios,
    get_model_supported_resolutions, get_all_supported_resolutions
)

# 导入 ComfyUI 原生进度条
try:
    from comfy.utils import ProgressBar
    PROGRESS_BAR_AVAILABLE = True
except ImportError:
    PROGRESS_BAR_AVAILABLE = False
    print("⚠️ 全能生图（批量）: comfy.utils.ProgressBar 不可用，将只使用终端进度显示")

# 导入 ComfyUI 的文件夹路径管理
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False
    print("⚠️ 全能生图（批量）: folder_paths 不可用，将无法使用默认保存路径")

# 内存监控（可选）
try:
    import psutil
    MEMORY_MONITOR_AVAILABLE = True
except ImportError:
    MEMORY_MONITOR_AVAILABLE = False
    print("⚠️ 全能生图（批量）: psutil 不可用，内存监控功能禁用")

# ============================================================================
# 调试日志配置
# ============================================================================
DEBUG_LOG_ENABLED = False
REQUEST_LOG_ENABLED = False
# ============================================================================


class BatchQuanNengShengTu:
    """
    全能生图（批量）节点

    功能：
    - 从多个文件夹加载图片
    - 支持三种配对模式：
      * 按相同图片命名 - 索引配对（文件夹之间按位置配对）
      * 1*N - 笛卡尔积配对（所有可能组合）
      * 不配对 - 固定参考图模式（文件夹图片依次与所有参考图组合）
    - 批量调用 API 生成图像
    - 智能命名保存（保留原始文件名）
    - 并发控制（默认最大 10）

    注意：
    - 「不配对」模式只支持单个文件夹
    - 支持的模型列表从 models_config.py 动态加载
    """

    MODELS = None
    ASPECT_RATIOS = [
        "1:1", "4:3", "3:4", "16:9", "9:16",
        "2:3", "3:2", "4:5", "5:4", "21:9",
        "1:4", "4:1", "1:8", "8:1"
    ]
    RESOLUTIONS = ["512", "1K", "2K", "4K"]
    PAIRING_MODES = ["按相同图片命名", "1*N", "不配对"]

    def __init__(self):
        """初始化节点"""
        self.client = None

    def resize_to_megapixels(
        self,
        image: Image.Image,
        target_megapixels: float
    ) -> Image.Image:
        """将图像缩放到指定的总像素数，保持纵横比"""
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
        """定义输入参数"""
        enabled_models = get_enabled_models()
        enabled_models = [m for m in enabled_models if "限时特价" not in m]

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

        optional_inputs["图片配对模式"] = (cls.PAIRING_MODES, {
            "default": "不配对"
        })

        return {
            "required": {
                "提示词": ("STRING", {
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
                "像素缩放": ("BOOLEAN", {
                    "default": False,
                    "label_on": "打开",
                    "label_off": "关闭"
                }),
                "分辨率像素": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "文件夹1": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "文件夹2": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "文件夹3": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "文件夹4": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "文件夹5": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "文件夹6": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "文件夹7": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "文件夹8": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "文件夹9": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "保存路径": ("STRING", {
                    "default": "",
                    "multiline": False
                })
            },
            "optional": optional_inputs
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "process_batch"
    CATEGORY = "image/batch"

    def _load_folders(
        self,
        folder1: str,
        folder2: Optional[str],
        folder3: Optional[str],
        folder4: Optional[str],
        enable_scaling: bool,
        target_megapixels: float,
        folder5: Optional[str] = None,
        folder6: Optional[str] = None,
        folder7: Optional[str] = None,
        folder8: Optional[str] = None,
        folder9: Optional[str] = None,
    ) -> List[List[ImageInfo]]:
        """加载所有文件夹中的图片"""
        folders = [folder1, folder2, folder3, folder4, folder5, folder6, folder7, folder8, folder9]
        all_images = []

        for i, folder in enumerate(folders, 1):
            if folder and folder.strip():
                try:
                    images = load_images_from_folder(folder)
                    if images:
                        if enable_scaling:
                            scaled_images = []
                            for img_info in images:
                                scaled_img = self.resize_to_megapixels(
                                    img_info.image,
                                    target_megapixels
                                )
                                scaled_info = ImageInfo(
                                    image=scaled_img,
                                    filename=img_info.filename,
                                    extension=img_info.extension,
                                    source_path=img_info.source_path
                                )
                                scaled_images.append(scaled_info)
                            images = scaled_images
                        all_images.append(images)
                except ValueError as e:
                    print(f"全能生图（批量）: 文件夹{i} 加载失败 - {e}")

        return all_images

    def _create_pairs(
        self,
        image_lists: List[List[ImageInfo]],
        pairing_mode: str,
        manual_images: Optional[List[ImageInfo]] = None
    ) -> List[Tuple[ImageInfo, ...]]:
        """根据配对模式创建图片组合"""
        if pairing_mode == "不配对":
            if len(image_lists) > 1:
                raise ValueError("「不配对」模式只支持单个文件夹，请清空其他文件夹路径")

            if image_lists and manual_images:
                folder_images = image_lists[0]
                pairs = []
                for img in folder_images:
                    pair = (img,) + tuple(manual_images)
                    pairs.append(pair)
                return pairs
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
        session: aiohttp.ClientSession,
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        images: List[ImageInfo],
        output_folder: str,
        task_index: int,
        base_filename: str = None,
    ) -> dict:
        """执行单个生成任务"""
        result = {
            "task_index": task_index,
            "prompt": prompt,
            "success": False,
            "generated_count": 0,
            "saved_files": [],
            "error": None
        }

        try:
            input_pil_images = [info.image for info in images]

            gen_result = await self.client.generate_single_async(
                prompt=prompt,
                model=model,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                images=input_pil_images,
                session=session,
                debug=DEBUG_LOG_ENABLED,
                debug_request=REQUEST_LOG_ENABLED,
                enable_grounding=False,
                enable_image_search=False
            )

            if gen_result:
                images_list, _ = gen_result

                import os
                for gen_img in images_list:
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
        pairs: List[Tuple[ImageInfo, ...]],
        prompt: str,
        model: str,
        resolution: str,
        aspect_ratio: str,
        output_folder: str,
        pbar=None,
        prompts_per_task: Optional[List[str]] = None,
    ) -> List[dict]:
        """异步批量处理所有任务"""
        if self.client is None:
            self.client = OpenAIAPIClient()

        total_tasks = len(pairs)
        max_concurrent = 10

        print(f"全能生图（批量）: 检测到 {total_tasks} 个任务")

        all_results = []
        completed = 0
        success_count = 0
        fail_count = 0

        num_batches = math.ceil(total_tasks / max_concurrent)

        if num_batches > 1:
            print(f"全能生图（批量）: 任务数 {total_tasks} 超过并发上限 {max_concurrent}，将分 {num_batches} 批执行")

        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)

        async with aiohttp.ClientSession(connector=connector) as session:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * max_concurrent
                end_idx = min(start_idx + max_concurrent, total_tasks)
                batch_pairs = pairs[start_idx:end_idx]

                if num_batches > 1:
                    print(f"全能生图（批量）: 执行第 {batch_idx + 1}/{num_batches} 批 ({start_idx + 1}-{end_idx})...")

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
                            session=session,
                            prompt=task_prompt,
                            model=model,
                            resolution=resolution,
                            aspect_ratio=aspect_ratio,
                            images=list(pair),
                            output_folder=output_folder,
                            task_index=start_idx + i,
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
                        else:
                            result_data = result
                        batch_results.append(result_data)
                    except Exception as e:
                        result_data = {"success": False, "error": str(e), "generated_count": 0, "saved_files": []}
                        batch_results.append(result_data)

                    completed += 1

                    if result_data and result_data.get("success", False):
                        success_count += 1
                        print(f"全能生图（批量）: 任务 {completed}/{total_tasks} 成功 ✓")
                    else:
                        fail_count += 1
                        error_msg = result_data.get("error", "未知错误") if result_data else "未知错误"
                        print(f"全能生图（批量）: 任务 {completed}/{total_tasks} 失败 ✗ - {error_msg}")

                    if pbar is not None:
                        pbar.update(1)

                all_results.extend(batch_results)

                import gc
                gc.collect()

                await asyncio.sleep(0.1)

        return all_results

    def process_batch(
        self,
        提示词: str,
        模型: str,
        宽高比: str,
        分辨率: str,
        像素缩放: bool,
        分辨率像素: float,
        seed: int,
        文件夹1: str,
        文件夹2: str,
        文件夹3: str,
        文件夹4: str,
        文件夹5: str,
        文件夹6: str,
        文件夹7: str,
        文件夹8: str,
        文件夹9: str,
        保存路径: str,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """批量处理图像生成"""
        start_time = time.time()

        try:
            random.seed(seed)
            np.random.seed(seed % (2**32))

            if self.client is None:
                self.client = OpenAIAPIClient()

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

            manual_images = []
            for i in range(1, 10):
                key = f"参考图{i}"
                if key in kwargs and kwargs[key] is not None:
                    pil_imgs = tensor_to_pil(kwargs[key])
                    for pil_img in pil_imgs:
                        manual_images.append(ImageInfo(
                            image=pil_img,
                            filename=f"manual_{i}",
                            extension=".png",
                            source_path=""
                        ))

            if 像素缩放 and manual_images:
                scaled_manual = []
                for img_info in manual_images:
                    scaled_img = self.resize_to_megapixels(img_info.image, 分辨率像素)
                    scaled_manual.append(ImageInfo(
                        image=scaled_img,
                        filename=img_info.filename,
                        extension=img_info.extension,
                        source_path=img_info.source_path
                    ))
                manual_images = scaled_manual

            folder_images = self._load_folders(
                文件夹1, 文件夹2, 文件夹3, 文件夹4,
                像素缩放, 分辨率像素,
                文件夹5, 文件夹6, 文件夹7, 文件夹8, 文件夹9
            )

            pairing_mode = kwargs.get("图片配对模式", "不配对")
            pairs = self._create_pairs(folder_images, pairing_mode, manual_images if manual_images else None)

            if not pairs:
                raise ValueError("没有可处理的图片组合，请检查文件夹路径和参考图输入")

            batch_prompts = parse_batch_prompts(提示词)
            prompts_per_task = None

            if batch_prompts:
                if len(batch_prompts) != len(pairs):
                    raise ValueError(
                        f"批量提示词数量 ({len(batch_prompts)}) 与任务数量 ({len(pairs)}) 不匹配！\n"
                        f"请确保提示词数量与图片组合数量一致"
                    )
                prompts_per_task = batch_prompts
                print(f"全能生图（批量）: 批量提示词模式 - {len(batch_prompts)} 个提示词")

            output_folder = 保存路径.strip() if 保存路径 else ""
            if not output_folder and FOLDER_PATHS_AVAILABLE:
                output_folder = folder_paths.get_output_directory()

            if not output_folder:
                raise ValueError("无法确定保存路径，请指定保存路径或确保 folder_paths 可用")

            import os
            os.makedirs(output_folder, exist_ok=True)
            print(f"全能生图（批量）: 保存路径 → {output_folder}")

            pbar = None
            if PROGRESS_BAR_AVAILABLE:
                pbar = ProgressBar(len(pairs))

            def run_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self._process_batch_async(
                            pairs=pairs,
                            prompt=提示词,
                            model=模型,
                            resolution=分辨率,
                            aspect_ratio=宽高比,
                            output_folder=output_folder,
                            pbar=pbar,
                            prompts_per_task=prompts_per_task,
                        )
                    )
                finally:
                    loop.close()

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_async)
                results = future.result(timeout=3600)

            success_count = sum(1 for r in results if r.get("success", False))
            fail_count = len(results) - success_count
            all_saved_files = []
            for r in results:
                all_saved_files.extend(r.get("saved_files", []))

            elapsed = time.time() - start_time
            print(f"全能生图（批量）: 完成！总耗时 {elapsed:.2f}s | 成功: {success_count}/{len(pairs)} | 失败: {fail_count}")

            output_images = []
            max_output = 10
            recent_files = all_saved_files[-min(max_output, len(all_saved_files)):]
            for file_path in recent_files:
                try:
                    img = Image.open(file_path)
                    output_images.append(img)
                except Exception as e:
                    print(f"全能生图（批量）: 无法加载 {file_path} - {e}")

            if not output_images:
                placeholder = Image.new('RGB', (512, 512), color=(128, 128, 128))
                output_images = [placeholder]

            output_tensor = pil_to_tensor(output_images)
            print(f"全能生图（批量）: 共保存 {len(all_saved_files)} 张图片，节点输出最后 {len(output_images)} 张")

            import gc
            gc.collect()
            return (output_tensor,)

        except Exception as e:
            print(f"全能生图（批量）: ❌ {str(e)}")
            raise

