"""
批量 Nano Banana Pro 节点
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
    generate_output_filename,
    save_image
)
from ..clients.gemini_client import GeminiAPIClient
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
    print("⚠️ BatchNanoBananaPro: comfy.utils.ProgressBar 不可用，将只使用终端进度显示")

# 导入 ComfyUI 的文件夹路径管理
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False
    print("⚠️ BatchNanoBananaPro: folder_paths 不可用，将无法使用默认保存路径")

# 内存监控（可选）
try:
    import psutil
    MEMORY_MONITOR_AVAILABLE = True
except ImportError:
    MEMORY_MONITOR_AVAILABLE = False
    print("⚠️ BatchNanoBananaPro: psutil 不可用，内存监控功能禁用")

# ============================================================================
# 调试日志配置
# ============================================================================
# 是否启用调试日志（打印完整的 API 响应内容）
# 设置为 True 以启用调试日志，False 以禁用
DEBUG_LOG_ENABLED = False
# 是否启用请求体日志（打印发送给 API 的请求体，base64 图片数据将自动截断）
# 设置为 True 以启用请求体日志，False 以禁用
REQUEST_LOG_ENABLED = False
# ============================================================================


class BatchNanoBananaPro:
    """
    批量 Nano Banana Pro 节点
    
    功能：
    - 从多个文件夹加载图片
    - 支持三种配对模式：
      * 1:1 - 索引配对（文件夹之间按位置配对）
      * 1*N - 笛卡尔积配对（所有可能组合）
      * 不配对 - 固定参考图模式（文件夹图片依次与所有参考图组合）
    - 批量调用 API 生成图像
    - 智能命名保存（保留原始文件名）
    - 并发控制（默认最大 100）
    
    注意：
    - 「不配对」模式只支持单个文件夹
    - 支持的模型列表从 models_config.py 动态加载
    - 要添加/禁用模型，请编辑 models_config.py 文件
    """
    
    # 支持的模型列表（从配置文件动态加载）
    MODELS = None  # 将在 INPUT_TYPES 中动态获取
    
    # 支持的宽高比列表（全量：所有启用模型的并集，动态加载）
    # 实际渲染时通过 get_all_supported_aspect_ratios() 获取
    ASPECT_RATIOS = [
        "1:1", "4:3", "3:4", "16:9", "9:16",
        "2:3", "3:2", "4:5", "5:4", "21:9",
        "1:4", "4:1", "1:8", "8:1"
    ]
    
    # 支持的分辨率列表（全量兜底，实际由 get_all_supported_resolutions() 动态生成）
    RESOLUTIONS = ["512", "1K", "2K", "4K"]
    
    # 配对模式
    PAIRING_MODES = ["按相同图片命名", "1*N", "不配对"]
    
    def __init__(self):
        """初始化节点"""
        self.client = None
    
    def resize_to_megapixels(
        self,
        image: Image.Image,
        target_megapixels: float
    ) -> Image.Image:
        """
        将图像缩放到指定的总像素数，保持纵横比
        
        Args:
            image: PIL Image 对象
            target_megapixels: 目标像素数（百万像素）
        
        Returns:
            缩放后的 PIL Image
        
        Example:
            >>> resized = self.resize_to_megapixels(img, 2.0)  # 缩放到2百万像素
        """
        # 计算当前像素数
        current_pixels = image.width * image.height
        target_pixels = int(target_megapixels * 1_000_000)
        
        # 如果当前像素数已经接近目标，则不缩放
        if abs(current_pixels - target_pixels) / target_pixels < 0.05:
            return image
        
        # 计算缩放比例
        scale = (target_pixels / current_pixels) ** 0.5
        
        # 计算新尺寸
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        
        # 确保至少为1像素
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # 使用 Lanczos 重采样
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return resized_image
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        定义输入参数
        
        ComfyUI 节点规范：
        - required: 必选参数
        - optional: 可选参数
        """
        # 从配置文件动态获取启用的模型列表
        enabled_models = get_enabled_models()
        
        # 如果没有启用的模型，使用空列表（会导致节点不可用，提示用户配置）
        if not enabled_models:
            enabled_models = ["请在 models_config.py 中启用至少一个模型"]
        
        # 动态获取所有启用模型支持的宽高比（去重合并）
        all_aspect_ratios = get_all_supported_aspect_ratios()
        if not all_aspect_ratios:
            all_aspect_ratios = cls.ASPECT_RATIOS
        
        # 动态获取所有启用模型支持的分辨率（去重合并）
        all_resolutions = get_all_supported_resolutions()
        if not all_resolutions:
            all_resolutions = cls.RESOLUTIONS
        
        # 创建9个独立的图像输入
        optional_inputs = {}
        for i in range(1, 10):  # 1-9
            optional_inputs[f"参考图{i}"] = ("IMAGE",)
        
        # 图片配对模式移到可选参数
        optional_inputs["图片配对模式"] = (cls.PAIRING_MODES, {
            "default": "不配对"
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
                "谷歌搜索（联网）": (["关闭", "打开"], {
                    "default": "关闭"
                }),
                "图片搜索（联网）": (["关闭", "打开"], {
                    "default": "关闭"
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
    
    # 返回值类型
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    
    # 执行函数名
    FUNCTION = "process_batch"
    
    # 节点分类
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
        """
        加载所有文件夹中的图片
        
        Args:
            folder1-9: 文件夹路径
            enable_scaling: 是否启用像素缩放
            target_megapixels: 目标像素数（百万像素）
        
        Returns:
            图片列表的列表
        """
        folders = [folder1, folder2, folder3, folder4, folder5, folder6, folder7, folder8, folder9]
        all_images = []
        
        for i, folder in enumerate(folders, 1):
            if folder and folder.strip():
                try:
                    images = load_images_from_folder(folder)
                    if images:
                        # 应用像素缩放
                        if enable_scaling:
                            scaled_images = []
                            for img_info in images:
                                scaled_img = self.resize_to_megapixels(
                                    img_info.image,
                                    target_megapixels
                                )
                                # 创建新的 ImageInfo，保留其他元数据
                                scaled_info = ImageInfo(
                                    image=scaled_img,
                                    filename=img_info.filename,
                                    extension=img_info.extension,
                                    source_path=img_info.source_path
                                )
                                scaled_images.append(scaled_info)
                            images = scaled_images
                        
                        all_images.append(images)
                    else:
                        # 空文件夹，静默跳过
                        pass
                except ValueError as e:
                    print(f"BatchNanoBananaPro: 文件夹{i} 加载失败 - {e}")
        
        return all_images
    
    def _create_pairs(
        self,
        image_lists: List[List[ImageInfo]],
        pairing_mode: str,
        manual_images: Optional[List[ImageInfo]] = None
    ) -> List[Tuple[ImageInfo, ...]]:
        """
        根据配对模式创建图片组合
        
        Args:
            image_lists: 从文件夹加载的图片列表
            pairing_mode: 配对模式 (1:1, 1*N, 不配对)
            manual_images: 手动输入的参考图
        
        Returns:
            配对后的元组列表
        
        Raises:
            ValueError: 不配对模式下填入多个文件夹时
        """
        # === 不配对模式 ===
        if pairing_mode == "不配对":
            # 验证：只支持单个文件夹
            if len(image_lists) > 1:
                raise ValueError("「不配对」模式只支持单个文件夹，请清空其他文件夹路径")
            
            # 场景1：有文件夹 + 有参考图 → 每张文件夹图片 + 所有参考图
            if image_lists and manual_images:
                folder_images = image_lists[0]
                pairs = []
                for img in folder_images:
                    pair = (img,) + tuple(manual_images)
                    pairs.append(pair)
                return pairs
            
            # 场景2：有文件夹 + 无参考图 → 每张图片单独成组
            elif image_lists:
                return [(img,) for img in image_lists[0]]
            
            else:
                return []
        
        # === 1:1 和 1*N 模式 ===
        # 参考图不参与配对，仅在文件夹图片之间进行配对
        if not image_lists:
            return []
        
        # 文件夹图片配对
        if len(image_lists) == 1:
            base_pairs = [(img,) for img in image_lists[0]]
        elif pairing_mode == "按相同图片命名":
            base_pairs = list(pair_images_by_name(*image_lists))
        else:  # 1*N
            base_pairs = list(pair_images_cartesian(*image_lists))
        
        # 将所有参考图追加到每组末尾（不参与配对逻辑）
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
        enable_image_search: bool = False
    ) -> dict:
        """
        执行单个生成任务
        
        Args:
            client: API 客户端
            session: aiohttp 会话
            prompt: 提示词
            model: 模型名称
            resolution: 分辨率
            aspect_ratio: 宽高比
            images: 输入图片列表
            output_folder: 输出文件夹
            task_index: 任务索引
        
        Returns:
            包含结果信息的字典
        """
        result = {
            "task_index": task_index,
            "success": False,
            "generated_count": 0,
            "saved_files": [],
            "output_images": [],  # 无保存路径时存储内存图片
            "error": None
        }
        
        try:
            # 准备输入图片
            input_pil_images = [info.image for info in images]
            
            # 调用 API 生成图片（固定生成1次）
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
                    enable_image_search=enable_image_search
                )
                if gen_result:
                    # 正确解包元组：第一个元素是图像列表，第二个是计时信息
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
            
            has_save_path = bool(output_folder and output_folder.strip())
            
            # 保存或存储生成的图片
            for i, gen_img in enumerate(generated_images):
                if has_save_path:
                    # 有保存路径：立即写入磁盘
                    output_path = generate_output_filename(
                        source_images=list(images),
                        batch_index=i,
                        output_folder=output_folder,
                        extension=".png",
                        task_id=f"task{task_index}"
                    )
                    save_image(gen_img, output_path)
                    result["saved_files"].append(output_path)
                    
                    # 立即释放内存中的图片对象
                    gen_img = None
                else:
                    # 无保存路径：存入内存（但限制数量）
                    if len(result["output_images"]) < 5:  # 最多保存5张在内存
                        result["output_images"].append(gen_img)
                    else:
                        # 超过限制，释放内存
                        gen_img = None
            
            # 只有生成了图片才标记为成功
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
        enable_image_search: bool = False
    ) -> List[dict]:
        """
        异步批量处理所有任务 - 改进版：支持分批保存
        
        Args:
            pairs: 配对后的图片组合
            prompt: 提示词（单提示词模式时使用）
            model: 模型名称
            resolution: 分辨率
            aspect_ratio: 宽高比
            output_folder: 输出文件夹
            pbar: ComfyUI 进度条
            prompts_per_task: 每个任务对应的提示词列表（批量提示词模式时使用）
        
        Returns:
            所有任务的结果列表
        """
        if self.client is None:
            self.client = GeminiAPIClient()
        
        total_tasks = len(pairs)
        
        # 保持并发数为10不变（按用户要求）
        max_concurrent = 10
        
        # 分批保存的批次大小（与并发数一致）
        save_batch_size = 10
        
        print(f"BatchNanoBananaPro: 检测到 {total_tasks} 个任务")
        
        # 创建进度记录文件（即使崩溃也能知道进度）
        import os
        progress_file = None
        if output_folder and output_folder.strip():
            progress_file = os.path.join(output_folder, ".batch_progress.txt")
            try:
                with open(progress_file, 'w') as f:
                    f.write(f"总任务数: {total_tasks}\n")
                    f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"已完成: 0\n")
                print(f"BatchNanoBananaPro: 进度记录文件已创建: {progress_file}")
            except:
                print("BatchNanoBananaPro: 无法创建进度记录文件，继续执行...")
        
        all_results = []
        completed = 0
        success_count = 0
        fail_count = 0
        
        # 计算生成批次数量
        num_batches = math.ceil(total_tasks / max_concurrent)
        
        # 内存监控初始化
        if MEMORY_MONITOR_AVAILABLE and total_tasks > 50:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            print(f"BatchNanoBananaPro: 初始内存使用: {initial_memory:.1f} MB")
        
        # 进度打印配置：任务数 >= 50 时，额外显示百分比里程碑
        show_milestone = total_tasks >= 50
        milestones = [0.2, 0.4, 0.6, 0.8, 1.0]  # 20%, 40%, 60%, 80%, 100%
        milestone_index = 0
        
        if num_batches > 1:
            print(f"BatchNanoBananaPro: 任务数 {total_tasks} 超过并发上限 {max_concurrent}，将分 {num_batches} 批执行")
        
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # 分批处理：每批最多10个任务
            for batch_idx in range(num_batches):
                start_idx = batch_idx * max_concurrent
                end_idx = min(start_idx + max_concurrent, total_tasks)
                batch_pairs = pairs[start_idx:end_idx]
                
                if num_batches > 1:
                    print(f"BatchNanoBananaPro: 执行第 {batch_idx + 1}/{num_batches} 批 ({start_idx + 1}-{end_idx})...")
                
                # 创建当前批次的任务
                tasks = []
                for i, pair in enumerate(batch_pairs):
                    # 批量提示词模式时，每个任务使用对应的提示词；否则使用统一提示词
                    task_prompt = prompts_per_task[start_idx + i] if prompts_per_task else prompt
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
                            enable_image_search=enable_image_search
                        )
                    )
                    tasks.append(task)
                
                # 收集当前批次的结果
                batch_results = []
                for coro in asyncio.as_completed(tasks):
                    result_data = None
                    try:
                        result = await coro
                        if isinstance(result, Exception):
                            result_data = {
                                "success": False,
                                "error": str(result),
                                "generated_count": 0,
                                "saved_files": []
                            }
                            batch_results.append(result_data)
                        else:
                            result_data = result
                            batch_results.append(result)
                    except Exception as e:
                        result_data = {
                            "success": False,
                            "error": str(e),
                            "generated_count": 0,
                            "saved_files": []
                        }
                        batch_results.append(result_data)
                    
                    completed += 1
                    
                    # 根据成功/失败状态打印不同信息
                    if result_data and result_data.get("success", False):
                        success_count += 1
                        print(f"BatchNanoBananaPro: 任务 {completed}/{total_tasks} 成功 ✓")
                    else:
                        fail_count += 1
                        # 打印完整错误信息（用于排查问题）
                        error_msg = result_data.get("error", "未知错误") if result_data else "未知错误"
                        print(f"BatchNanoBananaPro: 任务 {completed}/{total_tasks} 失败 ✗")
                        print(f"=" * 80)
                        print(f"🔍 【原始报错信息展示】")
                        print(f"=" * 80)
                        print(f"任务编号: {completed}/{total_tasks}")
                        print(f"失败时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"-" * 80)
                        print(f"错误详情:")
                        print(error_msg)
                        print(f"=" * 80)
                    
                    # 更新进度文件（每完成10个任务或每个批次）
                    if progress_file and (completed % 10 == 0 or completed == total_tasks):
                        try:
                            with open(progress_file, 'w') as f:
                                f.write(f"总任务数: {total_tasks}\n")
                                f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                                f.write(f"已完成: {completed}\n")
                                f.write(f"成功: {success_count}\n")
                                f.write(f"失败: {fail_count}\n")
                                f.write(f"进度: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)\n")
                            if completed % 50 == 0:
                                print(f"BatchNanoBananaPro: 进度已保存 - {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)")
                        except:
                            pass  # 进度文件更新失败不影响主流程
                    
                    # 更新 ComfyUI 原生进度条
                    if pbar is not None:
                        pbar.update(1)
                    
                    # 大任务额外显示百分比里程碑
                    if show_milestone and milestone_index < len(milestones):
                        progress = completed / total_tasks
                        if progress >= milestones[milestone_index]:
                            percentage = int(milestones[milestone_index] * 100)
                            print(f"BatchNanoBananaPro: >>> 进度 {percentage}% <<<")
                            milestone_index += 1
                
                # 当前批次完成后，立即保存结果并清理内存
                all_results.extend(batch_results)
                
                # 分批保存：每完成一批（10个任务），立即处理保存并清理内存
                print(f"BatchNanoBananaPro: 第 {batch_idx + 1} 批完成，开始分批保存...")
                
                # 统计当前批次的结果
                batch_success = sum(1 for r in batch_results if r.get("success", False))
                batch_fail = len(batch_results) - batch_success
                batch_generated = sum(r.get("generated_count", 0) for r in batch_results)
                
                print(f"BatchNanoBananaPro: 本批结果 - 成功: {batch_success}/{len(batch_results)}，生成: {batch_generated} 张")
                
                # 强制垃圾回收，释放内存
                import gc
                gc.collect()
                
                # 内存监控
                if MEMORY_MONITOR_AVAILABLE and total_tasks > 50:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    print(f"BatchNanoBananaPro: 内存使用: {current_memory:.1f} MB (+{memory_increase:.1f} MB)")
                    
                    # 内存警告阈值（2GB）
                    if current_memory > 2000:
                        print(f"⚠️ BatchNanoBananaPro: 内存使用过高！但图片已分批保存，即使崩溃也不会丢失已完成的任务")
                
                # 短暂暂停，让系统有时间处理文件I/O
                await asyncio.sleep(0.5)
        
        return all_results
    
    def process_batch(
        self,
        prompt: str,
        文件夹1: str,
        文件夹2: str,
        文件夹3: str,
        文件夹4: str,
        文件夹5: str,
        文件夹6: str,
        文件夹7: str,
        文件夹8: str,
        文件夹9: str,
        像素缩放: bool,
        分辨率像素: float,
        seed: int,
        图片配对模式: str,
        模型: str,
        宽高比: str,
        分辨率: str,
        保存路径: str = "",
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        批量处理图像生成任务
        
        Args:
            prompt: 提示词
            文件夹1-9: 图片文件夹路径
            像素缩放: 是否启用像素缩放
            分辨率像素: 目标像素数（百万像素）
            seed: 随机种子
            保存路径: 输出保存路径
            图片配对模式: 1:1 或 1*N
            模型: 模型名称
            宽高比: 输出宽高比
            分辨率: 输出分辨率
            **kwargs: 动态参考图输入 (参考图1-9)
        
        Returns:
            输出图像张量
        """
        start_time = time.time()
        
        # 从 kwargs 提取搜索参数（界面显示为「关闭/打开」，转为 bool 供调用）
        enable_grounding: bool = (kwargs.pop("谷歌搜索（联网）", "关闭") == "打开")
        enable_image_search: bool = (kwargs.pop("图片搜索（联网）", "关闭") == "打开")
        
        # 初始化进度文件变量（避免后续引用报错）
        progress_file = None
        
        try:
            # 设置随机种子（用于本地随机操作）
            random.seed(seed)
            np.random.seed(seed % (2**32))
            
            # 验证：至少需要填写一个文件夹路径
            has_any_folder = any(
                f and f.strip()
                for f in [文件夹1, 文件夹2, 文件夹3, 文件夹4, 文件夹5, 文件夹6, 文件夹7, 文件夹8, 文件夹9]
            )
            if not has_any_folder:
                raise ValueError("请至少填写一个文件夹路径，该节点专为批量文件夹处理设计")
            
            # 校验分辨率与模型的兼容性
            supported_resolutions = get_model_supported_resolutions(模型)
            if supported_resolutions and 分辨率 not in supported_resolutions:
                raise ValueError(
                    f"分辨率 \"{分辨率}\" 与模型 \"{模型}\" 不兼容！\n"
                    f"该模型支持的分辨率：{', '.join(supported_resolutions)}"
                )
            
            # 校验宽高比与模型的兼容性
            supported_ratios = get_model_supported_aspect_ratios(模型)
            if supported_ratios and 宽高比 not in supported_ratios:
                raise ValueError(
                    f"宽高比 \"{宽高比}\" 与模型 \"{模型}\" 不兼容！\n"
                    f"该模型支持的宽高比：{', '.join(supported_ratios)}"
                )
            
            # 校验图片搜索（联网）与模型的兼容性
            # 仅 nano-banana-2 和 gemini-3.1-flash-image-preview 支持图片搜索
            IMAGE_SEARCH_UNSUPPORTED_MODELS = ["nano-banana-pro", "gemini-3-pro-image-preview"]
            if enable_image_search and 模型 in IMAGE_SEARCH_UNSUPPORTED_MODELS:
                raise ValueError(
                    f"模型 \"{模型}\" 不支持【图片搜索（联网）】功能！"
                    f"请切换到 nano-banana-2 或 gemini-3.1-flash-image-preview 后再使用"
                )
            
            # 加载文件夹图片
            print("BatchNanoBananaPro: 开始加载图片...")
            image_lists = self._load_folders(
                文件夹1, 文件夹2, 文件夹3, 文件夹4,
                像素缩放, 分辨率像素,
                文件夹5, 文件夹6, 文件夹7, 文件夹8, 文件夹9
            )
            
            # 验证文件夹是否有可用图片
            total_folder_images = sum(len(lst) for lst in image_lists)
            if total_folder_images == 0:
                raise ValueError("文件夹中未找到任何图片，请检查文件夹路径是否正确")
            
            # 处理独立的参考图输入
            manual_images = []
            for i in range(1, 10):  # 1-9
                key = f"参考图{i}"
                if key in kwargs and kwargs[key] is not None:
                    pil_images = tensor_to_pil(kwargs[key])
                    for j, img in enumerate(pil_images):
                        # 如果启用像素缩放，也对参考图进行缩放
                        if 像素缩放:
                            img = self.resize_to_megapixels(img, 分辨率像素)
                        
                        manual_images.append(
                            ImageInfo(
                                image=img,
                                filename=f"manual_{i}_{j}",
                                extension=".png",
                                source_path=""
                            )
                        )
            
            # 创建配对
            pairs = self._create_pairs(image_lists, 图片配对模式, manual_images if manual_images else None)
            
            if not pairs:
                raise ValueError("配对结果为空，请检查输入")
            
            # 解析批量提示词（使用 --- 分隔多个提示词）
            batch_prompts = parse_batch_prompts(prompt)
            prompts_per_task = None
            if batch_prompts:
                # 展开 pairs × prompts：每个图片组合 × 每个提示词 = 一个任务
                expanded_pairs = []
                expanded_prompts = []
                for pair in pairs:
                    for bp in batch_prompts:
                        expanded_pairs.append(pair)
                        expanded_prompts.append(bp)
                pairs = expanded_pairs
                prompts_per_task = expanded_prompts
            
            total_tasks = len(pairs)
            
            # 打印首行概览
            # 图片搜索（联网）开启时隐含谷歌搜索接地，与客户端请求逻辑保持一致
            grounding_str = ""
            if enable_image_search:
                grounding_str = " | 谷歌图片搜索接地"
            elif enable_grounding:
                grounding_str = " | 谷歌搜索接地"
            
            if batch_prompts:
                print(f"BatchNanoBananaPro: 批量任务 | {图片配对模式} 配对模式 × {len(batch_prompts)}个提示词 | 共 {total_tasks} 任务{grounding_str}")
            else:
                print(f"BatchNanoBananaPro: 批量任务 | {图片配对模式} 配对模式 | 共 {total_tasks} 任务{grounding_str}")
            
            # 创建 ComfyUI 原生进度条
            pbar = None
            if PROGRESS_BAR_AVAILABLE:
                pbar = ProgressBar(total_tasks)
            
            # 检查保存路径（重要！）
            has_save_path = bool(保存路径 and 保存路径.strip())
            if not has_save_path:
                # 使用 ComfyUI 默认 output 目录作为保存路径
                if FOLDER_PATHS_AVAILABLE:
                    保存路径 = folder_paths.get_output_directory()
                    has_save_path = True
                    print(f"BatchNanoBananaPro: 未设置保存路径，将使用 ComfyUI 默认 output 目录: {保存路径}")
                else:
                    print("⚠️ BatchNanoBananaPro: 警告！未设置保存路径且无法获取 ComfyUI output 目录，图片将只保存在内存中")
                    print("⚠️ 对于大批量任务（如280张），强烈建议设置保存路径以避免内存崩溃")
            
            if has_save_path:
                # 验证保存路径
                import os
                try:
                    os.makedirs(保存路径, exist_ok=True)
                    # 测试写入权限
                    test_file = os.path.join(保存路径, ".write_test")
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                    print(f"BatchNanoBananaPro: 保存路径验证通过: {保存路径}")
                except Exception as e:
                    raise ValueError(f"保存路径无效或无写入权限: {保存路径} - {str(e)}")
            
            # 初始化 API 客户端
            if self.client is None:
                try:
                    self.client = GeminiAPIClient()
                except ValueError as e:
                    raise ValueError(f"初始化 API 客户端失败: {str(e)}")
            
            # 执行批量生成
            # 在新线程中运行异步代码，避免事件循环冲突
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
                            enable_image_search=enable_image_search
                        )
                    )
                except Exception as e:
                    # 即使崩溃，也记录错误
                    print(f"BatchNanoBananaPro: 异步任务执行异常: {str(e)}")
                    raise
                finally:
                    loop.close()
            
            # 使用线程池在新线程中运行事件循环
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_async_in_thread)
                try:
                    results = future.result(timeout=3600)  # 1小时超时
                except TimeoutError:
                    print("BatchNanoBananaPro: 任务执行超时（1小时）")
                    raise RuntimeError("任务执行超时，请减少任务数量或检查网络连接")
                except Exception as e:
                    print(f"BatchNanoBananaPro: 任务执行失败: {str(e)}")
                    # 即使失败，也尝试返回部分结果
                    if 'all_saved_files' in locals():
                        print(f"BatchNanoBananaPro: 部分保存的图片: {len(all_saved_files)} 张")
                    raise
            
            # 统计结果
            success_count = sum(1 for r in results if r.get("success", False))
            fail_count = len(results) - success_count
            total_generated = sum(r.get("generated_count", 0) for r in results)
            all_saved_files = []
            for r in results:
                all_saved_files.extend(r.get("saved_files", []))
            
            elapsed = time.time() - start_time
            
            # 格式化时间
            if elapsed < 1:
                time_str = f"{elapsed:.3f}s"
            else:
                time_str = f"{elapsed:.2f}s"
            
            # 计算平均耗时
            avg_time = elapsed / success_count if success_count > 0 else 0
            avg_time_str = f"{avg_time:.1f}s/张" if success_count > 0 else "N/A"
            
            # 精简统计信息
            has_save_path = bool(保存路径 and 保存路径.strip())
            is_default_path = not bool(kwargs.get('保存路径', '').strip() if '保存路径' in locals() else False)
            print("=" * 60)
            print(f"完成！总耗时 {time_str} | 成功: {success_count}/{total_tasks} | 生成 {total_generated} 张 | 平均 {avg_time_str}")
            if has_save_path:
                if is_default_path:
                    print(f"保存路径: {保存路径} (ComfyUI 默认 output 目录)")
                else:
                    print(f"保存路径: {保存路径}")
            else:
                print("保存路径: 未设置（仅输出到节点）")
            
            # 失败详情（如果有）
            failed_results = [r for r in results if not r.get("success", False)]
            if failed_results:
                print(f"-" * 60)
                print(f"❌ 失败任务汇总: {len(failed_results)} 个")
                print(f"-" * 60)
                
                # 显示前3个失败任务的详细信息
                for idx, failed in enumerate(failed_results[:3], 1):
                    task_num = failed.get('task_index', '?') + 1
                    error_msg = failed.get('error', '未知错误')
                    print(f"\n【失败任务 #{task_num}】")
                    print(f"错误信息: {error_msg}")
                
                if len(failed_results) > 3:
                    remaining = [str(r.get('task_index', '?') + 1) for r in failed_results[3:]]
                    print(f"\n其他失败任务编号: {', '.join(remaining)}")
                
                print(f"-" * 60)
            
            # 收集生成的图片（优化内存使用 - 分批保存版）
            output_images = []
            
            # 限制输出图片数量，避免内存爆炸
            max_output_images = 5  # 最多输出5张图片到ComfyUI（进一步减少）
            
            # 策略1：优先使用内存中的图片（如果有）
            saved_count = 0
            for r in results:
                for img in r.get("output_images", []):
                    if saved_count >= max_output_images:
                        break
                    output_images.append(img)
                    saved_count += 1
            
            # 策略2：如果内存图片不足，从磁盘加载少量图片
            if saved_count < max_output_images and all_saved_files:
                # 只加载最新保存的几张图片
                recent_files = all_saved_files[-min(5, len(all_saved_files)):]  # 最近5个文件
                for file_path in recent_files:
                    if saved_count >= max_output_images:
                        break
                    try:
                        img = Image.open(file_path)
                        output_images.append(img)
                        saved_count += 1
                    except Exception as e:
                        print(f"BatchNanoBananaPro: 无法加载图片 {file_path} - {e}")
            
            # 策略3：如果还是没有图片，创建一个占位图
            if not output_images:
                placeholder = Image.new('RGB', (512, 512), color=(128, 128, 128))
                output_images = [placeholder]
                print(f"BatchNanoBananaPro: 未找到可输出的图片，使用占位图")
            
            # 转换为张量
            output_tensor = pil_to_tensor(output_images)
            
            # 最终内存清理
            import gc
            gc.collect()
            
            # 打印最终统计信息
            total_saved = len(all_saved_files)
            print(f"BatchNanoBananaPro: 任务完成！共保存 {total_saved} 张图片到磁盘")
            if total_saved > 0:
                print(f"BatchNanoBananaPro: 最新保存的文件: {all_saved_files[-1]}")
            
            # 清理进度文件
            if progress_file and os.path.exists(progress_file):
                try:
                    os.remove(progress_file)
                    print(f"BatchNanoBananaPro: 进度文件已清理")
                except:
                    pass
            
            return (output_tensor,)
        
        except ValueError as e:
            # 检测是否为授权错误
            if str(e) == "未授权！":
                print("请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            else:
                # 用户输入错误 - 打印完整错误信息
                error_msg = str(e)
                print(f"BatchNanoBananaPro: ❌ {error_msg}")
            raise ValueError(error_msg) from None
        
        except RuntimeError as e:
            # 打印完整错误信息
            error_full = str(e)
            print(f"BatchNanoBananaPro: ❌ {error_full}")
            raise RuntimeError(error_full) from None
        
        except Exception as e:
            # 其他未知错误 - 打印完整错误信息
            error_msg = str(e)
            print(f"BatchNanoBananaPro: ❌ {error_msg}")
            
            # 即使崩溃，也更新进度文件
            if 'progress_file' in locals() and progress_file:
                try:
                    with open(progress_file, 'w') as f:
                        f.write(f"总任务数: {total_tasks}\n")
                        f.write(f"错误时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"错误信息: {error_msg}\n")
                        if 'completed' in locals():
                            f.write(f"已完成: {completed}\n")
                        if 'all_saved_files' in locals():
                            f.write(f"已保存图片: {len(all_saved_files)}\n")
                    print(f"BatchNanoBananaPro: 错误状态已保存到进度文件")
                except:
                    pass
            
            raise type(e)(error_msg) from None
        
        finally:
            # 查询余额
            if self.client is not None:
                try:
                    balance_data = self.client.query_balance_sync()
                    balance_info = self.client.format_balance_info(balance_data)
                    print(f"BatchNanaBananaPro: {balance_info}")
                    print("=" * 60)
                except Exception:
                    pass
            
            # 最终内存清理
            import gc
            gc.collect()
            print(f"BatchNanoBananaPro: 最终内存清理完成")