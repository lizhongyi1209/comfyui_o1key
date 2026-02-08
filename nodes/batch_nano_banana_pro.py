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

from ..utils.image_utils import tensor_to_pil, pil_to_tensor
from ..utils.file_utils import (
    ImageInfo,
    load_images_from_folder,
    pair_images_indexed,
    pair_images_cartesian,
    generate_output_filename,
    save_image
)
from ..clients.gemini_client import GeminiAPIClient
from ..models_config import get_enabled_models

# 导入 ComfyUI 原生进度条
try:
    from comfy.utils import ProgressBar
    PROGRESS_BAR_AVAILABLE = True
except ImportError:
    PROGRESS_BAR_AVAILABLE = False
    print("⚠️ BatchNanoBananaPro: comfy.utils.ProgressBar 不可用，将只使用终端进度显示")


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
    
    # 支持的宽高比列表
    ASPECT_RATIOS = [
        "1:1", "4:3", "3:4", "16:9", "9:16",
        "2:3", "3:2", "4:5", "5:4", "21:9"
    ]
    
    # 支持的分辨率列表
    RESOLUTIONS = ["1K", "2K", "4K"]
    
    # 配对模式
    PAIRING_MODES = ["1:1", "1*N", "不配对"]
    
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
        
        # 创建9个独立的图像输入
        optional_inputs = {}
        for i in range(1, 10):  # 1-9
            optional_inputs[f"参考图{i}"] = ("IMAGE",)
        
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "一个中国女子的OOTD",
                    "multiline": True
                }),
                "模型": (enabled_models, {
                    "default": enabled_models[0]
                }),
                "宽高比": (cls.ASPECT_RATIOS, {
                    "default": "1:1"
                }),
                "分辨率": (cls.RESOLUTIONS, {
                    "default": "2K"
                }),
                "像素缩放": ("BOOLEAN", {
                    "default": False
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
                "保存路径": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "图片配对模式": (cls.PAIRING_MODES, {
                    "default": "不配对"
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
        target_megapixels: float
    ) -> List[List[ImageInfo]]:
        """
        加载所有文件夹中的图片
        
        Args:
            folder1-4: 文件夹路径
            enable_scaling: 是否启用像素缩放
            target_megapixels: 目标像素数（百万像素）
        
        Returns:
            图片列表的列表
        """
        folders = [folder1, folder2, folder3, folder4]
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
        # === 新模式：不配对 ===
        if pairing_mode == "不配对":
            # 验证：只支持单个文件夹
            if len(image_lists) > 1:
                raise ValueError("「不配对」模式只支持单个文件夹，请清空其他文件夹路径")
            
            # 场景1：有文件夹 + 有参考图
            if image_lists and manual_images:
                folder_images = image_lists[0]
                # 每张文件夹图片 + 所有参考图
                pairs = []
                for img in folder_images:
                    pair = (img,) + tuple(manual_images)
                    pairs.append(pair)
                return pairs
            
            # 场景2：有文件夹 + 无参考图
            elif image_lists:
                # 每张图片单独成组
                return [(img,) for img in image_lists[0]]
            
            # 场景3：无文件夹 + 有参考图
            elif manual_images:
                # 每张参考图单独成组
                return [(img,) for img in manual_images]
            
            else:
                return []
        
        # === 原有逻辑：1:1 和 1*N ===
        # 如果有手动参考图，添加到列表中（所有参考图作为一个列表）
        if manual_images:
            image_lists.append(manual_images)
        
        if not image_lists:
            return []
        
        # 如果只有一个列表，直接返回每个图片作为单元素元组
        if len(image_lists) == 1:
            return [(img,) for img in image_lists[0]]
        
        # 根据配对模式选择配对函数
        if pairing_mode == "1:1":
            pairs = pair_images_indexed(*image_lists)
        else:  # 1*N
            pairs = pair_images_cartesian(*image_lists)
        
        return pairs
    
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
        task_index: int
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
                    session=session
                )
                if gen_result:
                    generated_images.extend(gen_result)
            except Exception as e:
                error_msg = str(e)
                print(f"BatchNanoBananaPro: 任务 {task_index + 1} 生成失败 - {error_msg}")
                result["error"] = error_msg
            
            # 保存生成的图片
            for i, gen_img in enumerate(generated_images):
                # 使用任务索引作为唯一标识，确保并发安全
                output_path = generate_output_filename(
                    source_images=list(images),
                    batch_index=i,
                    output_folder=output_folder,
                    extension=".png",
                    task_id=f"task{task_index}"
                )
                save_image(gen_img, output_path)
                result["saved_files"].append(output_path)
            
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
        pbar=None
    ) -> List[dict]:
        """
        异步批量处理所有任务
        
        Args:
            pairs: 配对后的图片组合
            prompt: 提示词
            model: 模型名称
            resolution: 分辨率
            aspect_ratio: 宽高比
            output_folder: 输出文件夹
        
        Returns:
            所有任务的结果列表
        """
        if self.client is None:
            self.client = GeminiAPIClient()
        
        # 固定最大并发数为 100
        max_concurrent = 100
        
        total_tasks = len(pairs)
        all_results = []
        completed = 0
        success_count = 0
        fail_count = 0
        
        # 计算分批数量
        num_batches = math.ceil(total_tasks / max_concurrent)
        
        # 进度打印配置：任务数 >= 50 时，额外显示百分比里程碑
        show_milestone = total_tasks >= 50
        milestones = [0.2, 0.4, 0.6, 0.8, 1.0]  # 20%, 40%, 60%, 80%, 100%
        milestone_index = 0
        
        if num_batches > 1:
            print(f"BatchNanoBananaPro: 任务数 {total_tasks} 超过并发上限 {max_concurrent}，将分 {num_batches} 批执行")
        
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * max_concurrent
                end_idx = min(start_idx + max_concurrent, total_tasks)
                batch_pairs = pairs[start_idx:end_idx]
                
                if num_batches > 1:
                    print(f"BatchNanoBananaPro: 执行第 {batch_idx + 1}/{num_batches} 批 ({start_idx + 1}-{end_idx})...")
                
                # 创建当前批次的任务
                tasks = []
                for i, pair in enumerate(batch_pairs):
                    task = asyncio.create_task(
                        self._generate_single_task(
                            client=self.client,
                            session=session,
                            prompt=prompt,
                            model=model,
                            resolution=resolution,
                            aspect_ratio=aspect_ratio,
                            images=list(pair),
                            output_folder=output_folder,
                            task_index=start_idx + i
                        )
                    )
                    tasks.append(task)
                
                # 使用 as_completed 实时获取完成的任务
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
                            all_results.append(result_data)
                        else:
                            result_data = result
                            all_results.append(result)
                    except Exception as e:
                        result_data = {
                            "success": False,
                            "error": str(e),
                            "generated_count": 0,
                            "saved_files": []
                        }
                        all_results.append(result_data)
                    
                    completed += 1
                    
                    # 根据成功/失败状态打印不同信息
                    if result_data and result_data.get("success", False):
                        success_count += 1
                        print(f"BatchNanoBananaPro: 任务 {completed}/{total_tasks} 成功 ✓")
                    else:
                        fail_count += 1
                        # 提取错误信息的第一行
                        error_msg = result_data.get("error", "未知错误") if result_data else "未知错误"
                        # 截取第一行或前50个字符
                        if '\n' in error_msg:
                            error_msg = error_msg.split('\n')[0]
                        if len(error_msg) > 50:
                            error_msg = error_msg[:50] + "..."
                        print(f"BatchNanoBananaPro: 任务 {completed}/{total_tasks} 失败 ✗ - {error_msg}")
                    
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
        
        return all_results
    
    def process_batch(
        self,
        prompt: str,
        文件夹1: str,
        文件夹2: str,
        文件夹3: str,
        文件夹4: str,
        像素缩放: bool,
        分辨率像素: float,
        seed: int,
        保存路径: str,
        图片配对模式: str,
        模型: str,
        宽高比: str,
        分辨率: str,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        """
        批量处理图像生成任务
        
        Args:
            prompt: 提示词
            文件夹1-4: 图片文件夹路径
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
        
        try:
            # 设置随机种子（用于本地随机操作）
            random.seed(seed)
            np.random.seed(seed % (2**32))
            # 验证保存路径
            if not 保存路径 or not 保存路径.strip():
                raise ValueError("请提供保存路径")
            
            # 加载文件夹图片
            print("BatchNanoBananaPro: 开始加载图片...")
            image_lists = self._load_folders(
                文件夹1, 文件夹2, 文件夹3, 文件夹4,
                像素缩放, 分辨率像素
            )
            
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
            
            # 验证是否有图片
            total_folder_images = sum(len(lst) for lst in image_lists)
            total_manual_images = len(manual_images)
            
            if total_folder_images == 0 and total_manual_images == 0:
                raise ValueError("未找到任何图片，请检查文件夹路径或提供参考图")
            
            # 创建配对
            pairs = self._create_pairs(image_lists, 图片配对模式, manual_images if manual_images else None)
            
            if not pairs:
                raise ValueError("配对结果为空，请检查输入")
            
            total_tasks = len(pairs)
            
            # 打印首行概览
            print(f"BatchNanoBananaPro: 批量任务 | {图片配对模式} 配对模式 | 共 {total_tasks} 任务")
            
            # 创建 ComfyUI 原生进度条
            pbar = None
            if PROGRESS_BAR_AVAILABLE:
                pbar = ProgressBar(total_tasks)
            
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
                            pbar=pbar
                        )
                    )
                finally:
                    loop.close()
            
            # 使用线程池在新线程中运行事件循环
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_async_in_thread)
                results = future.result()
            
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
            
            # 精简统计信息
            print("=" * 60)
            print(f"完成！总耗时 {time_str} | 成功: {success_count}/{total_tasks} | 生成 {total_generated} 张")
            print(f"保存路径: {保存路径}")
            
            # 失败详情（如果有）
            failed_results = [r for r in results if not r.get("success", False)]
            if failed_results:
                # 收集失败任务的索引
                failed_indices = [str(r.get('task_index', '?') + 1) for r in failed_results[:5]]
                failed_str = ",".join(failed_indices)
                if len(failed_results) > 5:
                    failed_str += f"... (共{len(failed_results)}个)"
                # 显示第一个失败原因作为示例
                first_error = failed_results[0].get('error', '未知错误')
                # 截取错误信息的第一行
                if '\n' in first_error:
                    first_error = first_error.split('\n')[0]
                print(f"失败 {len(failed_results)}个: 任务{failed_str} - {first_error}")
            
            # 收集所有生成的图片
            output_images = []
            for file_path in all_saved_files:
                try:
                    img = Image.open(file_path)
                    output_images.append(img)
                except Exception as e:
                    print(f"BatchNanoBananaPro: 无法加载图片 {file_path} - {e}")
            
            # 如果没有生成成功的图片，创建一个占位图
            if not output_images:
                placeholder = Image.new('RGB', (512, 512), color=(128, 128, 128))
                output_images = [placeholder]
            
            # 转换为张量
            output_tensor = pil_to_tensor(output_images)
            
            return (output_tensor,)
        
        except ValueError as e:
            # 检测是否为授权错误
            if str(e) == "未授权！":
                print("请联系作者授权后方可使用！")
            else:
                print(f"BatchNanoBananaPro: 输入错误 - {str(e)}")
            raise
        
        except RuntimeError as e:
            print(f"BatchNanoBananaPro: 运行时错误 - {str(e)}")
            raise
        
        except Exception as e:
            print(f"BatchNanoBananaPro: 未知错误 - {str(e)}")
            raise
        
        finally:
            # 余额查询功能已停用（代码保留）
            # if self.client is not None:
            #     try:
            #         balance_data = self.client.query_balance_sync()
            #         balance_info = self.client.format_balance_info(balance_data)
            #         print(f"{balance_info}")
            #         print("=" * 50)
            #     except Exception as e:
            #         print(f"⚠️ 余额查询失败 - {str(e)}")
            #         print("=" * 50)
            pass