"""
Nano Banana Pro 节点
ComfyUI 自定义节点，用于调用 Gemini 模型生成图像
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
from ..clients.gemini_client import GeminiAPIClient
from ..models_config import (
    get_enabled_models, get_model_description,
    get_model_supported_aspect_ratios, get_all_supported_aspect_ratios,
    get_model_supported_resolutions, get_all_supported_resolutions
)

# 检查 folder_paths 是否可用
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

# 导入 ComfyUI 原生进度条
try:
    from comfy.utils import ProgressBar
    PROGRESS_BAR_AVAILABLE = True
except ImportError:
    PROGRESS_BAR_AVAILABLE = False
    print("⚠️ NanoBananaPro: comfy.utils.ProgressBar 不可用，将只使用终端进度显示")

# 内存监控（可选）
try:
    import psutil
    MEMORY_MONITOR_AVAILABLE = True
except ImportError:
    MEMORY_MONITOR_AVAILABLE = False
    print("⚠️ NanoBananaPro: psutil 不可用，内存监控功能禁用")

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

_NODE = "Nano Banana Pro"


def _images_to_tensor_safe(images: List[Image.Image], node_label: str) -> torch.Tensor:
    """
    将 PIL Image 列表转换为 ComfyUI tensor，安全处理多张不同尺寸的情况。

    策略：
    - 以像素数最大的图尺寸为基准
    - 只输出与最大尺寸相同的图，其余较小的图丢弃
    """
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


class NanoBananaPro:
    """
    Nano Banana Pro 节点
    
    功能：
    - 文生图：基于提示词生成图像
    - 图生图：基于输入图像和提示词生成新图像
    - 批量生成：支持并发生成多张图像
    
    注意：
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
    RESOLUTIONS = ["512px", "1K", "2K", "4K"]
    
    def __init__(self):
        """初始化节点"""
        self.client = None
    
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

        optional_inputs["代理端口（如7897）"] = ("STRING", {
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
                "图片搜索（联网）": (["关闭", "打开"], {
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
    
    # 返回值类型
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    
    # 导入 ComfyUI 的文件夹路径管理
    try:
        import folder_paths
        FOLDER_PATHS_AVAILABLE = True
    except ImportError:
        FOLDER_PATHS_AVAILABLE = False

    # 执行函数名
    FUNCTION = "generate"
    
    # 节点分类
    CATEGORY = "image/generation"
    
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
    
    def validate_inputs(
        self,
        images: Optional[torch.Tensor],
        batch_size: int
    ) -> None:
        """
        验证输入参数
        
        Args:
            images: 输入图像张量（可选）
            batch_size: 批次大小
        
        Raises:
            ValueError: 如果输入参数不合法
        """
        # 检查图像数量
        if images is not None:
            num_images = images.shape[0]
            if num_images > 14:
                raise ValueError(
                    f"输入图像数量 {num_images} 超过限制 14 张，请减少输入图像数量"
                )
        
        # 检查批次大小
        if batch_size < 1 or batch_size > 1000:
            raise ValueError(
                f"批次大小 {batch_size} 超出范围 [1, 1000]"
            )
    
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
        save_to_disk: bool = True,
    ) -> dict:
        """执行单个生成任务"""
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
        enable_image_search: bool = False,
        save_to_disk: bool = True,
    ) -> List[dict]:
        """异步批量处理：每个提示词独立调用 API"""
        # 构建任务列表：(prompt, sub_index) 用于 images_per_prompt > 1 的情况
        tasks_def = []
        for p_idx, prompt in enumerate(prompts):
            for sub_idx in range(images_per_prompt):
                tasks_def.append((p_idx, sub_idx, prompt))
        
        total_tasks = len(tasks_def)
        num_prompts = len(prompts)

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
                            enable_image_search=enable_image_search,
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
                        print(f"Nano Banana Pro: [{completed}/{total_tasks}] {prompt_snippet}{'...' if len(prompt_snippet) >= 30 else ''} → ✓成功({count}张)")
                    else:
                        fail_count += 1
                        error_msg = result_data.get("error", "未知错误") if result_data else "未知错误"
                        print(f"Nano Banana Pro: [{completed}/{total_tasks}] {prompt_snippet}{'...' if len(prompt_snippet) >= 30 else ''} → ✗失败: {error_msg}")
                    
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
        """
        生成图像

        Args:
            prompt: 提示词
            模型: 模型名称
            宽高比: 宽高比
            分辨率: 分辨率
            生图数量: 批次大小
            seed: 随机种子
            **kwargs: 搜索开关（谷歌搜索（联网）/ 图片搜索（联网））及动态参考图输入 (参考图1-9)
                      注：两个搜索参数名含全角括号，不能作为 Python 形参，从 kwargs 中提取
            
        注意：
            调试日志功能已移至文件顶部配置，通过修改 DEBUG_LOG_ENABLED 常量控制
        
        Returns:
            生成的图像张量 (IMAGE,)
        """
        start_time = time.time()
        
        # 从 kwargs 提取搜索参数（界面显示为「关闭/打开」，转为 bool 供调用）
        enable_grounding: bool = (kwargs.pop("谷歌搜索（联网）", "关闭") == "打开")
        enable_image_search: bool = (kwargs.pop("图片搜索（联网）", "关闭") == "打开")
        proxy_port: str = kwargs.pop("代理端口（如7897）", "")

        # 创建 ComfyUI 原生进度条
        pbar = None
        if PROGRESS_BAR_AVAILABLE:
            pbar = ProgressBar(生图数量)
        
        try:
            # 设置随机种子（用于本地随机操作）
            random.seed(seed)
            np.random.seed(seed % (2**32))
            
            # 内存监控初始化
            if MEMORY_MONITOR_AVAILABLE and 生图数量 > 50:
                import psutil
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024
                print(f"Nano Banana Pro: 初始内存使用: {initial_memory:.1f} MB")
            
            # 初始化 API 客户端
            if self.client is None:
                try:
                    self.client = GeminiAPIClient()
                except ValueError as e:
                    raise ValueError(f"初始化失败: {str(e)}")

            # 注入代理设置（每次执行都刷新，支持用户中途修改端口）
            self.client.proxy_url = GeminiAPIClient.build_proxy_url(proxy_port)
            if self.client.proxy_url:
                print(f"Nano Banana Pro: 已启用代理加速 → {self.client.proxy_url}")
            
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
            # 仅 nano-banana-2-限时特价 和 gemini-3.1-flash-image-preview 支持图片搜索
            IMAGE_SEARCH_UNSUPPORTED_MODELS = ["nano-banana-pro-限时特价", "nano-banana-pro-官方计费", "gemini-3-pro-image-preview"]
            if enable_image_search and 模型 in IMAGE_SEARCH_UNSUPPORTED_MODELS:
                raise ValueError(
                    f"模型 \"{模型}\" 不支持【图片搜索（联网）】功能！"
                    f"请切换到 nano-banana-2-限时特价 或 gemini-3.1-flash-image-preview 后再使用"
                )
            
            # 收集独立输入的参考图
            input_images = []
            for i in range(1, 10):  # 1-9
                key = f"参考图{i}"
                if key in kwargs and kwargs[key] is not None:
                    pil_imgs = tensor_to_pil(kwargs[key])
                    input_images.extend(pil_imgs)
            
            # 验证输入图像数量
            if input_images:
                if len(input_images) > 14:
                    raise ValueError(
                        f"输入图像数量 {len(input_images)} 超过限制 14 张，请减少输入图像数量"
                    )

            # 解析批量提示词
            batch_prompts = parse_batch_prompts(prompt)
            
            # 打印首行概览
            # 图片搜索（联网）开启时隐含谷歌搜索接地，与客户端请求逻辑保持一致
            grounding_str = ""
            if enable_image_search:
                grounding_str = " | 谷歌图片搜索接地"
            elif enable_grounding:
                grounding_str = " | 谷歌搜索接地"
            
            if batch_prompts:
                # 批量提示词模式
                num_prompts = len(batch_prompts)
                total_images = num_prompts * 生图数量
                mode_str = f"批量提示词模式 ({num_prompts}个提示词)"
                if input_images:
                    mode_str += f" (输入{len(input_images)}张)"
                print(f"Nano Banana Pro: {mode_str} | {分辨率} {宽高比} | 共{total_images}张{grounding_str}")
                
                # 大批量警告
                if total_images > 100:
                    print(f"⚠️ Nano Banana Pro: 警告！批量生成 {total_images} 张图片，内存占用可能较高")
                    print(f"⚠️ 建议：分批执行或减少生图数量")
            else:
                # 单提示词模式
                mode_str = f"图生图模式 (输入{len(input_images)}张)" if input_images else "文生图模式"
                print(f"Nano Banana Pro: {mode_str} | {分辨率} {宽高比} | {生图数量}张{grounding_str}")
                
                # 大批量警告
                if 生图数量 > 100:
                    print(f"⚠️ Nano Banana Pro: 警告！批量生成 {生图数量} 张图片，内存占用可能较高")
                    print(f"⚠️ 建议：分批执行或减少生图数量")
            
            # 统计变量
            success_count = 0
            fail_count = 0
            
            # 进度回调 - 打印错误信息并更新进度条，添加内存监控
            def progress_callback(current, total, success, error_msg=None):
                nonlocal success_count, fail_count
                if success:
                    success_count += 1
                else:
                    fail_count += 1

                # 更新 ComfyUI 原生进度条
                if pbar is not None:
                    pbar.update(1)

                # 内存监控（每完成10个任务检查一次）
                if MEMORY_MONITOR_AVAILABLE and total > 50 and current % 10 == 0:
                    import gc
                    gc.collect()  # 强制垃圾回收
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    print(f"Nano Banana Pro: 内存使用: {current_memory:.1f} MB (+{memory_increase:.1f} MB)")
                    
                    # 内存警告阈值（2GB）
                    if current_memory > 2000:
                        print(f"⚠️ Nano Banana Pro: 内存使用过高！建议减少生图数量或分批执行")
            
            # 根据是否有批量提示词选择生成模式
            if batch_prompts:
                num_prompts = len(batch_prompts)
                total_images = num_prompts * 生图数量
                
                # ===== 批量提示词模式：异步并发，内存输出 =====
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
                                enable_image_search=enable_image_search,
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

                # 统计结果
                success_count = sum(1 for r in results if r.get("success", False))
                fail_count = len(results) - success_count
                total_generated = sum(r.get("generated_count", 0) for r in results)

                elapsed = time.time() - start_time
                time_str = f"{elapsed:.3f}s" if elapsed < 1 else f"{elapsed:.2f}s"

                print(f"完成！总耗时 {time_str} | 成功: {success_count}/{total_images} | 失败: {fail_count}")

                # 失败详情
                failed_results = [r for r in results if not r.get("success", False)]
                if failed_results:
                    for fr in failed_results:
                        idx = fr.get("global_task_index", -1) + 1
                        prompt_snippet = (fr.get("prompt", "") or "")[:30]
                        error_msg = fr.get("error", "未知错误")
                        print(f"  失败 #{idx}: {prompt_snippet}{'...' if len(prompt_snippet) >= 30 else ''} → {error_msg}")

                # 收集内存中的图像
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
                # 单提示词模式
                if 生图数量 == 1:
                    # 单张：同步生成，自动重试（429/503/504）
                    _RETRY_CODES = ("429", "503", "504")
                    _MAX_RETRIES = 5
                    for _attempt in range(1, _MAX_RETRIES + 1):
                        try:
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
                            break
                        except RuntimeError as e:
                            error_msg = str(e)
                            if any(code in error_msg for code in _RETRY_CODES) and _attempt < _MAX_RETRIES:
                                _wait = 2 ** _attempt
                                print(f"{'=' * 60}")
                                print(f"⚠️  Nano Banana Pro 自动重试 [{_attempt}/{_MAX_RETRIES - 1}]")
                                print(f"   原因：{error_msg}")
                                print(f"   等待 {_wait}s 后重试...")
                                print(f"{'=' * 60}")
                                time.sleep(_wait)
                            else:
                                raise
                else:
                    # 多张：异步并发，内存输出

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
                                    enable_image_search=enable_image_search,
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

                    # 失败详情
                    failed_results = [r for r in results if not r.get("success", False)]
                    if failed_results:
                        for fr in failed_results:
                            idx = fr.get("global_task_index", -1) + 1
                            error_msg = fr.get("error", "未知错误")
                            print(f"  失败 #{idx}: {prompt[:30]}{'...' if len(prompt) >= 30 else ''} → {error_msg}")

                    # 收集内存中的图像
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

                
            # 优化：限制输出图片数量，避免内存爆炸
            max_output_images = 20  # 最多输出20张图片到ComfyUI
            
            if len(generated_images) > max_output_images:
                print(f"Nano Banana Pro: 生成 {len(generated_images)} 张图片，限制输出前 {max_output_images} 张到ComfyUI")
                output_images = generated_images[:max_output_images]
            else:
                output_images = generated_images
            
            # 转换输出图像
            output_tensor = _images_to_tensor_safe(output_images, _NODE)
            
            # 计算耗时并打印最终统计
            elapsed = time.time() - start_time
            if elapsed < 1:
                time_str = f"{elapsed:.3f}s"
            else:
                time_str = f"{elapsed:.2f}s"
            
            # 打印最终汇总
            if fail_count > 0:
                print(f"完成！总耗时 {time_str} | 成功 {success_count}张 | 失败 {fail_count}张")
            else:
                print(f"完成！总耗时 {time_str} | 成功 {len(generated_images)}张")
            
            # 最终内存清理
            import gc
            gc.collect()
            if MEMORY_MONITOR_AVAILABLE and 生图数量 > 50:
                final_memory = process.memory_info().rss / 1024 / 1024
                print(f"Nano Banana Pro: 最终内存使用: {final_memory:.1f} MB")

            return (output_tensor,)
        
        except ValueError as e:
            # 检测是否为授权错误
            if str(e) == "未授权！":
                print("请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            raise ValueError(str(e)) from None

        except RuntimeError as e:
            raise RuntimeError(str(e)) from None

        except Exception as e:
            raise type(e)(str(e)) from None
        
        finally:
            # 查询余额
            if self.client is not None:
                try:
                    balance_data = self.client.query_balance_sync()
                    balance_info = self.client.format_balance_info(balance_data)
                    print(f"Nano Banana Pro: {balance_info}")
                except Exception:
                    pass

            # 最终内存清理
            import gc
            gc.collect()