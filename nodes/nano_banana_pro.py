"""
Nano Banana Pro 节点
ComfyUI 自定义节点，用于调用 Gemini 模型生成图像
"""

import time
import random
from typing import Optional, Tuple

import torch
import numpy as np
from PIL import Image

from ..utils.image_utils import tensor_to_pil, pil_to_tensor, parse_batch_prompts
from ..clients.gemini_client import GeminiAPIClient
from ..models_config import (
    get_enabled_models, get_model_description,
    get_model_supported_aspect_ratios, get_all_supported_aspect_ratios,
    get_model_supported_resolutions, get_all_supported_resolutions
)

# 导入 ComfyUI 原生进度条
try:
    from comfy.utils import ProgressBar
    PROGRESS_BAR_AVAILABLE = True
except ImportError:
    PROGRESS_BAR_AVAILABLE = False
    print("⚠️ NanoBananaPro: comfy.utils.ProgressBar 不可用，将只使用终端进度显示")

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
    RESOLUTIONS = ["512", "1K", "2K", "4K"]
    
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
                "像素缩放": ("BOOLEAN", {
                    "default": True
                }),
                "分辨率像素": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "谷歌搜索（联网）": ("BOOLEAN", {
                    "default": True
                }),
                "图片搜索（联网）": ("BOOLEAN", {
                    "default": False
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
    
    def generate(
        self,
        prompt: str,
        模型: str,
        宽高比: str,
        分辨率: str,
        生图数量: int,
        像素缩放: bool,
        分辨率像素: float,
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
            像素缩放: 是否启用像素缩放
            分辨率像素: 目标像素数（百万像素）
            seed: 随机种子
            **kwargs: 搜索开关（谷歌搜索（联网）/ 图片搜索（联网））及动态参考图输入 (参考图1-9)
                      注：两个搜索参数名含全角括号，不能作为 Python 形参，从 kwargs 中提取
            
        注意：
            调试日志功能已移至文件顶部配置，通过修改 DEBUG_LOG_ENABLED 常量控制
        
        Returns:
            生成的图像张量 (IMAGE,)
        """
        start_time = time.time()
        
        # 从 kwargs 提取搜索参数（参数名含全角括号，无法直接声明为 Python 形参）
        enable_grounding: bool = kwargs.pop("谷歌搜索（联网）", True)
        enable_image_search: bool = kwargs.pop("图片搜索（联网）", False)
        
        # 创建 ComfyUI 原生进度条
        pbar = None
        if PROGRESS_BAR_AVAILABLE:
            pbar = ProgressBar(生图数量)
        
        try:
            # 设置随机种子（用于本地随机操作）
            random.seed(seed)
            np.random.seed(seed % (2**32))
            
            # 初始化 API 客户端
            if self.client is None:
                try:
                    self.client = GeminiAPIClient()
                except ValueError as e:
                    raise ValueError(f"初始化失败: {str(e)}")
            
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
            
            # 应用像素缩放（如果启用）
            if input_images and 像素缩放:
                scaled_images = []
                for img in input_images:
                    scaled = self.resize_to_megapixels(img, 分辨率像素)
                    scaled_images.append(scaled)
                input_images = scaled_images
            
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
            else:
                # 单提示词模式
                mode_str = f"图生图模式 (输入{len(input_images)}张)" if input_images else "文生图模式"
                print(f"Nano Banana Pro: {mode_str} | {分辨率} {宽高比} | {生图数量}张{grounding_str}")
            
            # 统计变量
            success_count = 0
            fail_count = 0
            
            # 进度回调 - 静默模式，只更新 ComfyUI 进度条
            def progress_callback(current, total, success, error_msg=None):
                nonlocal success_count, fail_count
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                
                # 更新 ComfyUI 原生进度条
                if pbar is not None:
                    pbar.update(1)
            
            # 根据是否有批量提示词选择生成模式
            if batch_prompts:
                # 批量提示词模式
                num_prompts = len(batch_prompts)
                total_images = num_prompts * 生图数量
                
                # 重新创建进度条以匹配实际总数
                if pbar is not None:
                    pbar = ProgressBar(total_images)
                
                generated_images = self.client.generate_multi_prompts_sync(
                    prompts=batch_prompts,
                    model=模型,
                    resolution=分辨率,
                    aspect_ratio=宽高比,
                    images_per_prompt=生图数量,
                    images=input_images,
                    progress_callback=progress_callback,
                    debug=DEBUG_LOG_ENABLED,
                    debug_request=REQUEST_LOG_ENABLED,
                    enable_grounding=enable_grounding,
                    enable_image_search=enable_image_search
                )
            else:
                # 单提示词模式
                generated_images = self.client.generate_sync(
                    prompt=prompt,
                    model=模型,
                    resolution=分辨率,
                    aspect_ratio=宽高比,
                    batch_size=生图数量,
                    images=input_images,
                    progress_callback=progress_callback,
                    debug=DEBUG_LOG_ENABLED,
                    debug_request=REQUEST_LOG_ENABLED,
                    enable_grounding=enable_grounding,
                    enable_image_search=enable_image_search
                )
            
            # 转换输出图像
            output_tensor = pil_to_tensor(generated_images)
            
            # 计算耗时并打印最终统计
            elapsed = time.time() - start_time
            if elapsed < 1:
                time_str = f"{elapsed:.3f}s"
            else:
                time_str = f"{elapsed:.2f}s"
            
            # 打印最终汇总
            if fail_count > 0:
                print(f"[4/4] 完成！总耗时 {time_str} | 成功 {success_count}张 | 失败 {fail_count}张")
            else:
                print(f"[4/4] 完成！总耗时 {time_str} | 成功 {len(generated_images)}张")
            
            return (output_tensor,)
        
        except ValueError as e:
            # 检测是否为授权错误
            if str(e) == "未授权！":
                print("请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            else:
                # 用户输入错误 - 只显示简洁信息
                error_msg = str(e).split('\n')[0]  # 只取第一行
                print(f"Nano Banana Pro: ❌ {error_msg}")
            raise ValueError(error_msg) from None
        
        except RuntimeError as e:
            # API 或网络错误 - 只显示简洁信息
            error_msg = str(e).split('\n')[0]  # 只取第一行
            print(f"Nano Banana Pro: ❌ {error_msg}")
            raise RuntimeError(error_msg) from None
        
        except Exception as e:
            # 其他未知错误 - 只显示简洁信息
            error_msg = str(e).split('\n')[0]
            print(f"Nano Banana Pro: ❌ {error_msg}")
            raise type(e)(error_msg) from None
        
        finally:
            if self.client is not None:
                try:
                    balance_data = self.client.query_balance_sync()
                    balance_info = self.client.format_balance_info(balance_data)
                    print(f"Nano Banana Pro: {balance_info}")
                except Exception:
                    pass