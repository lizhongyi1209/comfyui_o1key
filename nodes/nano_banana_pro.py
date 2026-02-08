"""
Nano Banana Pro 节点
ComfyUI 自定义节点，用于调用 Gemini 3 Pro 模型生成图像
"""

import time
import random
from typing import Optional, Tuple

import torch
import numpy as np
from PIL import Image

from ..utils.image_utils import tensor_to_pil, pil_to_tensor, parse_batch_prompts
from ..clients.gemini_client import GeminiAPIClient
from ..models_config import get_enabled_models, get_model_description

# 导入 ComfyUI 原生进度条
try:
    from comfy.utils import ProgressBar
    PROGRESS_BAR_AVAILABLE = True
except ImportError:
    PROGRESS_BAR_AVAILABLE = False
    print("⚠️ NanoBananaPro: comfy.utils.ProgressBar 不可用，将只使用终端进度显示")


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
    
    # 支持的宽高比列表
    ASPECT_RATIOS = [
        "1:1", "4:3", "3:4", "16:9", "9:16",
        "2:3", "3:2", "4:5", "5:4", "21:9"
    ]
    
    # 支持的分辨率列表
    RESOLUTIONS = ["1K", "2K", "4K"]
    
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
                "生图数量": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000,
                    "step": 1
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
            **kwargs: 动态参考图输入 (参考图1-9)
        
        Returns:
            生成的图像张量 (IMAGE,)
        """
        start_time = time.time()
        
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
            if batch_prompts:
                # 批量提示词模式
                num_prompts = len(batch_prompts)
                total_images = num_prompts * 生图数量
                mode_str = f"批量提示词模式 ({num_prompts}个提示词)"
                if input_images:
                    mode_str += f" (输入{len(input_images)}张)"
                print(f"Nano Banana Pro: {mode_str} | {分辨率} {宽高比} | 共{total_images}张")
            else:
                # 单提示词模式
                mode_str = f"图生图模式 (输入{len(input_images)}张)" if input_images else "文生图模式"
                print(f"Nano Banana Pro: {mode_str} | {分辨率} {宽高比} | {生图数量}张")
            
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
                    progress_callback=progress_callback
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
                    progress_callback=progress_callback
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
            else:
                # 用户输入错误
                print(f"Nano Banana Pro: 输入错误 - {str(e)}")
            raise
        
        except RuntimeError as e:
            # API 或网络错误
            print(f"Nano Banana Pro: API 错误 - {str(e)}")
            raise
        
        except Exception as e:
            # 其他未知错误
            print(f"Nano Banana Pro: 未知错误 - {str(e)}")
            raise
        
        finally:
            # 余额查询功能已停用（代码保留）
            # if self.client is not None:
            #     try:
            #         balance_data = self.client.query_balance_sync()
            #         balance_info = self.client.format_balance_info(balance_data)
            #         print(f"Nano Banana Pro: {balance_info}")
            #     except Exception as e:
            #         print(f"Nano Banana Pro: ⚠️ 余额查询失败 - {str(e)}")
            pass