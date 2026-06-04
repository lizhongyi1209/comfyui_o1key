"""
o1key GPT Image 节点
支持 gpt-image-1 / gpt-image-1.5 模型的文生图、图生图、图像编辑（带蒙版）
"""

import os
import time
from typing import List, Optional, Tuple

from PIL import Image

from ..clients.gpt_image_client import GptImageClient
from ..utils.image_utils import parse_batch_prompts, pil_to_tensor, tensor_to_pil
from ..utils.config import NETWORK_ROUTE_OPTIONS, get_base_url_by_route
from ..utils.file_utils import (
    ImageInfo,
    generate_timestamp_filename,
    load_images_from_folder,
    pair_images_by_name,
    pair_images_cartesian,
    save_image,
)

try:
    from comfy.model_management import processing_interrupted, InterruptProcessingException
    _INTERRUPT_AVAILABLE = True
except ImportError:
    _INTERRUPT_AVAILABLE = False
    processing_interrupted = lambda: False
    InterruptProcessingException = RuntimeError

try:
    from comfy.utils import ProgressBar
    _PROGRESS_BAR_AVAILABLE = True
except ImportError:
    _PROGRESS_BAR_AVAILABLE = False

try:
    import folder_paths
    _FOLDER_PATHS_AVAILABLE = True
except ImportError:
    _FOLDER_PATHS_AVAILABLE = False


def _make_node_progress_callback(progress_bar, task_index: int, total_tasks: int):
    if progress_bar is None:
        return None

    total_units = max(1, total_tasks) * 100
    base_units = max(0, task_index - 1) * 100
    last_pct = {"value": -1}

    def _callback(pct: int):
        try:
            pct_value = int(round(float(pct)))
        except (TypeError, ValueError):
            return
        pct_value = max(0, min(100, pct_value))
        if pct_value < last_pct["value"]:
            return
        last_pct["value"] = pct_value
        progress_bar.update_absolute(
            min(total_units, base_units + pct_value),
            total_units,
        )

    return _callback


def _resolve_async_size(value: str) -> str:
    value = (value or "").strip()
    if not value or value == "智能" or value.lower() == "auto":
        return "auto"

    first_part = value.split("（")[0].strip()
    normalized_size = first_part.lower().replace("*", "x").replace("×", "x")
    size_parts = [part.strip() for part in normalized_size.split("x")]
    if len(size_parts) == 2 and all(part.isdigit() for part in size_parts):
        return f"{int(size_parts[0])}x{int(size_parts[1])}"

    allowed = {"auto", "1024x1024", "1K", "2K", "4K"}
    if first_part in allowed:
        return first_part

    if "4K" in value:
        return "4K"
    if "2K" in value:
        return "2K"
    if "1K" in value:
        return "1K"

    return "auto"


class O1keyGPTImage:
    """
    o1key GPT Image 节点

    功能：
      - 文生图：仅提供 prompt
      - 图生图：提供 prompt + 图片（无遮罩）
      - 图像编辑：提供 prompt + 图片 + 遮罩（白色区域将被替换）
      - 批量模式：prompt 中用单独一行 --- 分隔多条提示词

    参数：
      - prompt   : 文本提示词（多行；用 --- 独占一行分隔批量提示词）
      - 模型     : 模型选择
      - 分辨率   : 图像尺寸（auto 让 API 自动决定）
      - 生图数量 : 每条提示词生成数量 1-8
      - 质量     : 生成质量
      - seed     : 随机种子（0 表示不指定）
      - 图片     : 可选参考图（用于图生图或编辑）
      - 遮罩     : 可选蒙版（白色区域将被替换）
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 创建9个独立的参考图输入
        optional_inputs = {}
        for i in range(1, 10):
            optional_inputs[f"参考图{i}"] = ("IMAGE", {
                "tooltip": f"Optional reference image {i} for image editing.",
            })

        optional_inputs["模型"] = ([
            "gpt-image-2-按量",
            "gpt-image-2-次卡",
        ], {
            "default": "gpt-image-2-次卡",
        })
        optional_inputs["网络"] = (NETWORK_ROUTE_OPTIONS, {
            "default": "全球加速",
        })
        optional_inputs["分辨率"] = ([
            "智能",
            # ── 1K ──
            "1024x1024（1K 正方形 1:1）",
            "1536x1024（1K 横版 3:2）",
            "1024x1536（1K 竖版 2:3）",
            "1360x1024（1K 横版 4:3）",
            "1024x1360（1K 竖版 3:4）",
            "1824x1024（1K 横版 16:9）",
            "1024x1824（1K 竖版 9:16）",
            # ── 2K ──
            "2048x2048（2K 正方形 1:1）",
            "3072x2048（2K 横版 3:2）",
            "2048x3072（2K 竖版 2:3）",
            "2736x2048（2K 横版 4:3）",
            "2048x2736（2K 竖版 3:4）",
            "3648x2048（2K 横版 16:9）",
            "2048x3648（2K 竖版 9:16）",
            # ── 4K ──
            "2880x2880（4K 正方形 1:1）",
            "3504x2336（4K 横版 3:2）",
            "2336x3504（4K 竖版 2:3）",
            "3264x2448（4K 横版 4:3）",
            "2448x3264（4K 竖版 3:4）",
            "3840x2160（4K 横版 16:9）",
            "2160x3840（4K 竖版 9:16）",
        ], {
            "default": "智能",
            "tooltip": "Image size (智能 = API decides)",
        })
        optional_inputs["生图数量"] = ("INT", {
            "default": 1,
            "min": 1,
            "max": 8,
            "step": 1,
            "display": "number",
            "tooltip": "How many images to generate per prompt",
        })
        optional_inputs["质量"] = (["高", "中", "低", "自动"], {
            "default": "自动",
            "tooltip": "Image quality: 高=high, 中=medium, 低=low, 自动=auto",
        })
        optional_inputs["输出格式"] = (["png", "jpeg", "webp"], {
            "default": "jpeg",
            "tooltip": "Generated image output format",
        })
        optional_inputs["seed"] = ("INT", {
            "default": 0,
            "min": 0,
            "max": 2**31 - 1,
            "step": 1,
            "display": "number",
            "control_after_generate": True,
            "tooltip": "Random seed (0 = not specified)",
        })
        optional_inputs["遮罩"] = ("MASK", {
            "tooltip": "Optional mask for inpainting (white areas will be replaced)",
        })

        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text prompt for GPT Image. Use --- on its own line to separate batch prompts.",
                }),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "o1key/image"
    OUTPUT_NODE = False

    def generate(
        self,
        prompt: str,
        模型: str = "gpt-image-2-次卡",
        网络: str = "全球加速",
        分辨率: str = "auto",
        质量: str = "自动",
        输出格式: str = "jpeg",
        生图数量: int = 1,
        seed: int = 0,
        遮罩=None,
        **kwargs,
    ):
        """
        生成图像（文生图 / 图生图 / 图像编辑 / 批量提示词）

        路由逻辑：
          - 无图片           → generations 接口（文生图）
          - 有图片，无遮罩   → edits 接口（图生图）
          - 有图片，有遮罩   → edits 接口（图像编辑 + 蒙版）
          - prompt 含 ---    → 批量模式，逐条调用上述接口
        """
        start_time = time.time()

        # ── 0. 收集多参考图输入 ────────────────────────────────────────────────
        reference_tensors = []
        for i in range(1, 10):
            key = f"参考图{i}"
            if key in kwargs and kwargs[key] is not None:
                reference_tensors.append(kwargs[key])

        图片 = reference_tensors if reference_tensors else None

        # ── 1. 参数校验 ───────────────────────────────────────────────────────
        if 遮罩 is not None and 图片 is None:
            raise ValueError("提供了遮罩但未提供图片，请同时提供图片和遮罩")

        # ── 2. 解析分辨率显示值 → API 参数值 ──────────────────────────────────
        size = _resolve_async_size(分辨率)

        # ── 2b. 解析模型显示值 → API 参数值 ───────────────────────────────────
        _model_map = {"gpt-image-2-次卡": "gpt-image-2-c", "gpt-image-2-按量": "gpt-image-2"}
        model = _model_map.get(模型, 模型)

        # ── 2c. 解析质量显示值 → API 参数值 ───────────────────────────────────
        _quality_map = {"高": "high", "中": "medium", "低": "low", "自动": "auto"}
        quality = _quality_map.get(质量, "auto")

        # ── 3. 创建客户端 ─────────────────────────────────────────────────────
        try:
            client = GptImageClient()
            client.base_url = get_base_url_by_route(网络)
        except ValueError as e:
            if str(e) == "未授权！":
                print("[o1key GPT Image] 请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            raise

        try:
            # ── 4. 解析批量提示词 ─────────────────────────────────────────────
            batch_prompts = parse_batch_prompts(prompt)

            # ── 5. 调用 API ───────────────────────────────────────────────────
            all_pil_images = []
            progress_total = len(batch_prompts) if batch_prompts else 1
            progress_bar = ProgressBar(progress_total * 100) if _PROGRESS_BAR_AVAILABLE else None

            if batch_prompts:
                # 批量模式：逐条提示词调用
                total = len(batch_prompts)
                print(f"[o1key GPT Image] 批量模式 | {total} 条提示词 | 每条生成 {生图数量} 张")
                for idx, p in enumerate(batch_prompts, 1):
                    if _INTERRUPT_AVAILABLE and processing_interrupted():
                        print("[o1key GPT Image] 用户取消，已中断批量生成")
                        raise InterruptProcessingException()
                    try:
                        pil_images = client.generate_image_async_sync(
                            prompt=p,
                            model=model,
                            quality=quality,
                            size=size,
                            n=生图数量,
                            seed=seed,
                            image_tensor=图片,
                            mask_tensor=遮罩,
                            output_format=输出格式,
                            progress_callback=_make_node_progress_callback(progress_bar, idx, total),
                        )
                        all_pil_images.extend(pil_images)
                        snippet = p[:30] + ("..." if len(p) >= 30 else "")
                        print(f"[o1key GPT Image] [{idx}/{total}] ✓ {snippet}")
                    except InterruptProcessingException:
                        raise
                    except Exception as e:
                        error_msg = str(e).split('\n')[0]
                        snippet = p[:30] + ("..." if len(p) >= 30 else "")
                        print(f"[o1key GPT Image] [{idx}/{total}] ❌ {snippet} → {error_msg}")
                    if progress_bar is not None:
                        progress_bar.update_absolute(idx * 100, total * 100)
            else:
                # 单提示词模式
                if not prompt or not prompt.strip():
                    raise ValueError("提示词不能为空")
                try:
                    pil_images = client.generate_image_async_sync(
                        prompt=prompt,
                        model=model,
                        quality=quality,
                        size=size,
                        n=生图数量,
                        seed=seed,
                        image_tensor=图片,
                        mask_tensor=遮罩,
                        output_format=输出格式,
                        progress_callback=_make_node_progress_callback(progress_bar, 1, 1),
                    )
                    all_pil_images.extend(pil_images)
                except InterruptProcessingException:
                    raise
                except Exception as e:
                    error_msg = str(e).split('\n')[0]
                    print(f"[o1key GPT Image] ❌ {error_msg}")
                    raise RuntimeError(error_msg) from None

            # ── 6. 检查是否有可用图像 ─────────────────────────────────────────
            if not all_pil_images:
                raise RuntimeError("所有提示词均生成失败，无可用图像输出")

            # ── 7. PIL → tensor ───────────────────────────────────────────────
            output_tensor = GptImageClient._pil_list_to_tensor(all_pil_images)

            # ── 8. 完成日志 ───────────────────────────────────────────────────
            elapsed = time.time() - start_time
            print(
                f"[o1key GPT Image] 完成！耗时 {elapsed:.1f}s，"
                f"输出 {output_tensor.shape[0]} 张 "
                f"{output_tensor.shape[2]}×{output_tensor.shape[1]}"
            )

            return (output_tensor,)

        finally:
            self._print_balance(client)

    def _print_balance(self, client):
        try:
            balance_data = client.query_balance_sync()
            balance_info = client.format_balance_info(balance_data)
            print(f"[o1key GPT Image] {balance_info}")
        except Exception:
            pass


class O1keyGPTImageBatch:
    """
    o1key GPT Image 批量节点

    复用 BatchNanoBananaPro 的批量思路：
      - 从文件夹批量加载图片
      - 按文件名同名 / 1*N / 不配对 三种模式创建任务
      - 可追加节点手动输入参考图
      - prompt 支持用独占一行 --- 展开为多提示词任务
      - 每个任务调用 GPT Image 客户端并保存到磁盘
    """

    PAIRING_MODES = ["按相同图片命名", "1*N", "不配对"]
    IMAGE_FORMATS = ["原始", "JPEG", "PNG", "WebP"]
    MODEL_OPTIONS = ["gpt-image-2-按量", "gpt-image-2-次卡"]
    QUALITY_OPTIONS = ["高", "中", "低", "自动"]
    RESOLUTION_OPTIONS = [
        "智能",
        "1024x1024（1K 正方形 1:1）",
        "1536x1024（1K 横版 3:2）",
        "1024x1536（1K 竖版 2:3）",
        "1360x1024（1K 横版 4:3）",
        "1024x1360（1K 竖版 3:4）",
        "1824x1024（1K 横版 16:9）",
        "1024x1824（1K 竖版 9:16）",
        "2048x2048（2K 正方形 1:1）",
        "3072x2048（2K 横版 3:2）",
        "2048x3072（2K 竖版 2:3）",
        "2736x2048（2K 横版 4:3）",
        "2048x2736（2K 竖版 3:4）",
        "3648x2048（2K 横版 16:9）",
        "2048x3648（2K 竖版 9:16）",
        "2880x2880（4K 正方形 1:1）",
        "3504x2336（4K 横版 3:2）",
        "2336x3504（4K 竖版 2:3）",
        "3264x2448（4K 横版 4:3）",
        "2448x3264（4K 竖版 3:4）",
        "3840x2160（4K 横版 16:9）",
        "2160x3840（4K 竖版 9:16）",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {}
        for image_index in range(1, 10):
            optional_inputs[f"参考图{image_index}"] = ("IMAGE", {
                "tooltip": "追加到每个批量任务末尾的固定参考图。",
            })

        optional_inputs["遮罩"] = ("MASK", {
            "tooltip": "可选蒙版，会应用到每个任务的第一张参考图；请确保尺寸一致。",
        })
        optional_inputs["图片配对模式"] = (cls.PAIRING_MODES, {
            "default": "不配对",
            "tooltip": "文件夹图片的组合方式；手动参考图只追加，不参与配对。",
        })

        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "提示词；可用独占一行的 --- 分隔多条批量提示词。",
                }),
                "模型": (cls.MODEL_OPTIONS, {
                    "default": "gpt-image-2-次卡",
                }),
                "网络": (NETWORK_ROUTE_OPTIONS, {
                    "default": "全球加速",
                }),
                "分辨率": (cls.RESOLUTION_OPTIONS, {
                    "default": "智能",
                }),
                "生图数量": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "display": "number",
                }),
                "质量": (cls.QUALITY_OPTIONS, {
                    "default": "自动",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "step": 1,
                    "display": "number",
                    "control_after_generate": True,
                }),
                "图片格式": (cls.IMAGE_FORMATS, {
                    "default": "原始",
                }),
                "文件夹1": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "文件夹2": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "文件夹3": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "文件夹4": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "文件夹5": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "保存路径": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "为空时优先使用 ComfyUI 默认 output 目录。",
                }),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "process_batch"
    CATEGORY = "o1key/image"
    OUTPUT_NODE = False

    def _load_folders(self, folders: List[str]) -> List[List[ImageInfo]]:
        image_lists = []
        for folder_index, folder in enumerate(folders, 1):
            if not folder or not folder.strip():
                continue
            try:
                loaded_images = load_images_from_folder(folder)
                if loaded_images:
                    image_lists.append(loaded_images)
            except ValueError as error:
                print(f"[o1key GPT Image Batch] 文件夹{folder_index} 加载失败 - {error}")
        return image_lists

    def _create_pairs(
        self,
        image_lists: List[List[ImageInfo]],
        pairing_mode: str,
        manual_images: Optional[List[ImageInfo]] = None,
    ) -> List[Tuple[ImageInfo, ...]]:
        if pairing_mode == "不配对":
            if len(image_lists) > 1:
                raise ValueError("「不配对」模式只支持单个文件夹，请清空其他文件夹路径")

            if image_lists and manual_images:
                return [
                    (folder_image,) + tuple(manual_images)
                    for folder_image in image_lists[0]
                ]
            if image_lists:
                return [(folder_image,) for folder_image in image_lists[0]]
            return []

        if not image_lists:
            return []

        if len(image_lists) == 1:
            base_pairs = [(folder_image,) for folder_image in image_lists[0]]
        elif pairing_mode == "按相同图片命名":
            base_pairs = list(pair_images_by_name(*image_lists))
        else:
            base_pairs = list(pair_images_cartesian(*image_lists))

        if manual_images:
            manual_tuple = tuple(manual_images)
            base_pairs = [pair + manual_tuple for pair in base_pairs]

        return base_pairs

    def _collect_manual_images(self, kwargs) -> List[ImageInfo]:
        manual_images = []
        for image_index in range(1, 10):
            key = f"参考图{image_index}"
            if key not in kwargs or kwargs[key] is None:
                continue
            for tensor_index, image in enumerate(tensor_to_pil(kwargs[key])):
                manual_images.append(ImageInfo(
                    image=image,
                    filename=f"manual_{image_index}_{tensor_index}",
                    extension=".png",
                    source_path="",
                ))
        return manual_images

    @staticmethod
    def _pair_to_tensors(pair: Tuple[ImageInfo, ...]) -> List:
        return [pil_to_tensor([image_info.image]) for image_info in pair]

    @staticmethod
    def _resolve_size(分辨率: str) -> str:
        return _resolve_async_size(分辨率)

    @staticmethod
    def _resolve_model(模型: str) -> str:
        model_map = {
            "gpt-image-2-次卡": "gpt-image-2-c",
            "gpt-image-2-按量": "gpt-image-2",
        }
        return model_map.get(模型, 模型)

    @staticmethod
    def _resolve_quality(质量: str) -> str:
        quality_map = {"高": "high", "中": "medium", "低": "low", "自动": "auto"}
        return quality_map.get(质量, "auto")

    @staticmethod
    def _resolve_output_format(图片格式: str) -> str:
        output_format_map = {
            "JPEG": "jpeg",
            "PNG": "png",
            "WebP": "webp",
        }
        return output_format_map.get(图片格式, "png")

    @staticmethod
    def _ensure_output_folder(保存路径: str) -> str:
        output_folder = (保存路径 or "").strip()
        if not output_folder and _FOLDER_PATHS_AVAILABLE:
            output_folder = folder_paths.get_output_directory()
            print(f"[o1key GPT Image Batch] 未设置保存路径，使用 ComfyUI 默认 output 目录: {output_folder}")

        if not output_folder:
            raise ValueError("未设置保存路径，且当前环境无法获取 ComfyUI 默认 output 目录")

        os.makedirs(output_folder, exist_ok=True)
        test_path = os.path.join(output_folder, ".write_test")
        with open(test_path, "w", encoding="utf-8") as test_file:
            test_file.write("test")
        os.remove(test_path)
        return output_folder

    @staticmethod
    def _save_images(
        images: List[Image.Image],
        output_folder: str,
        image_format: str,
        base_filename: Optional[str] = None,
    ) -> List[str]:
        format_ext_map = {"JPEG": ".jpg", "PNG": ".png", "WebP": ".webp"}
        save_ext = format_ext_map.get(image_format, ".png")
        saved_files = []

        for image in images:
            if base_filename:
                counter = 0
                while True:
                    suffix = "" if counter == 0 else f"+{counter}"
                    filename = f"{base_filename}{suffix}{save_ext}"
                    output_path = os.path.join(output_folder, filename)
                    if not os.path.exists(output_path):
                        break
                    counter += 1
            else:
                output_path = generate_timestamp_filename(
                    output_folder=output_folder,
                    extension=save_ext,
                )

            if image_format == "JPEG":
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(output_path, quality=100)
            elif image_format == "WebP":
                image.save(output_path, lossless=True)
            else:
                save_image(image, output_path)

            saved_files.append(output_path)

        return saved_files

    def process_batch(
        self,
        prompt: str,
        模型: str,
        网络: str,
        分辨率: str,
        生图数量: int,
        质量: str,
        seed: int,
        图片格式: str,
        文件夹1: str,
        文件夹2: str,
        文件夹3: str,
        文件夹4: str,
        文件夹5: str,
        保存路径: str = "",
        图片配对模式: str = "不配对",
        遮罩=None,
        **kwargs,
    ):
        start_time = time.time()
        client = None

        try:
            if not prompt or not prompt.strip():
                raise ValueError("提示词不能为空")

            folders = [文件夹1, 文件夹2, 文件夹3, 文件夹4, 文件夹5]
            if not any(folder and folder.strip() for folder in folders):
                raise ValueError("请至少填写一个文件夹路径，该节点专为批量文件夹处理设计")

            image_lists = self._load_folders(folders)
            total_folder_images = sum(len(image_list) for image_list in image_lists)
            if total_folder_images == 0:
                raise ValueError("文件夹中未找到任何图片，请检查文件夹路径是否正确")

            manual_images = self._collect_manual_images(kwargs)
            pairs = self._create_pairs(
                image_lists=image_lists,
                pairing_mode=图片配对模式,
                manual_images=manual_images if manual_images else None,
            )
            if not pairs:
                raise ValueError("配对结果为空，请检查输入")

            batch_prompts = parse_batch_prompts(prompt)
            prompts_per_task = None
            if batch_prompts:
                expanded_pairs = []
                expanded_prompts = []
                for pair in pairs:
                    for batch_prompt in batch_prompts:
                        expanded_pairs.append(pair)
                        expanded_prompts.append(batch_prompt)
                pairs = expanded_pairs
                prompts_per_task = expanded_prompts

            total_tasks = len(pairs)
            if batch_prompts:
                print(
                    f"[o1key GPT Image Batch] 批量任务 | {图片配对模式} × "
                    f"{len(batch_prompts)} 个提示词 | 共 {total_tasks} 任务"
                )
            else:
                print(f"[o1key GPT Image Batch] 批量任务 | {图片配对模式} | 共 {total_tasks} 任务")

            output_folder = self._ensure_output_folder(保存路径)
            size = self._resolve_size(分辨率)
            model = self._resolve_model(模型)
            quality = self._resolve_quality(质量)
            output_format = self._resolve_output_format(图片格式)

            client = GptImageClient()
            client.base_url = get_base_url_by_route(网络)

            progress_bar = ProgressBar(total_tasks * 100) if _PROGRESS_BAR_AVAILABLE else None
            results = []
            all_saved_files = []

            for task_index, pair in enumerate(pairs, 1):
                if _INTERRUPT_AVAILABLE and processing_interrupted():
                    print("[o1key GPT Image Batch] 用户取消，已中断批量生成")
                    raise InterruptProcessingException()

                task_prompt = prompts_per_task[task_index - 1] if prompts_per_task else prompt
                base_filename = pair[0].filename if pair else None
                result = {
                    "task_index": task_index,
                    "success": False,
                    "generated_count": 0,
                    "saved_files": [],
                    "error": None,
                }

                try:
                    pil_images = client.generate_image_async_sync(
                        prompt=task_prompt,
                        model=model,
                        quality=quality,
                        size=size,
                        n=生图数量,
                        seed=seed,
                        image_tensor=self._pair_to_tensors(pair),
                        mask_tensor=遮罩,
                        output_format=output_format,
                        progress_callback=_make_node_progress_callback(progress_bar, task_index, total_tasks),
                    )
                    saved_files = self._save_images(
                        images=pil_images,
                        output_folder=output_folder,
                        image_format=图片格式,
                        base_filename=base_filename,
                    )
                    result["success"] = bool(pil_images)
                    result["generated_count"] = len(pil_images)
                    result["saved_files"] = saved_files
                    all_saved_files.extend(saved_files)
                    print(f"[o1key GPT Image Batch] [{task_index}/{total_tasks}] ✓ {base_filename or 'task'}")
                except InterruptProcessingException:
                    raise
                except Exception as error:
                    error_msg = str(error).split("\n")[0]
                    result["error"] = error_msg
                    print(f"[o1key GPT Image Batch] [{task_index}/{total_tasks}] ❌ {base_filename or 'task'} → {error_msg}")

                results.append(result)
                if progress_bar is not None:
                    progress_bar.update_absolute(task_index * 100, total_tasks * 100)

            success_count = sum(1 for result in results if result.get("success", False))
            total_generated = sum(result.get("generated_count", 0) for result in results)
            if success_count == 0:
                raise RuntimeError("所有批量任务均生成失败，无可用图像输出")

            output_images = []
            for file_path in all_saved_files[-10:]:
                try:
                    loaded_image = Image.open(file_path)
                    loaded_image.load()
                    output_images.append(loaded_image)
                except Exception as error:
                    print(f"[o1key GPT Image Batch] 无法加载输出图片 {file_path} - {error}")

            if not output_images:
                output_images = [Image.new("RGBA", (512, 512), (128, 128, 128, 255))]

            output_tensor = GptImageClient._pil_list_to_tensor(output_images)
            elapsed = time.time() - start_time
            print("=" * 60)
            print(
                f"[o1key GPT Image Batch] 完成！耗时 {elapsed:.1f}s | "
                f"成功 {success_count}/{total_tasks} | 生成 {total_generated} 张"
            )
            print(f"[o1key GPT Image Batch] 保存路径: {output_folder}")
            if all_saved_files:
                print(f"[o1key GPT Image Batch] 最新保存文件: {all_saved_files[-1]}")

            failed_results = [result for result in results if not result.get("success", False)]
            if failed_results:
                print(f"[o1key GPT Image Batch] 失败任务: {len(failed_results)} 个")
                for failed_result in failed_results[:3]:
                    print(
                        f"  - #{failed_result.get('task_index')}: "
                        f"{failed_result.get('error', '未知错误')}"
                    )

            return (output_tensor,)

        except ValueError as error:
            if str(error) == "未授权！":
                print("[o1key GPT Image Batch] 请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            raise ValueError(str(error)) from None
        except RuntimeError as error:
            raise RuntimeError(str(error)) from None
        finally:
            if client is not None:
                try:
                    balance_data = client.query_balance_sync()
                    balance_info = client.format_balance_info(balance_data)
                    print(f"[o1key GPT Image Batch] {balance_info}")
                except Exception:
                    pass
