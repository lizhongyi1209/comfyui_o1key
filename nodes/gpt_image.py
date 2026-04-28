"""
o1key GPT Image 节点
支持 gpt-image-1 / gpt-image-1.5 模型的文生图、图生图、图像编辑（带蒙版）
"""

import time
import torch

from ..clients.gpt_image_client import GptImageClient
from ..utils.image_utils import parse_batch_prompts


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
      - seed     : 随机种子（0 表示不指定）
      - 质量     : 生成质量
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
            "gpt-image-2",
            "gpt-image-1.5",
            "gpt-image-2-特价",
            "gpt-image-1.5-特价",
        ], {
            "default": "gpt-image-2",
        })
        optional_inputs["分辨率"] = ([
            "auto（默认）",
            "1024x1024（正方形）",
            "1536x1024（景观）",
            "1024x1536（肖像）",
            "2048x2048（2K 平方）",
            "2048x1152（2K 横屏）",
            "3840x2160（4K 横屏）",
            "2160x3840（4K 竖屏）",
        ], {
            "default": "auto（默认）",
            "tooltip": "Image size (auto = API decides)",
        })
        optional_inputs["生图数量"] = ("INT", {
            "default": 1,
            "min": 1,
            "max": 8,
            "step": 1,
            "display": "number",
            "tooltip": "How many images to generate per prompt",
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
        optional_inputs["质量"] = (["高", "中", "低", "自动"], {
            "default": "自动",
            "tooltip": "Image quality: 高=high, 中=medium, 低=low, 自动=auto",
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
        模型: str = "gpt-image-2",
        分辨率: str = "auto",
        质量: str = "自动",
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

        if reference_tensors:
            图片 = torch.cat(reference_tensors, dim=0)
        else:
            图片 = None

        # ── 1. 参数校验 ───────────────────────────────────────────────────────
        if 遮罩 is not None and 图片 is None:
            raise ValueError("提供了遮罩但未提供图片，请同时提供图片和遮罩")

        # ── 2. 解析分辨率显示值 → API 参数值 ──────────────────────────────────
        size = 分辨率.split("（")[0].strip()

        # ── 2b. 解析质量显示值 → API 参数值 ───────────────────────────────────
        _quality_map = {"高": "high", "中": "medium", "低": "low", "自动": "auto"}
        quality = _quality_map.get(质量, "auto")

        # ── 3. 创建客户端 ─────────────────────────────────────────────────────
        try:
            client = GptImageClient()
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

            if batch_prompts:
                # 批量模式：逐条提示词调用
                total = len(batch_prompts)
                print(f"[o1key GPT Image] 批量模式 | {total} 条提示词 | 每条生成 {生图数量} 张")
                for idx, p in enumerate(batch_prompts, 1):
                    try:
                        pil_images = client.run_sync(
                            prompt=p,
                            model=模型,
                            quality=quality,
                            background="auto",
                            size=size,
                            n=生图数量,
                            seed=seed,
                            image_tensor=图片,
                            mask_tensor=遮罩,
                        )
                        all_pil_images.extend(pil_images)
                        snippet = p[:30] + ("..." if len(p) >= 30 else "")
                        print(f"[o1key GPT Image] [{idx}/{total}] ✓ {snippet}")
                    except Exception as e:
                        error_msg = str(e).split('\n')[0]
                        snippet = p[:30] + ("..." if len(p) >= 30 else "")
                        print(f"[o1key GPT Image] [{idx}/{total}] ❌ {snippet} → {error_msg}")
            else:
                # 单提示词模式
                if not prompt or not prompt.strip():
                    raise ValueError("提示词不能为空")
                try:
                    pil_images = client.run_sync(
                        prompt=prompt,
                        model=模型,
                        quality=quality,
                        background="auto",
                        size=size,
                        n=生图数量,
                        seed=seed,
                        image_tensor=图片,
                        mask_tensor=遮罩,
                    )
                    all_pil_images.extend(pil_images)
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
