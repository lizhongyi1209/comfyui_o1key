"""
o1key Grok Image 节点
支持 Grok Image / Grok Image Pro 模型的文生图和图生图
"""

import time
from ..clients.grok_image_client import GrokImageClient
from ..utils.image_utils import parse_batch_prompts
from ..utils.config import NETWORK_ROUTE_OPTIONS

try:
    from comfy.model_management import processing_interrupted, InterruptProcessingException
    _INTERRUPT_AVAILABLE = True
except ImportError:
    _INTERRUPT_AVAILABLE = False
    processing_interrupted = lambda: False
    InterruptProcessingException = RuntimeError

_ASPECT_RATIOS = [
    "auto", "1:1", "16:9", "9:16", "4:3", "3:4",
    "3:2", "2:3", "2:1", "1:2", "19.5:9", "9:19.5", "20:9", "9:20",
]


class O1keyGrokImage:

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {}
        for i in range(1, 4):
            optional_inputs[f"参考图{i}"] = ("IMAGE", {
                "tooltip": f"Optional reference image {i}",
            })

        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "文本提示词，用 --- 独占一行分隔批量提示词",
                }),
            },
            "optional": {
                "模型": (["Grok Image", "Grok Image Pro"], {
                    "default": "Grok Image Pro",
                }),
                "宽高比": (_ASPECT_RATIOS, {
                    "default": "auto",
                }),
                "分辨率": (["1k", "2k"], {
                    "default": "1k",
                }),
                "生图数量": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "display": "number",
                }),
                "网络线路": (NETWORK_ROUTE_OPTIONS, {
                    "default": NETWORK_ROUTE_OPTIONS[0],
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "step": 1,
                    "display": "number",
                    "control_after_generate": True,
                }),
                **optional_inputs,
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "o1key/image"
    OUTPUT_NODE = False

    def generate(
        self,
        prompt: str,
        模型: str = "Grok Image Pro",
        宽高比: str = "auto",
        分辨率: str = "1k",
        生图数量: int = 1,
        网络线路: str = "全球加速",
        seed: int = 0,
        **kwargs,
    ):
        start_time = time.time()

        reference_tensors = []
        for i in range(1, 4):
            key = f"参考图{i}"
            if key in kwargs and kwargs[key] is not None:
                reference_tensors.append(kwargs[key])
        image_list = reference_tensors if reference_tensors else None

        try:
            client = GrokImageClient(route=网络线路)
        except ValueError as e:
            if str(e) == "未授权！":
                print("[o1key Grok Image] 请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            raise

        try:
            batch_prompts = parse_batch_prompts(prompt)
            all_pil_images = []

            if batch_prompts:
                total = len(batch_prompts)
                print(f"[o1key Grok Image] 批量模式 | {total} 条提示词 | 每条生成 {生图数量} 张")
                for idx, p in enumerate(batch_prompts, 1):
                    if _INTERRUPT_AVAILABLE and processing_interrupted():
                        print("[o1key Grok Image] 用户取消")
                        raise InterruptProcessingException()
                    try:
                        pil_images = client.run_sync(
                            prompt=p, model=模型, aspect_ratio=宽高比,
                            resolution=分辨率, n=生图数量, image_list=image_list,
                        )
                        all_pil_images.extend(pil_images)
                        snippet = p[:30] + ("..." if len(p) >= 30 else "")
                        print(f"[o1key Grok Image] [{idx}/{total}] done: {snippet}")
                    except InterruptProcessingException:
                        raise
                    except Exception as e:
                        error_msg = str(e).split('\n')[0]
                        snippet = p[:30] + ("..." if len(p) >= 30 else "")
                        print(f"[o1key Grok Image] [{idx}/{total}] fail: {snippet} → {error_msg}")
            else:
                if not prompt or not prompt.strip():
                    raise ValueError("提示词不能为空")
                pil_images = client.run_sync(
                    prompt=prompt, model=模型, aspect_ratio=宽高比,
                    resolution=分辨率, n=生图数量, image_list=image_list,
                )
                all_pil_images.extend(pil_images)

            if not all_pil_images:
                raise RuntimeError("所有提示词均生成失败，无可用图像输出")

            output_tensor = GrokImageClient._pil_list_to_tensor(all_pil_images)

            elapsed = time.time() - start_time
            print(
                f"[o1key Grok Image] 完成！耗时 {elapsed:.1f}s，"
                f"输出 {output_tensor.shape[0]} 张 "
                f"{output_tensor.shape[2]}x{output_tensor.shape[1]}"
            )
            return (output_tensor,)

        finally:
            self._print_balance(client)

    def _print_balance(self, client):
        try:
            balance_data = client.query_balance_sync()
            balance_info = client.format_balance_info(balance_data)
            print(f"[o1key Grok Image] {balance_info}")
        except Exception:
            pass
