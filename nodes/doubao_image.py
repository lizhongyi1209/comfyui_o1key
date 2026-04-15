"""
豆包生图节点
后端通过 new-api 兼容层调用豆包官方 API
"""

import asyncio
import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from typing import List, Optional

from ..clients.doubao_image_client import DoubaoImageClient
from ..utils.image_utils import tensor_to_pil


# ── 模型列表 ──────────────────────────────────────────────────────────────────
_MODELS = [
    "doubao-seedream-5-0-260128",
    "doubao-seedream-4-5-251128",
]

# ── 宽高比列表 ─────────────────────────────────────────────────────────────────
_ASPECT_RATIOS = ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9"]

# ── 分辨率档位（每个模型支持的档位不同）──────────────────────────────────────
# 5.0：2K / 3K
# 4.5：2K / 4K
_RESOLUTIONS = ["2K", "3K", "4K"]

# ── 像素对照表 ─────────────────────────────────────────────────────────────────
# 结构：{ 模型版本key: { 分辨率: { 宽高比: (宽, 高) } } }
_SIZE_TABLE = {
    "5-0": {
        "2K": {
            "1:1":  (2048, 2048),
            "4:3":  (2304, 1728),
            "3:4":  (1728, 2304),
            "16:9": (2848, 1600),
            "9:16": (1600, 2848),
            "3:2":  (2496, 1664),
            "2:3":  (1664, 2496),
            "21:9": (3136, 1344),
        },
        "3K": {
            "1:1":  (3072, 3072),
            "4:3":  (3456, 2592),
            "3:4":  (2592, 3456),
            "16:9": (4096, 2304),
            "9:16": (2304, 4096),
            "3:2":  (3744, 2496),
            "2:3":  (2496, 3744),
            "21:9": (4704, 2016),
        },
    },
    "4-5": {
        "2K": {
            "1:1":  (2048, 2048),
            "4:3":  (2304, 1728),
            "3:4":  (1728, 2304),
            "16:9": (2848, 1600),
            "9:16": (1600, 2848),
            "3:2":  (2496, 1664),
            "2:3":  (1664, 2496),
            "21:9": (3136, 1344),
        },
        "4K": {
            "1:1":  (4096, 4096),
            "4:3":  (4704, 3520),
            "3:4":  (3520, 4704),
            "16:9": (5504, 3040),
            "9:16": (3040, 5504),
            "3:2":  (4992, 3328),
            "2:3":  (3328, 4992),
            "21:9": (6240, 2656),
        },
    },
}

# 每个模型版本支持的分辨率档位
_MODEL_RESOLUTIONS = {
    "5-0": ["2K", "3K"],
    "4-5": ["2K", "4K"],
}

# 并发请求超时（秒）
_CONCURRENT_TIMEOUT = 330


def _model_key(model: str) -> str:
    """从模型 ID 中提取版本 key（'5-0' 或 '4-5'）。"""
    for key in _SIZE_TABLE:
        if key in model:
            return key
    raise ValueError(f"无法识别模型版本：{model}，支持的模型：{_MODELS}")


def _pil_list_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """
    PIL Image 列表 → ComfyUI IMAGE tensor [B, H, W, C]，值域 [0, 1]。
    多张尺寸不同时，以最大尺寸为准，较小图像丢弃。
    """
    if not images:
        placeholder = Image.new("RGB", (512, 512), color=(128, 128, 128))
        images = [placeholder]

    base_size = max(images, key=lambda img: img.size[0] * img.size[1]).size
    matched = [img for img in images if img.size == base_size]
    skipped = len(images) - len(matched)
    if skipped:
        print(f"[豆包生图] 丢弃 {skipped} 张非最大尺寸图像，仅输出 {base_size[0]}×{base_size[1]} 的 {len(matched)} 张")

    tensors = []
    for img in matched:
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr))

    return torch.stack(tensors, dim=0)  # [B, H, W, C]


class DoubaoImage:
    """豆包生图 —— 通过宽高比 + 分辨率档位选择尺寸，后端自动换算真实像素"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": (
                    _MODELS,
                    {"default": _MODELS[0]},
                ),
                "提示词": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "用于创建或编辑图像的文本提示",
                    },
                ),
                "宽高比": (
                    _ASPECT_RATIOS,
                    {
                        "default": "1:1",
                        "tooltip": "图像宽高比。所有分辨率档位均支持这些比例",
                    },
                ),
                "分辨率": (
                    _RESOLUTIONS,
                    {
                        "default": "2K",
                        "tooltip": (
                            "图像分辨率档位。\n"
                            "• Seedream 5.0：支持 2K / 3K\n"
                            "• Seedream 4.5：支持 2K / 4K\n"
                            "（3K 与 4.5 或 4K 与 5.0 搭配时将报错）"
                        ),
                    },
                ),
                "生图数量": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "生成图像的数量。2-10 张时自动并发请求，加快出图速度",
                    },
                ),
                "种子": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "step": 1,
                        "control_after_generate": True,
                        "tooltip": "用于生成的随机种子",
                    },
                ),
                "部分失败时停止": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "启用时：任意一张失败即抛出错误并中止。\n"
                            "禁用时：返回已成功生成的图像，忽略失败项"
                        ),
                    },
                ),
            },
            "optional": {
                "图像": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "用于图生图的输入图像。"
                            "单参考或多参考生成时，可输入1-10张图像列表"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "generate"
    CATEGORY = "comfyui_o1key/豆包"

    # ── 并发核心：在新 event loop 里 gather N 个 _generate_async ─────────────

    async def _run_concurrent(
        self,
        client: DoubaoImageClient,
        生图数量: int,
        model: str,
        prompt: str,
        size: str,
        seed: int,
        image_tensor,
        pbar,
    ) -> List[dict]:
        """
        并发发起 生图数量 个独立请求，每完成一个推进一格进度条。
        返回结果列表：[{"index": int, "images": [...], "error": str|None}]
        """
        # 固定参数（顺序生成功能暂时隐藏）
        seq = "disabled"
        max_img = 1

        async def _one(idx: int) -> dict:
            try:
                imgs = await client._generate_async(
                    model=model,
                    prompt=prompt,
                    size=size,
                    seed=seed,
                    sequential_image_generation=seq,
                    max_images=max_img,
                    image_tensor=image_tensor,
                )
                return {"index": idx, "images": imgs, "error": None}
            except Exception as e:
                return {"index": idx, "images": [], "error": str(e)}

        # 用 as_completed 方式逐个推进进度条
        tasks = [asyncio.create_task(_one(i)) for i in range(生图数量)]
        results = [None] * 生图数量
        completed = 0

        for coro in asyncio.as_completed(tasks):
            res = await coro
            results[res["index"]] = res
            completed += 1
            status = "✓" if res["error"] is None else f"✗ {res['error']}"
            print(f"[豆包生图] [{completed}/{生图数量}] 第 {res['index'] + 1} 张 → {status}")
            if pbar is not None:
                pbar.update(1)

        return results

    # ── 节点主入口 ────────────────────────────────────────────────────────────

    def generate(
        self,
        模型: str,
        提示词: str,
        宽高比: str,
        分辨率: str,
        生图数量: int,
        种子: int,
        部分失败时停止: bool,
        图像=None,
    ):
        start_time = time.time()

        # 顺序图像生成功能暂时隐藏，固定使用默认值
        顺序图像生成 = "disabled"
        最大图片数 = 1

        # ── 1. 校验提示词 ─────────────────────────────────────────────────────
        if not 提示词.strip():
            raise ValueError("提示词不能为空，请输入图像描述后重试。")

        # ── 2. 解析模型版本并校验分辨率兼容性 ────────────────────────────────
        try:
            mkey = _model_key(模型)
        except ValueError as e:
            raise ValueError(str(e)) from None

        supported = _MODEL_RESOLUTIONS[mkey]
        if 分辨率 not in supported:
            raise ValueError(
                f"模型 {模型} 不支持 {分辨率} 分辨率。\n"
                f"该模型支持：{' / '.join(supported)}"
            )

        # ── 3. 查表换算真实像素 ───────────────────────────────────────────────
        w, h = _SIZE_TABLE[mkey][分辨率][宽高比]
        size_str = f"{w}x{h}"

        # ── 4. 打印概要 ───────────────────────────────────────────────────────
        mode_str = "图生图" if 图像 is not None else "文生图"
        print(
            f"[豆包生图] {mode_str} | 模型={模型} | {分辨率} {宽高比} → {size_str}"
            f" | 数量={生图数量} | 种子={种子}"
        )

        # ── 5. 初始化客户端 ───────────────────────────────────────────────────
        try:
            client = DoubaoImageClient()
        except ValueError as e:
            raise ValueError(str(e)) from None

        # ── 6. 进度条（按张数计）──────────────────────────────────────────────
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(生图数量)
        except Exception:
            pbar = None

        # ── 7. 单张 / 多张分支 ────────────────────────────────────────────────
        if 生图数量 == 1:
            # 单张：走原有同步路径
            try:
                pil_images: List[Image.Image] = client.generate_sync(
                    model=模型,
                    prompt=提示词,
                    size=size_str,
                    seed=种子,
                    sequential_image_generation=顺序图像生成,
                    max_images=最大图片数,
                    image_tensor=图像,
                )
            except RuntimeError as e:
                raise RuntimeError(str(e)) from None
            except Exception as e:
                raise RuntimeError(f"豆包生图请求失败: {e}") from None

            if pbar is not None:
                pbar.update(1)

        else:
            # 多张：并发请求
            def _run_in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self._run_concurrent(
                            client=client,
                            生图数量=生图数量,
                            model=模型,
                            prompt=提示词,
                            size=size_str,
                            seed=种子,
                            image_tensor=图像,
                            pbar=pbar,
                        )
                    )
                finally:
                    loop.close()

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_in_thread)
                try:
                    results = future.result(timeout=_CONCURRENT_TIMEOUT)
                except TimeoutError:
                    raise RuntimeError(
                        f"并发生图超时（>{_CONCURRENT_TIMEOUT}s），请检查网络或减少生图数量"
                    )

            # 统计成功 / 失败
            success_results = [r for r in results if r and r["error"] is None]
            failed_results  = [r for r in results if r and r["error"] is not None]

            if failed_results:
                fail_info = "；".join(
                    f"第{r['index']+1}张: {r['error']}" for r in failed_results
                )
                if 部分失败时停止:
                    raise RuntimeError(
                        f"{len(failed_results)}/{生图数量} 张生成失败：{fail_info}\n"
                        "（可将【部分失败时停止】设为 False 以返回已成功的图像）"
                    )
                else:
                    print(f"[豆包生图] 警告：{len(failed_results)}/{生图数量} 张失败，已忽略：{fail_info}")

            if not success_results:
                raise RuntimeError("所有图像均生成失败，请检查网络或 API 配置。")

            # 按原始 index 排序，展平为 PIL 列表
            success_results.sort(key=lambda r: r["index"])
            pil_images = []
            for r in success_results:
                pil_images.extend(r["images"])

        # ── 8. PIL → tensor ───────────────────────────────────────────────────
        output_tensor = _pil_list_to_tensor(pil_images)

        # ── 9. 完成日志 ───────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        print(
            f"[豆包生图] 完成！耗时 {elapsed:.1f}s，"
            f"输出 {output_tensor.shape[0]} 张 "
            f"{output_tensor.shape[2]}×{output_tensor.shape[1]}"
        )

        return (output_tensor,)


# ── 节点注册 ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "DoubaoImage": DoubaoImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoImage": "豆包生图",
}
