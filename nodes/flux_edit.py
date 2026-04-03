"""
Flux2 图像编辑节点
通过 vip.o1key.com 调用 Flux2 + SeedVR2 远程服务进行图像编辑和超分辨率

功能：
- 接收主图和参考图
- 上传到远程服务器执行图像编辑
- 轮询等待 SeedVR2 超分辨率结果
- 返回最终放大后的图像
"""

import time
from io import BytesIO
from typing import Tuple

import torch
from PIL import Image

from ..utils.image_utils import tensor_to_pil, pil_to_tensor
from ..clients.flux_edit_client import FluxEditClient


class FluxImageEdit:
    """
    Flux2 图像编辑节点

    通过远程 API 将主图与参考图结合，按照提示词进行图像编辑，
    并经 SeedVR2 超分辨率放大后返回最终结果。
    """

    SIZES = ["2K", "4K"]

    def __init__(self):
        self.client = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "主图": ("IMAGE",),
                "参考图": ("IMAGE",),
                "提示词": ("STRING", {
                    "default": "Replace the woman's underwear in Figure 1 with the strapless bra in Figure 2",
                    "multiline": True,
                }),
                "分辨率": (cls.SIZES, {
                    "default": "4K",
                }),
                "轮询间隔": ("INT", {
                    "default": 15,
                    "min": 5,
                    "max": 60,
                    "step": 5,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "generate"
    CATEGORY = "image/edit"

    def _image_to_jpeg_bytes(self, image: Image.Image, quality: int = 92) -> bytes:
        """将 PIL Image 转为 JPEG 二进制"""
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def generate(
        self,
        主图: torch.Tensor,
        参考图: torch.Tensor,
        提示词: str,
        分辨率: str,
        轮询间隔: int,
        seed: int,
    ) -> Tuple[torch.Tensor]:
        """
        执行图像编辑

        Args:
            主图: 要编辑的原始图像 (ComfyUI tensor, [B, H, W, C])
            参考图: 参考/风格图像 (ComfyUI tensor, [B, H, W, C])
            提示词: 编辑指令
            分辨率: 超分辨率目标 ("2K" 或 "4K"，会自动映射为 2048/4096)
            轮询间隔: 轮询秒数
            seed: 随机种子

        Returns:
            输出图像 tensor (IMAGE,)
        """
        start_time = time.time()

        try:
            # 初始化客户端
            if self.client is None:
                self.client = FluxEditClient()

            # Tensor → PIL（取第一张）
            main_pils = tensor_to_pil(主图)
            ref_pils = tensor_to_pil(参考图)

            if not main_pils:
                raise ValueError("主图不能为空")
            if not ref_pils:
                raise ValueError("参考图不能为空")

            main_img = main_pils[0]
            ref_img = ref_pils[0]

            # PIL → JPEG bytes
            main_bytes = self._image_to_jpeg_bytes(main_img)
            ref_bytes = self._image_to_jpeg_bytes(ref_img)

            print(f"Flux Edit: 开始处理 | 主图 {main_img.size} | 参考图 {ref_img.size} | 分辨率 {分辨率} | seed {seed}")

            # 进度回调
            def progress_callback(status_str: str):
                print(f"Flux Edit: {status_str}")

            # 提交任务并等待结果
            result_bytes = self.client.submit_and_wait(
                image_bytes=main_bytes,
                mask_bytes=ref_bytes,
                prompt=提示词,
                size=分辨率,
                poll_interval=轮询间隔,
                progress_callback=progress_callback,
            )

            # 解码结果
            result_img = Image.open(BytesIO(result_bytes))
            if result_img.mode != "RGB":
                result_img = result_img.convert("RGB")

            print(f"Flux Edit: 结果图像尺寸 {result_img.size}")

            # 转为 tensor
            output_tensor = pil_to_tensor([result_img])

            # 打印耗时
            elapsed = time.time() - start_time
            if elapsed < 60:
                time_str = f"{elapsed:.1f}s"
            else:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                time_str = f"{minutes}m {seconds:.0f}s"
            print(f"Flux Edit: 完成！总耗时 {time_str}")

            return (output_tensor,)

        except ValueError as e:
            if str(e) == "未授权！":
                print("请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            print(f"Flux Edit: ❌ {e}")
            raise

        except Exception as e:
            error_msg = str(e)
            print(f"Flux Edit: ❌ {error_msg}")
            raise RuntimeError(error_msg) from None
