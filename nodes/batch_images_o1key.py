"""
批量图像（o1key）节点
复刻 ComfyUI 原生「批量图像」节点的动态输入行为：

- 默认显示 2 个图像输入端口（图1, 图2）
- 当最后一个端口连上图像后，自动追加新端口
- 断开连线后，多余的端口自动消失，最少保留 2 个

与原生节点的区别：
  原生节点会把所有图像强制 resize 到第一张的分辨率再合并为单一 tensor。
  本节点保留每张图的原始分辨率，以 list[Tensor] 形式输出（is_output_list）。
  下游节点（如「多分辨率图像预览」）需开启 INPUT_IS_LIST 才能正确接收。

实现方式：使用 V3 API 的 io.Autogrow.TemplateNames，
框架原生支持动态 slot 增减，无需编写任何 JS 扩展。
"""

import torch
from comfy_api.latest import io

# 预生成 50 个端口名：图1, 图2, ..., 图50
_SLOT_NAMES = [f"图{i}" for i in range(1, 51)]


class BatchImagesO1key(io.ComfyNode):
    """
    批量图像（o1key）

    - 动态输入端口（默认 2 个，最多 50 个），端口名为 图1、图2、图3...
    - 连接最后一个端口时自动增加新端口
    - 断开后自动减少，保持界面整洁
    - 保留每张图的原始分辨率，不做任何 resize / 裁剪
    - 输出为图像列表，可直接接入「多分辨率图像预览」节点
    """

    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplateNames(
            input=io.Image.Input("image"),
            names=_SLOT_NAMES,
            min=2,
        )
        return io.Schema(
            node_id="BatchImagesO1key",
            display_name="加载图像（批量）",
            category="image",
            description=(
                "将多个独立图像收集为图像列表输出，保留每张图的原始分辨率。\n"
                "• 默认显示 2 个输入端口（图1、图2），连接最后一个后自动追加新端口\n"
                "• 断开连线后端口自动减少，最少保留 2 个\n"
                "• 不做任何 resize / 裁剪，原图尺寸原样输出\n"
                "• 输出为图像列表，可直接接入「多分辨率图像预览」节点"
            ),
            search_aliases=["批量图像", "batch images", "合并图像", "图像合并", "stack images"],
            inputs=[
                io.Autogrow.Input("images", template=autogrow_template)
            ],
            outputs=[
                io.Image.Output(display_name="图像", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, images: io.Autogrow.Type) -> io.NodeOutput:
        # images 是 dict，key 为 "图1", "图2", ... ；未连接的 slot 值为 None
        tensors = [v for v in images.values() if v is not None]

        if not tensors:
            raise ValueError("批量图像（o1key）：请至少连接一张图像")

        for i, t in enumerate(tensors):
            h, w = t.shape[1], t.shape[2]
            print(f"批量图像（o1key）：图{i + 1} → {w}×{h}，shape={list(t.shape)}")

        print(f"批量图像（o1key）：共收集 {len(tensors)} 张，原始分辨率原样输出")

        # 以 list[Tensor] 形式返回，每张图保持自身分辨率
        return io.NodeOutput(tensors)


