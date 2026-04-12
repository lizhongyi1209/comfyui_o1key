"""
流式文本预览节点
接收文本输入，支持 markdown 渲染，通过 ComfyUI 事件系统实时推送内容
"""


class StreamPreview:
    """
    流式 Markdown 预览节点
    接收任意文本，在节点面板中实时渲染为 Markdown 格式
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文本": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "preview"
    CATEGORY = "text/preview"
    OUTPUT_NODE = True

    def preview(self, 文本: str):
        return {"ui": {"text": [文本]}, "result": (文本,)}
