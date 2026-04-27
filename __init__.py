"""
Comfyui_o1key - ComfyUI 自定义节点集合
通过 api.o1key.com 调用 AI 模型进行图像生成和文本生成

项目结构:
├── nodes/          # 节点实现
├── utils/          # 工具模块
├── clients/        # API 客户端
└── __init__.py     # 节点注册入口
"""


import ssl

from .nodes import NanoBananaPro, NanoBananaProAsync, BatchNanoBananaPro, GoogleGemini, LoadFile, ImageStitchPro, SaveCleanImage, BatchCleanMetadata, VideoPreview, GoogleVeo, FluxImageEdit, UniversalLLMChat, KlingVideo, KlingFirstLastFrame, KlingMotionControlTest, AspectRatioPreset, MultiResPreview, BatchImagesO1key, Seedance, SeedanceMultiModal, StreamPreview, DoubaoImage, O1keyGPTImage, KVideo
from .nodes import K3Video, K3VideoFirstLast, K3MotionControl, K3MotionVideoCheck

# 报错弹框友好文案（不修改原节点代码，仅在外层统一处理）
_MSG_TIMEOUT = "API 请求超时，请稍后重试或检查网络。"
_MSG_SSL_NETWORK = (
    "本地网络不太稳定!解决方案如下:\n"
    "1. 重启程序再试试看 (优先)\n"
    "2. 调整一下网络环境,如wifi或宽带等\n"
    "3. 切换VPN节点,或更换代理模式\n"
    "4. 关掉杀毒软件或防火墙\n"
    "5. 关掉浏览器VPN插件,避免冲突"
)

def _wrap_generate_for_error_display(cls, attr="generate"):
    original = getattr(cls, attr, None)
    if original is None:
        return
    def wrapped(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except TimeoutError as e:
            msg = (str(e) or "").strip()
            if not msg:
                msg = _MSG_TIMEOUT
            raise TimeoutError(msg) from None
        except (ssl.SSLError, OSError) as e:
            err_str = str(e)
            if "DECRYPTION_FAILED_OR_BAD_RECORD_MAC" in err_str or "decryption failed or bad record mac" in err_str.lower():
                raise RuntimeError(_MSG_SSL_NETWORK) from None
            raise
    setattr(cls, attr, wrapped)

_wrap_generate_for_error_display(NanoBananaPro)
_wrap_generate_for_error_display(NanoBananaProAsync)
_wrap_generate_for_error_display(BatchNanoBananaPro)

# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "NanoBananaPro": NanoBananaPro,
    "NanoBananaProAsync": NanoBananaProAsync,
    "BatchNanoBananaPro": BatchNanoBananaPro,
    "GoogleGemini": GoogleGemini,
    "LoadFile": LoadFile,
    "ImageStitchPro": ImageStitchPro,
    "SaveCleanImage": SaveCleanImage,
    "BatchCleanMetadata": BatchCleanMetadata,
    "VideoPreview": VideoPreview,
    "GoogleVeo": GoogleVeo,
    "FluxImageEdit": FluxImageEdit,
    "UniversalLLMChat": UniversalLLMChat,
    "KlingVideo": KlingVideo,
    "KlingFirstLastFrame": KlingFirstLastFrame,
    "KlingMotionControlTest": KlingMotionControlTest,
    "AspectRatioPreset": AspectRatioPreset,
    "MultiResPreview": MultiResPreview,
    "BatchImagesO1key": BatchImagesO1key,
    "Seedance": Seedance,
    "SeedanceMultiModal": SeedanceMultiModal,
    "StreamPreview": StreamPreview,
    "DoubaoImage": DoubaoImage,
    "O1keyGPTImage": O1keyGPTImage,
    "KVideo": KVideo,
    "K3Video": K3Video,
    "K3VideoFirstLast": K3VideoFirstLast,
    "K3MotionControl": K3MotionControl,
    "K3MotionVideoCheck": K3MotionVideoCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaPro": "Nano Banana",
    "NanoBananaProAsync": "Nano Banana（异步）",
    "BatchNanoBananaPro": "批量 Nano Banana",
    "GoogleGemini": "Google Gemini",
    "LoadFile": "加载文件",
    "ImageStitchPro": "图像拼接 Pro",
    "SaveCleanImage": "保存图像（防AI识别）",
    "BatchCleanMetadata": "批量任务（防AI识别）",
    "VideoPreview": "预览视频",
    "GoogleVeo": "Google Veo - ab",
    "FluxImageEdit": "Flux2 图像编辑",
    "UniversalLLMChat": "全能LLM对话助手",
    "KlingVideo": "文/图生视频 自研模型",
    "KlingFirstLastFrame": "首尾帧生视频 自研模型",
    "KlingMotionControlTest": "动作控制 自研模型",
    "AspectRatioPreset": "图片宽高比预设",
    "MultiResPreview": "预览图像（v2）",
    "BatchImagesO1key": "加载图像（批量）",
    "Seedance": "Seedance 视频生成",
    "SeedanceMultiModal": "Seedance 多模态参考生视频",
    "StreamPreview": "流式文本预览",
    "DoubaoImage": "豆包生图",
    "O1keyGPTImage": "o1key GPT Image",
    "KVideo": "K26 图生视频",
    "K3Video": "K3 图生视频 自研",
    "K3VideoFirstLast": "首尾帧 K3 自研",
    "K3MotionControl": "动作控制 K3 自研",
    "K3MotionVideoCheck": "视频时长检测 K3",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# 注册 /o1key/input_dir 接口，供前端文件上传按钮获取 input 目录绝对路径
try:
    from aiohttp import web
    from server import PromptServer
    import folder_paths

    @PromptServer.instance.routes.get("/o1key/input_dir")
    async def get_input_dir(request):
        import os
        path = os.path.abspath(folder_paths.get_input_directory())
        return web.json_response({"path": path})
except Exception:
    pass
