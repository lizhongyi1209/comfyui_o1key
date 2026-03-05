"""
Comfyui_o1key - ComfyUI 自定义节点集合
通过 api.o1key.com 调用 AI 模型进行图像生成和文本生成

项目结构:
├── nodes/          # 节点实现
├── utils/          # 工具模块
├── clients/        # API 客户端
└── __init__.py     # 节点注册入口
"""

# 检查更新（仅在启动时检查一次）
try:
    from .utils.update_checker import check_for_updates, notify_update_available
    
    if check_for_updates():
        notify_update_available()
except Exception:
    # 静默失败，不影响插件加载
    pass

import ssl

from .nodes import NanoBananaPro, BatchNanoBananaPro, GoogleGemini, LoadFile, ImageStitchPro, SaveCleanImage, BatchCleanMetadata, SoraVideo, VideoPreview

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
_wrap_generate_for_error_display(BatchNanoBananaPro)

# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "NanoBananaPro": NanoBananaPro,
    "BatchNanoBananaPro": BatchNanoBananaPro,
    "GoogleGemini": GoogleGemini,
    "LoadFile": LoadFile,
    "ImageStitchPro": ImageStitchPro,
    "SaveCleanImage": SaveCleanImage,
    "BatchCleanMetadata": BatchCleanMetadata,
    "SoraVideo": SoraVideo,
    "VideoPreview": VideoPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaPro": "Nano Banana",
    "BatchNanoBananaPro": "批量 Nano Banana",
    "GoogleGemini": "Google Gemini",
    "LoadFile": "加载文件",
    "ImageStitchPro": "图像拼接 Pro",
    "SaveCleanImage": "保存图像（防AI识别）",
    "BatchCleanMetadata": "批量任务（防AI识别）",
    "SoraVideo": "Sora 视频生成",
    "VideoPreview": "视频预览",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
