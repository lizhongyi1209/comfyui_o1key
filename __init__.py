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

from .nodes import NanoBananaPro, BatchNanoBananaPro, GoogleGemini, LoadFile

# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "NanoBananaPro": NanoBananaPro,
    "BatchNanoBananaPro": BatchNanoBananaPro,
    "GoogleGemini": GoogleGemini,
    "LoadFile": LoadFile
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaPro": "Nano Banana Pro",
    "BatchNanoBananaPro": "批量 Nano Banana Pro",
    "GoogleGemini": "Google Gemini",
    "LoadFile": "加载文件"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
