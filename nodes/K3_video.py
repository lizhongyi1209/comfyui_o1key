"""
K3 图生视频 自研节点（方案A：经典 INPUT_TYPES 前端壳子）
复刻 Kling 3.0 Video 的前端参数，多镜头通过下拉选择控制分镜数量。
后端请求尚未实现。
"""

try:
    from comfy_api.latest import InputImpl
    import folder_paths
    _FOLDER_PATHS_OK = True
except Exception:
    _FOLDER_PATHS_OK = False


# ── 常量 ──────────────────────────────────────────────────────────────────────

_MULTI_SHOT_OPTIONS = [
    "disabled",
    "1 storyboard",
    "2 storyboards",
    "3 storyboards",
    "4 storyboards",
    "5 storyboards",
    "6 storyboards",
]

_MODELS        = ["kling-v3"]
_MODES         = ["std", "pro", "4k"]


# ── 节点 ──────────────────────────────────────────────────────────────────────

class K3Video:
    """K3 图生视频 自研（前端壳子，后端待实现）"""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            # 多镜头模式选择
            "多镜头": (_MULTI_SHOT_OPTIONS, {
                "default": "disabled",
                "tooltip": "disabled：单段模式；N storyboards：启用 N 段分镜。",
            }),
            # 单段模式字段（多镜头 disabled 时使用）
            "提示词":    ("STRING", {"multiline": True, "default": ""}),
            "负向提示词": ("STRING", {"multiline": True, "default": ""}),
            "时长":      ([5, 10, 15], {
                "default": 5,
                "tooltip": "单段模式时长（秒）；多镜头模式下由各分镜时长决定。",
            }),
            # 通用参数
            "生成音频": (["关闭", "打开"], {"default": "关闭"}),
            "模型":     (_MODELS, {"default": "kling-v3"}),
            "模式":     (_MODES,  {"default": "std"}),
            "seed": ("INT", {
                "default": 0, "min": 0, "max": 2147483647,
                "tooltip": "seed 仅控制节点是否重新运行，结果本身不可复现。",
            }),
        }

        # 分镜字段放最后，按 分镜N_提示词 / 分镜N_时长 交替排列
        for i in range(1, 7):
            required[f"分镜{i}_提示词"] = ("STRING", {
                "multiline": True, "default": "",
                "tooltip": f"第 {i} 段分镜提示词，最多 512 字符。",
            })
            required[f"分镜{i}_时长"] = ("INT", {
                "default": 4, "min": 1, "max": 15,
                "display": "slider",
                "tooltip": f"第 {i} 段分镜时长（秒）。",
            })

        optional = {
            "起始帧": ("IMAGE",),
            "尾帧":   ("IMAGE",),
        }

        return {"required": required, "optional": optional}

    RETURN_TYPES  = ("VIDEO",)
    RETURN_NAMES  = ("视频",)
    FUNCTION      = "generate"
    CATEGORY      = "comfyui_o1key/KVideo"

    async def generate(self, 多镜头, 提示词, 负向提示词, 时长,
                       生成音频, 模型, 模式, seed,
                       起始帧=None, 尾帧=None, **kwargs):
        raise NotImplementedError("K3 图生视频 自研：后端尚未实现。")


# ── 节点注册 ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "K3Video": K3Video,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "K3Video": "K3 图生视频 自研",
}
