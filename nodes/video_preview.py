"""
通用视频预览节点
ComfyUI 自定义节点，接收视频文件路径并在前端展示预览
"""

import os

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".3gp"}


def _get_output_dir() -> str:
    if FOLDER_PATHS_AVAILABLE:
        return folder_paths.get_output_directory()
    plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(os.path.dirname(os.path.dirname(plugin_dir)), "output")


class VideoPreview:
    """
    通用视频预览节点

    功能：
    - 接收视频文件路径（STRING）
    - 在 ComfyUI 前端节点上内嵌 <video> 播放器进行预览
    - 支持 mp4, webm, mov, avi, mkv 等主流格式
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "预览视频": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "video"

    DESCRIPTION = (
        "通用视频预览节点。\n"
        "接收视频文件路径，在节点上显示视频播放器。\n"
        "支持 mp4, webm, mov, avi, mkv 等主流视频格式。"
    )

    def preview(self, **kwargs) -> dict:
        video_path = kwargs.get("预览视频", "")
        if not video_path or not video_path.strip():
            raise ValueError("视频路径为空")

        video_path = video_path.strip()

        if not os.path.isfile(video_path):
            raise ValueError(f"视频文件不存在: {video_path}")

        ext = os.path.splitext(video_path)[1].lower()
        if ext not in SUPPORTED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"不支持的视频格式 '{ext}'，"
                f"支持: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
            )

        output_dir = _get_output_dir()
        abs_video = os.path.abspath(video_path)
        abs_output = os.path.abspath(output_dir)

        if abs_video.startswith(abs_output):
            rel_path = os.path.relpath(abs_video, abs_output)
            subfolder = os.path.dirname(rel_path).replace("\\", "/")
            filename = os.path.basename(rel_path)
            file_type = "output"
        else:
            filename = os.path.basename(abs_video)
            subfolder = ""
            file_type = "output"

            # 如果文件不在 output 目录下，复制一份到 output/video/
            target_dir = os.path.join(output_dir, "video")
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, filename)

            if not os.path.exists(target_path) or abs_video != os.path.abspath(target_path):
                import shutil
                shutil.copy2(abs_video, target_path)

            subfolder = "video"

        file_size = os.path.getsize(abs_video)
        size_mb = file_size / (1024 * 1024)
        print(f"视频预览: {filename} ({size_mb:.1f}MB)")

        return {
            "ui": {
                "videos": [{
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": file_type,
                }],
            }
        }
