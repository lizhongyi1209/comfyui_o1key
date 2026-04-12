"""
视频预览节点
接收 VIDEO 类型，在前端内嵌播放器预览
"""

import os
import io

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False


def _get_output_dir() -> str:
    if FOLDER_PATHS_AVAILABLE:
        return folder_paths.get_output_directory()
    plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(os.path.dirname(os.path.dirname(plugin_dir)), "output")


class VideoPreview:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "视频": ("VIDEO",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "comfyui_o1key/Utils"

    def preview(self, 视频) -> dict:
        # 用官方接口取文件路径
        source = 视频.get_stream_source()

        if isinstance(source, io.BytesIO):
            # BytesIO 情况：写到 output/video/ 临时文件
            output_dir = os.path.join(_get_output_dir(), "video")
            os.makedirs(output_dir, exist_ok=True)
            filename = "preview_tmp.mp4"
            tmp_path = os.path.join(output_dir, filename)
            source.seek(0)
            with open(tmp_path, "wb") as f:
                f.write(source.read())
            subfolder = "video"
        else:
            video_path = source
            output_dir = _get_output_dir()
            abs_video = os.path.abspath(video_path)
            abs_output = os.path.abspath(output_dir)

            if abs_video.startswith(abs_output):
                rel_path = os.path.relpath(abs_video, abs_output)
                subfolder = os.path.dirname(rel_path).replace("\\", "/")
                filename = os.path.basename(rel_path)
            else:
                # 文件在 output 目录外，复制一份
                target_dir = os.path.join(output_dir, "video")
                os.makedirs(target_dir, exist_ok=True)
                filename = os.path.basename(abs_video)
                target_path = os.path.join(target_dir, filename)
                if not os.path.exists(target_path):
                    import shutil
                    shutil.copy2(abs_video, target_path)
                subfolder = "video"

        return {
            "ui": {
                "videos": [{
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": "output",
                }],
            }
        }


NODE_CLASS_MAPPINGS = {
    "VideoPreview": VideoPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoPreview": "预览视频",
}
