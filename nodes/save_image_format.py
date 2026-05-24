"""保存图像节点 - 支持 PNG/JPEG/WebP 格式输出"""

import os
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args


class SaveImageFormat:
    """保存图像，支持 PNG / JPEG / WebP 三种格式"""

    FORMATS = ["PNG", "JPEG", "WebP"]

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "format": (cls.FORMATS, {"default": "PNG"}),
            },
            "optional": {},
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"
    DESCRIPTION = "保存图像，支持 PNG / JPEG / WebP 格式输出。"

    _EXT_MAP = {"PNG": ".png", "JPEG": ".jpg", "WebP": ".webp"}

    def save_images(self, images, filename_prefix="ComfyUI", format="PNG",
                    prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir,
                images[0].shape[1], images[0].shape[0]
            )

        ext = self._EXT_MAP.get(format, ".png")
        results = []

        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_{ext}"

            filepath = os.path.join(full_output_folder, file)

            if format == "PNG":
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                img.save(filepath, pnginfo=metadata,
                         compress_level=self.compress_level)
            elif format == "JPEG":
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save(filepath, quality=100, optimize=True)
            elif format == "WebP":
                img.save(filepath, lossless=True)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
            })
            counter += 1

        return {"ui": {"images": results}}
