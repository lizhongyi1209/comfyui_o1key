"""
Kling 3.0 Video Nodes
"""

import os
import tempfile

from ..clients.kling_client import KlingClient
from ..clients.gemini_client import GeminiAPIClient
from ..utils.image_utils import tensor_to_pil, encode_image_to_base64
from ..utils.config import NETWORK_ROUTE_OPTIONS, get_base_url_by_route

from comfy_api.latest import InputImpl


def _tensor_to_base64(tensor) -> str:
    """ComfyUI IMAGE tensor → base64 PNG 字符串"""
    pil_images = tensor_to_pil(tensor)
    return encode_image_to_base64(pil_images[0], format="PNG")


def _validate_prompt(prompt: str, *, required: bool = True) -> None:
    """校验单条提示词。

    Args:
        prompt:   提示词字符串。
        required: 为 True 时不允许为空（多镜头关闭或 shot_type 为 intelligence 时适用）。
    """
    if required and not prompt.strip():
        raise ValueError("提示词不能为空（非多镜头模式下必填）。")
    if len(prompt) > 2500:
        raise ValueError(
            f"提示词长度 ({len(prompt)}) 超过上限 2500 个字符，请缩短后重试。"
        )


def _validate_multi_prompt(multi_prompt_list: list, total_duration: int) -> None:
    """校验多镜头分镜列表。

    规则：
    - 分镜数量：1 ~ 6；
    - 每个分镜提示词不超过 512 个字符；
    - 每个分镜时长 ≥ 1 且 ≤ total_duration；
    - 所有分镜时长之和必须等于 total_duration。
    """
    count = len(multi_prompt_list)
    if count < 1 or count > 6:
        raise ValueError(
            f"多镜头分镜数量须在 1~6 之间，当前为 {count}。"
        )

    duration_sum = 0
    for entry in multi_prompt_list:
        idx      = entry["index"]
        p        = entry.get("prompt", "")
        dur      = entry.get("duration", 0)

        if len(p) > 512:
            raise ValueError(
                f"镜头 {idx} 提示词长度 ({len(p)}) 超过上限 512 个字符。"
            )
        if dur < 1:
            raise ValueError(
                f"镜头 {idx} 时长 ({dur}s) 不能小于 1 秒。"
            )
        if dur > total_duration:
            raise ValueError(
                f"镜头 {idx} 时长 ({dur}s) 超过任务总时长 ({total_duration}s)。"
            )
        duration_sum += dur

    if duration_sum != total_duration:
        raise ValueError(
            f"所有分镜时长之和 ({duration_sum}s) 必须等于任务总时长 ({total_duration}s)。"
        )


def _validate_image(tensor, label: str = "图片") -> None:
    """校验图片张量。

    规则：
    - 文件大小（PNG）不超过 10MB；
    - 宽、高均不小于 300px；
    - 宽高比介于 1:2.5 ~ 2.5:1 之间（即 ratio ∈ [0.4, 2.5]）。
    """
    import io

    pil_images = tensor_to_pil(tensor)
    img = pil_images[0]
    w, h = img.size

    # ── 最小尺寸 ──────────────────────────────────────────────────────
    if w < 300 or h < 300:
        raise ValueError(
            f"{label} 宽高不得小于 300px，当前为 {w}×{h}px。"
        )

    # ── 宽高比 ────────────────────────────────────────────────────────
    ratio = w / h
    if ratio < 1 / 2.5 or ratio > 2.5:
        raise ValueError(
            f"{label} 宽高比须在 1:2.5 ~ 2.5:1 之间，"
            f"当前为 {w}:{h}（比值 {ratio:.2f}）。"
        )

    # ── 文件大小 ──────────────────────────────────────────────────────
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    size_mb = buf.tell() / (1024 * 1024)
    if size_mb > 10:
        raise ValueError(
            f"{label} PNG 大小 ({size_mb:.1f}MB) 超过上限 10MB。"
        )


class KlingVideo:
    """Kling 视频生成节点（支持多镜头）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "反向提示词": ("STRING", {"multiline": True, "default": ""}),
                "网络线路": (NETWORK_ROUTE_OPTIONS, {"default": "全球加速"}),
                "模型版本": (["v3", "v2-6"], {"default": "v3"}),
                "时长": ([5, 10, 15],),
                "分辨率": (["1080p", "720p"],),
                "宽高比": (["智能", "16:9", "9:16", "1:1"], {"default": "智能"}),
                "生成音频": (["打开", "关闭"], {"default": "打开"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
            "optional": {
                "起始帧": ("IMAGE",),
                "镜头1_提示词": ("STRING", {"multiline": True, "default": ""}),
                "镜头1_时长": ("STRING", {"default": "5"}),
                "镜头2_提示词": ("STRING", {"multiline": True, "default": ""}),
                "镜头2_时长": ("STRING", {"default": "5"}),
                "镜头3_提示词": ("STRING", {"multiline": True, "default": ""}),
                "镜头3_时长": ("STRING", {"default": "5"}),
                "镜头4_提示词": ("STRING", {"multiline": True, "default": ""}),
                "镜头4_时长": ("STRING", {"default": "5"}),
                "镜头5_提示词": ("STRING", {"multiline": True, "default": ""}),
                "镜头5_时长": ("STRING", {"default": "5"}),
                "镜头6_提示词": ("STRING", {"multiline": True, "default": ""}),
                "镜头6_时长": ("STRING", {"default": "5"}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频",)
    FUNCTION = "generate"
    CATEGORY = "comfyui_o1key/Kling"

    async def generate(self, **kwargs):
        """生成视频（支持多镜头）"""
        prompt          = kwargs["提示词"]
        negative_prompt = kwargs["反向提示词"]
        model_ver       = kwargs.get("模型版本", "v3")
        duration        = kwargs["时长"]
        resolution      = kwargs["分辨率"]
        aspect_ratio    = kwargs["宽高比"]
        generate_audio  = kwargs["生成音频"]
        start_frame     = kwargs.get("起始帧", None)
        seed            = kwargs.get("seed", 0)  # noqa: F841 — 触发 ComfyUI 缓存刷新

        mode  = "pro" if resolution == "1080p" else "std"
        voice = "voice" if generate_audio == "打开" else "novoice"

        # ── v2-6 模型约束校验 ──────────────────────────────────────────
        if model_ver == "v2-6":
            if duration == 15:
                raise ValueError(
                    "v2-6 模型不支持 15s 时长，请选择 5s 或 10s。"
                )
            if mode == "std" and voice == "voice":
                raise ValueError(
                    "v2-6 模型的标准画质（720p）不支持生成音频，请关闭生成音频或切换至 1080p。"
                )

        # ── 多镜头检测 ────────────────────────────────────────────────
        multi_prompt_list = []
        for i in range(1, 7):
            sb_prompt = kwargs.get(f"镜头{i}_提示词", "").strip()
            if sb_prompt:
                raw_dur = kwargs.get(f"镜头{i}_时长", "5")
                try:
                    sb_duration = int(str(raw_dur).strip()) if str(raw_dur).strip() else 5
                except ValueError:
                    sb_duration = 5
                multi_prompt_list.append({
                    "index": i,
                    "prompt": sb_prompt,
                    "duration": sb_duration,
                })

        multi_shot_enabled = len(multi_prompt_list) > 0

        if multi_shot_enabled:
            total_duration = sum(e["duration"] for e in multi_prompt_list)
            if total_duration < 3 or total_duration > 15:
                raise ValueError(
                    f"多镜头总时长 ({total_duration}s) 必须在 3~15 秒之间。"
                )
            _validate_multi_prompt(multi_prompt_list, total_duration)
            duration = total_duration
        else:
            _validate_prompt(prompt, required=True)

        # ── 构建模型名 & 请求体 ───────────────────────────────────────
        import json, base64, copy
        model_name = f"kling-{model_ver}-{mode}-{duration}s-{voice}"

        body = {
            "model":    model_name,
            "mode":     mode,
            "duration": duration,
        }

        sound = "on" if generate_audio == "打开" else "off"

        if multi_shot_enabled or sound == "on":
            ms_payload = {}
            ms_payload["prompt"] = prompt

            if sound == "on":
                ms_payload["sound"] = "on"

            if multi_shot_enabled:
                ms_payload["multi_shot"] = True
                ms_payload["shot_type"] = "customize"
                ms_payload["multi_prompt"] = multi_prompt_list

            encoded = base64.b64encode(
                json.dumps(ms_payload, ensure_ascii=False).encode("utf-8")
            ).decode("utf-8")
            body["prompt"] = f"__MS__:{encoded}"
        else:
            body["prompt"] = prompt

        if negative_prompt.strip():
            body["negative_prompt"] = negative_prompt

        if start_frame is not None:
            _validate_image(start_frame, "起始帧")
            body["image"] = _tensor_to_base64(start_frame)
            endpoint_type = "image2video"
        else:
            if aspect_ratio and aspect_ratio != "智能":
                body["metadata"] = {"aspect_ratio": aspect_ratio}
            endpoint_type = "text2video"

        # ── 保存路径（临时文件，避免与下游保存节点重复落盘）──────────────────
        _, save_path = tempfile.mkstemp(suffix=".mp4", prefix="kling_")

        client = KlingClient()
        client.base_url = get_base_url_by_route(kwargs.get("网络线路", "全球加速"))

        # ── 进度条 ────────────────────────────────────────────────────
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(100)
        except Exception:
            pbar = None

        def on_stage(stage: str):
            if stage == "submitting":
                print("[视频生成] 提交中...")
                if pbar: pbar.update_absolute(0, 100)
            elif stage.startswith("submitted:"):
                print(f"[视频生成] 任务已提交 → {stage.split(':',1)[1]}")
                if pbar: pbar.update_absolute(5, 100)
            elif stage == "downloading":
                print("[视频生成] 下载视频...")
                if pbar: pbar.update_absolute(99, 100)
            elif stage == "done":
                print("[视频生成] 完成")
                if pbar: pbar.update_absolute(100, 100)

        def on_progress(pct: int):
            mapped = 5 + int(pct * 0.94)
            if pbar: pbar.update_absolute(mapped, 100)

        try:
            result_path = await client.generate_async(
                endpoint_type=endpoint_type,
                body=body,
                save_path=save_path,
                on_stage=on_stage,
                on_progress=on_progress,
            )
            return (InputImpl.VideoFromFile(result_path),)
        finally:
            # 查询余额
            try:
                _balance_client = GeminiAPIClient()
                balance_data = _balance_client.query_balance_sync()
                balance_info = _balance_client.format_balance_info(balance_data)
                print(f"自研视频模型: {balance_info}")
            except Exception:
                pass


class KlingFirstLastFrame:
    """Kling 首尾帧到视频节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "首帧": ("IMAGE",),
                "尾帧": ("IMAGE",),
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "网络线路": (NETWORK_ROUTE_OPTIONS, {"default": "全球加速"}),
                "模型": (["v3", "v2-6"], {"default": "v3"}),
                "分辨率": (["1080p", "720p"],),
                "时长": ([5, 10, 15],),
                "生成音频": (["打开", "关闭"], {"default": "打开"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频",)
    FUNCTION = "generate"
    CATEGORY = "comfyui_o1key/Kling"

    async def generate(self, **kwargs):
        first_frame  = kwargs["首帧"]
        end_frame    = kwargs["尾帧"]
        prompt       = kwargs["提示词"]
        duration     = kwargs["时长"]
        generate_audio = kwargs["生成音频"]
        model_base   = kwargs["模型"]
        model_base   = "kling-" + model_base          # v3/v2-6 → kling-v3/kling-v2-6（后端值还原）
        resolution   = kwargs["分辨率"]
        seed         = kwargs.get("seed", 0)  # noqa: F841 — 触发 ComfyUI 缓存刷新

        _validate_prompt(prompt, required=True)

        # 时长校验
        if duration not in (5, 10, 15):
            raise ValueError(f"时长仅支持 5、10、15 秒，当前值为 {duration}，请重新选择。")

        # 拼接模型名：kling-{ver}-{mode}-{dur}s-{voice}
        mode  = "pro" if resolution == "1080p" else "std"
        voice = "voice" if generate_audio == "打开" else "novoice"

        # ── v2-6 模型约束校验 ──────────────────────────────────────────
        model_ver = kwargs["模型"]   # "v3" or "v2-6"
        if model_ver == "v2-6":
            if duration == 15:
                raise ValueError(
                    "v2-6 模型不支持 15s 时长，请选择 5s 或 10s。"
                )
            if mode == "std" and voice == "voice":
                raise ValueError(
                    "v2-6 模型的标准画质（720p）不支持生成音频，请关闭生成音频或切换至 1080p。"
                )

        model_name = f"{model_base}-{mode}-{duration}s-{voice}"

        # 图片校验 & 转 base64
        _validate_image(first_frame, "首帧")
        _validate_image(end_frame,   "尾帧")
        image_b64      = _tensor_to_base64(first_frame)
        image_tail_b64 = _tensor_to_base64(end_frame)

        # ── 按规范编码 prompt 和 sound ──────────────────────────
        import json, base64
        sound = "on" if generate_audio == "打开" else "off"

        body = {
            "model":    model_name,
            "image":    image_b64,
            "mode":     mode,
            "duration": duration,
            "metadata": {
                "image_tail": image_tail_b64,
            },
        }

        if sound == "on":
            ms_payload = {
                "prompt": prompt,
                "sound": "on",
            }
            encoded = base64.b64encode(
                json.dumps(ms_payload, ensure_ascii=False).encode("utf-8")
            ).decode("utf-8")
            body["prompt"] = f"__MS__:{encoded}"
        else:
            body["prompt"] = prompt

        # 保存路径（临时文件，避免与下游保存节点重复落盘）
        _, save_path = tempfile.mkstemp(suffix=".mp4", prefix="kling_")

        client = KlingClient()
        client.base_url = get_base_url_by_route(kwargs.get("网络线路", "全球加速"))

        # 进度条：0~100 步
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(100)
        except Exception:
            pbar = None

        def on_stage(stage: str):
            if stage == "submitting":
                print("[视频生成] 提交中...")
                if pbar:
                    pbar.update_absolute(0, 100)
            elif stage.startswith("submitted:"):
                print(f"[视频生成] 任务已提交 → {stage.split(':',1)[1]}")
                if pbar:
                    pbar.update_absolute(5, 100)
            elif stage == "downloading":
                print("[视频生成] 下载视频...")
                if pbar:
                    pbar.update_absolute(99, 100)
            elif stage == "done":
                print("[视频生成] 完成")
                if pbar:
                    pbar.update_absolute(100, 100)

        def on_progress(pct: int):
            # pct 来自 API progress 字段，如 50 表示 50%
            # 生成阶段占 5~99 区间
            mapped = 5 + int(pct * 0.94)
            if pbar:
                pbar.update_absolute(mapped, 100)

        try:
            result_path = await client.generate_async(
                endpoint_type="image2video",
                body=body,
                save_path=save_path,
                on_stage=on_stage,
                on_progress=on_progress,
            )
            return (InputImpl.VideoFromFile(result_path),)
        finally:
            # 查询余额
            try:
                _balance_client = GeminiAPIClient()
                balance_data = _balance_client.query_balance_sync()
                balance_info = _balance_client.format_balance_info(balance_data)
                print(f"自研视频模型: {balance_info}")
            except Exception:
                pass


class KlingMotionControlTest:
    """Kling 动作控制（测试）节点 —— reference_video 接受 VIDEO 类型输入"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词":     ("STRING", {"multiline": True, "default": ""}),
                "参考图片":   ("IMAGE",),
                "参考视频":   ("VIDEO",),
                "网络线路":   (NETWORK_ROUTE_OPTIONS, {"default": "全球加速"}),
            },
            "optional": {
                "模型":       (["v3", "v2-6"], {"default": "v3"}),
                "分辨率":     (["1080p", "720p"],),
                "时长":       ([5, 10, 15], {"default": 5}),
                "人物朝向":   (["video", "image"],),
                "保留原声":   (["打开", "关闭"], {"default": "打开"}),
                "seed":       ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频",)
    FUNCTION = "generate"
    CATEGORY = "comfyui_o1key/Kling"

    async def generate(self, **kwargs):
        """动作控制：VIDEO 类型参考视频 + 图片人物动作迁移（走 new API 三段式）"""
        import base64

        prompt                = kwargs["提示词"]
        reference_image       = kwargs["参考图片"]
        reference_video       = kwargs["参考视频"]
        keep_original_sound   = kwargs.get("保留原声", "打开")
        character_orientation = kwargs.get("人物朝向", "video")
        mode                  = kwargs.get("分辨率", "1080p")
        duration              = kwargs.get("时长", 5)
        mode_api              = "pro" if mode == "1080p" else "std"  # 映射为 API 参数值
        model                 = kwargs.get("模型", "v3")
        model_name            = f"kling-{model}-motion-{mode_api}-{duration}s"
        seed                  = kwargs.get("seed", 0)  # noqa: F841 — 触发 ComfyUI 缓存刷新

        # ── 校验提示词 ────────────────────────────────────────────────
        _validate_prompt(prompt, required=True)

        # ── 校验参考图片 ──────────────────────────────────────────────
        _validate_image(reference_image, "参考图片")
        image_b64 = _tensor_to_base64(reference_image)

        # ── 从 VIDEO 对象获取本地文件路径并读取 ───────────────────────
        video_path = None
        if hasattr(reference_video, "source_path"):
            video_path = reference_video.source_path
        elif hasattr(reference_video, "path"):
            video_path = reference_video.path
        elif isinstance(reference_video, str):
            video_path = reference_video.strip()

        if not video_path or not os.path.isfile(video_path):
            raise ValueError(
                f"无法获取参考视频文件路径，请确保连接的是本地视频文件。"
                f"（当前路径：{video_path}）"
            )

        # ── 校验视频时长约束 ──────────────────────────────────────────
        # 人物朝向="video" → 3~30 秒；人物朝向="image" → 3~10 秒
        try:
            import subprocess, json as _json
            ffprobe_cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                video_path,
            ]
            result_proc = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=30)
            if result_proc.returncode == 0:
                info = _json.loads(result_proc.stdout)
                duration_sec = float(info.get("format", {}).get("duration", 0))
                if character_orientation == "video":
                    if not (3 <= duration_sec <= 30):
                        raise ValueError(
                            f"当人物朝向为 'video' 时，"
                            f"参考视频时长须在 3~30 秒之间，当前为 {duration_sec:.1f}s。"
                        )
                else:  # "image"
                    if not (3 <= duration_sec <= 10):
                        raise ValueError(
                            f"当人物朝向为 'image' 时，"
                            f"参考视频时长须在 3~10 秒之间，当前为 {duration_sec:.1f}s。"
                        )
        except FileNotFoundError:
            print("[动作控制] 警告：ffprobe 未找到，跳过视频时长校验。")
        except ValueError:
            raise
        except Exception as e:
            print(f"[动作控制] 时长校验异常（已跳过）：{e}")

        # ── 视频转 base64 ─────────────────────────────────────────────
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        # ── 构建请求体（new API 格式）─────────────────────────────────
        body = {
            "model":                 model_name,
            "prompt":                prompt,
            "image_url":             image_b64,
            "video_url":             video_b64,
            "character_orientation": character_orientation,
            "mode":                  mode_api,
            "keep_original_sound":   "yes" if keep_original_sound == "打开" else "no",
        }

        # ── 保存路径（临时文件，避免与下游保存节点重复落盘）──────────────────
        _, save_path = tempfile.mkstemp(suffix=".mp4", prefix="kling_motion_")

        client = KlingClient()
        client.base_url = get_base_url_by_route(kwargs.get("网络线路", "全球加速"))

        # ── 进度条 ────────────────────────────────────────────────────
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(100)
        except Exception:
            pbar = None

        def on_stage(stage: str):
            if stage == "submitting":
                print("[动作控制] 提交中...")
                if pbar: pbar.update_absolute(0, 100)
            elif stage.startswith("submitted:"):
                print(f"[动作控制] 任务已提交 → {stage.split(':',1)[1]}")
                if pbar: pbar.update_absolute(5, 100)
            elif stage == "downloading":
                print("[动作控制] 下载视频...")
                if pbar: pbar.update_absolute(99, 100)
            elif stage == "done":
                print("[动作控制] 完成")
                if pbar: pbar.update_absolute(100, 100)

        def on_progress(pct: int):
            mapped = 5 + int(pct * 0.94)
            if pbar: pbar.update_absolute(mapped, 100)

        try:
            result_path = await client.motion_control_async(
                body=body,
                save_path=save_path,
                on_stage=on_stage,
                on_progress=on_progress,
            )
            return (InputImpl.VideoFromFile(result_path),)
        finally:
            # 查询余额
            try:
                _balance_client = GeminiAPIClient()
                balance_data = _balance_client.query_balance_sync()
                balance_info = _balance_client.format_balance_info(balance_data)
                print(f"自研视频模型: {balance_info}")
            except Exception:
                pass


class AspectRatioPreset:
    """图片宽高比预设节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "宽高比": (["智能", "16:9", "9:16", "4:3", "3:4", "1:1"], {"default": "智能"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "resize"
    CATEGORY = "comfyui_o1key/Utils"

    def resize(self, 图像, 宽高比):
        import torch
        from PIL import Image
        import numpy as np

        pil_images = tensor_to_pil(图像)
        img = pil_images[0]
        w, h = img.size
        img_ratio = w / h

        # 确定原图所属的宽高比家族
        ratios = {"16:9": 16/9, "9:16": 9/16, "4:3": 4/3, "3:4": 3/4, "1:1": 1.0}
        closest_ratio = min(ratios.keys(), key=lambda k: abs(ratios[k] - img_ratio))

        # 智能模式：使用最接近的比例
        if 宽高比 == "智能":
            宽高比 = closest_ratio

        # 解析目标比例
        target_w, target_h = map(int, 宽高比.split(":"))
        target_ratio = target_w / target_h

        # 确定分辨率级别（1K/2K）
        max_dim = max(w, h)
        if max_dim <= 1080:
            base = 1080
        elif max_dim <= 2160:
            base = 2160
        else:
            base = 2160

        # 计算目标尺寸
        if target_ratio >= 1:
            target_width = base
            target_height = int(base / target_ratio)
        else:
            target_height = base
            target_width = int(base * target_ratio)

        # 判断是否同家族（横向家族：16:9, 4:3；纵向家族：9:16, 3:4；正方形：1:1）
        horizontal_family = ["16:9", "4:3"]
        vertical_family = ["9:16", "3:4"]

        same_family = False
        if closest_ratio in horizontal_family and 宽高比 in horizontal_family:
            same_family = True
        elif closest_ratio in vertical_family and 宽高比 in vertical_family:
            same_family = True
        elif closest_ratio == "1:1" and 宽高比 == "1:1":
            same_family = True

        # 同家族：直接缩放或裁剪（无白底）
        if same_family:
            if img_ratio > target_ratio:
                # 图像更宽，以高度为准缩放后裁剪
                scale = target_height / h
                scaled_w = int(w * scale)
                scaled_h = target_height
                scaled = img.resize((scaled_w, scaled_h), Image.LANCZOS)
                left = (scaled_w - target_width) // 2
                result = scaled.crop((left, 0, left + target_width, target_height))
            else:
                # 图像更高，以宽度为准缩放后裁剪
                scale = target_width / w
                scaled_w = target_width
                scaled_h = int(h * scale)
                scaled = img.resize((scaled_w, scaled_h), Image.LANCZOS)
                top = (scaled_h - target_height) // 2
                result = scaled.crop((0, top, target_width, top + target_height))

        # 不同家族：保持宽高比 + 白底填充
        else:
            if img_ratio > target_ratio:
                scaled_w = target_width
                scaled_h = int(target_width / img_ratio)
            else:
                scaled_h = target_height
                scaled_w = int(target_height * img_ratio)

            scaled = img.resize((scaled_w, scaled_h), Image.LANCZOS)
            canvas = Image.new("RGB", (target_width, target_height), (255, 255, 255))
            paste_x = (target_width - scaled_w) // 2
            paste_y = (target_height - scaled_h) // 2
            canvas.paste(scaled, (paste_x, paste_y))
            result = canvas

        # 转回 tensor
        arr = np.array(result).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)

        return (tensor,)


NODE_CLASS_MAPPINGS = {
    "KlingVideo": KlingVideo,
    "KlingFirstLastFrame": KlingFirstLastFrame,
    "KlingMotionControlTest": KlingMotionControlTest,
    "AspectRatioPreset": AspectRatioPreset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KlingVideo": "文/图生视频 自研模型",
    "KlingFirstLastFrame": "首尾帧生视频 自研模型",
    "KlingMotionControlTest": "动作控制 自研模型",
    "AspectRatioPreset": "图片宽高比预设",
}
