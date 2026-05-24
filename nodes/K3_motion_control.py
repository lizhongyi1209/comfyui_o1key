"""
K3 动作控制 自研节点
用参考视频驱动参考图中人物动作，生成视频。
视频通过 R2 上传后传 URL，图片转 base64 直传。
"""

import asyncio
import io
import json
import os
import struct
import tempfile

import aiohttp

from ..utils.config import get_api_key_or_raise, get_async_api_base_url, NETWORK_ROUTE_OPTIONS, get_base_url_by_route
from ..utils.r2_uploader import upload_video, upload_image
from ..utils.image_utils import tensor_to_pil
from ..utils.http_error import async_request_with_retry

try:
    from comfy_api.latest import InputImpl
    import folder_paths
    _FOLDER_PATHS_OK = True
except Exception:
    _FOLDER_PATHS_OK = False


# ── 常量 ──────────────────────────────────────────────────────────────────────

_ENDPOINT_CREATE = "/kling/v1/videos/motion-control"
_ENDPOINT_STATUS = "/kling/v1/videos/motion-control/{task_id}"

_POLL_INIT = 5
_POLL_MAX  = 15


# ── 工具函数 ───────────────────────────────────────────────────────────────────



# ── 视频时长检测（纯标准库，跨平台） ──────────────────────────────────────────

def _parse_video_duration(data: bytes) -> float | None:
    """从 MP4/MOV 原始字节解析时长（秒）。读取 mvhd box。"""
    idx = data.find(b"mvhd")
    if idx == -1:
        return None
    box = data[idx + 4:]
    if len(box) < 32:
        return None
    version = box[0]
    try:
        if version == 0:
            timescale = struct.unpack(">I", box[12:16])[0]
            duration  = struct.unpack(">I", box[16:20])[0]
        else:  # version == 1
            timescale = struct.unpack(">I", box[20:24])[0]
            duration  = struct.unpack(">Q", box[24:32])[0]
    except struct.error:
        return None
    return (duration / timescale) if timescale > 0 else None


def _get_video_duration(reference_video) -> float | None:
    """从 ComfyUI VIDEO 对象获取视频时长（秒），失败返回 None。"""
    try:
        source = reference_video.get_stream_source()
        if isinstance(source, str) and os.path.isfile(source):
            with open(source, "rb") as f:
                data = f.read()
        elif isinstance(source, io.BytesIO):
            source.seek(0)
            data = source.read()
        else:
            return None
        return _parse_video_duration(data)
    except Exception:
        return None


def _validate_video_duration(reference_video, character_orientation: str):
    """校验视频时长，超限时抛出 ValueError。解析失败时静默跳过。"""
    duration = _get_video_duration(reference_video)
    if duration is None:
        print("[K3 动作控制] 无法解析视频时长，跳过校验。")
        return
    limit = 10 if character_orientation == "image" else 30
    print(f"[K3 动作控制] 检测到视频时长: {duration:.2f}s（限制: 3~{limit}s）")
    if not (3 <= duration <= limit):
        orientation_label = "图片" if character_orientation == "image" else "视频"
        raise ValueError(
            f"参考视频时长 {duration:.1f}s 不符合要求。\n"
            f"角色朝向为「{orientation_label}」时，时长须在 3~{limit}s 之间。"
        )


# ── 节点 ──────────────────────────────────────────────────────────────────────

class K3MotionControl:
    """K3 动作控制 自研 —— 用参考视频驱动参考图人物动作"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "参考图片":   ("IMAGE",),
                "参考视频":   ("VIDEO",),
                "提示词":     ("STRING", {"multiline": True, "default": ""}),
                "网络线路":   (NETWORK_ROUTE_OPTIONS, {"default": "全球加速"}),
                "模型":       (["v3", "v2-6"], {"default": "v3"}),
                "模式":       (["720p", "1080p"], {"default": "1080p"}),
                "时长":       ([5, 10, 15, 20, 25, 30], {"default": 5}),
                "角色朝向":   (["图片", "视频"], {"default": "图片"}),
                "保留原声":   (["打开", "关闭"], {"default": "打开"}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2147483647,
                    "tooltip": "seed 仅控制节点是否重新运行，结果本身不可复现。",
                }),
            },
        }

    RETURN_TYPES  = ("VIDEO",)
    RETURN_NAMES  = ("视频",)
    FUNCTION      = "generate"
    CATEGORY      = "comfyui_o1key/KVideo"

    async def generate(self, 参考图片, 参考视频, 提示词, 保留原声, 角色朝向, 模式, 模型, 时长, 网络线路, seed, **kwargs):
        api_key  = get_api_key_or_raise()
        base_url = get_base_url_by_route(网络线路)
        headers  = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }

        # ── 参数映射 ──────────────────────────────────────────────────
        mode_api              = "std" if 模式 == "720p" else "pro"
        model_name            = f"kling-{模型}-motion-{mode_api}-{时长}s"
        character_orientation = "image" if 角色朝向 == "图片" else "video"
        keep_sound            = "yes" if 保留原声 == "打开" else "no"
        prompt                = 提示词.strip()

        if len(prompt) > 2500:
            raise ValueError(f"提示词长度（{len(prompt)}）超过上限 2500 个字符，请缩短后重试。")

        # ── 进度条 ────────────────────────────────────────────────────
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(100)
        except Exception:
            pbar = None

        def _stage(s: str):
            if s == "uploading":
                print("[K3 动作控制] 上传视频到 R2...")
                if pbar: pbar.update_absolute(0, 100)
            elif s == "submitting":
                print("[K3 动作控制] 提交任务...")
                if pbar: pbar.update_absolute(10, 100)
            elif s.startswith("submitted:"):
                print(f"[K3 动作控制] 任务已提交 → {s.split(':', 1)[1]}")
                if pbar: pbar.update_absolute(15, 100)
            elif s == "downloading":
                print("[K3 动作控制] 下载视频...")
                if pbar: pbar.update_absolute(99, 100)
            elif s == "done":
                print("[K3 动作控制] 完成")
                if pbar: pbar.update_absolute(100, 100)

        def _progress(pct: int):
            if pbar: pbar.update_absolute(15 + int(pct * 0.84), 100)

        # ── 视频时长校验 ──────────────────────────────────────────────
        _validate_video_duration(参考视频, character_orientation)

        # 参考视频时长不得超过所选时长（防止用长视频生成短计费）
        _dur = _get_video_duration(参考视频)
        if _dur is not None and _dur > 时长 + 0.5:
            raise ValueError(
                f"参考视频时长 {_dur:.1f}s 超过所选时长 {时长}s。\n"
                f"请将时长调整为 ≥{_dur:.0f}s 的档位，或更换更短的参考视频。"
            )

        # ── 图片 & 视频上传 R2 → 获取公网 URL ────────────────────────
        _stage("uploading")
        pil_list  = tensor_to_pil(参考图片)
        image_url = await upload_image(pil_list[0].convert("RGB"))
        video_url = await upload_video(参考视频)

        # ── 构建请求体 ────────────────────────────────────────────────
        body: dict = {
            "model_name":            model_name,
            "model":                 model_name,
            "image_url":             image_url,
            "video_url":             video_url,
            "character_orientation": character_orientation,
            "mode":                  mode_api,
            "keep_original_sound":   keep_sound,
        }
        if prompt:
            body["prompt"] = prompt

        # ── 保存路径（临时文件，避免与下游保存节点重复落盘）──────────────────
        tmp_fd, save_path = tempfile.mkstemp(suffix=".mp4", prefix="k3_motion_")

        connector = aiohttp.TCPConnector(ssl=False, force_close=True)
        async with aiohttp.ClientSession(connector=connector) as session:

            # 1. 提交任务
            _stage("submitting")
            create_url = f"{base_url}{_ENDPOINT_CREATE}"
            resp = await async_request_with_retry(
                session, "POST", create_url,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers=headers, prefix="K3 动作控制提交: "
            )
            text = await resp.text()
            create_resp = json.loads(text)

            # task_id 兼容扁平结构和 data 嵌套结构
            task_id = (
                create_resp.get("task_id")
                or create_resp.get("id")
                or create_resp.get("data", {}).get("task_id")
            )
            if not task_id:
                raise RuntimeError(f"API 未返回任务 ID，响应：{create_resp}")
            _stage(f"submitted:{task_id}")

            # 2. 轮询
            status_url = f"{base_url}{_ENDPOINT_STATUS.format(task_id=task_id)}"
            interval   = _POLL_INIT
            video_result_url = None

            while True:
                await asyncio.sleep(interval)
                async with session.get(status_url, headers=headers) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        try:
                            err = json.loads(text)
                            msg = err.get("message") or text
                        except Exception:
                            msg = text
                        raise RuntimeError(f"状态查询失败 ({resp.status}): {msg}")
                    sr = json.loads(text)

                # 兼容扁平结构和 data 嵌套结构
                data   = sr.get("data", sr)
                status = (data.get("status") or sr.get("status") or "").lower()

                pct_raw = data.get("progress", 0)
                try:
                    pct = int(str(pct_raw).rstrip("%").strip())
                except (ValueError, AttributeError):
                    pct = 0
                print(f"[K3 动作控制] 生成中 {pct}%")
                _progress(pct)

                if status in ("success", "completed", "done", "finished", "succeed"):
                    video_result_url = (
                        data.get("video_url")
                        or data.get("result_url")
                        or data.get("url")
                        or (data.get("result", {}) or {}).get("url")
                        or sr.get("video_url")
                        or sr.get("url")
                    )
                    break
                elif status in ("failed", "fail"):
                    err_info = data.get("error") or sr.get("error") or {}
                    err_msg  = (err_info.get("message", "未知错误")
                                if isinstance(err_info, dict) else str(err_info))
                    raise RuntimeError(f"K3 动作控制生成失败：{err_msg}")

                interval = min(interval * 1.3, _POLL_MAX)

            if not video_result_url:
                raise RuntimeError(f"API 未返回视频 URL，响应：{sr}")

            # 3. 下载视频
            _stage("downloading")
            async with session.get(video_result_url, allow_redirects=True) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"视频下载失败 ({resp.status})")
                os.close(tmp_fd)
                with open(save_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        f.write(chunk)

        _stage("done")

        if _FOLDER_PATHS_OK:
            return (InputImpl.VideoFromFile(save_path),)
        return (save_path,)


# ── 视频时长检测测试节点 ──────────────────────────────────────────────────────

class K3MotionVideoCheck:
    """检测视频时长并校验是否满足动作控制的限制，不调用 API。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "参考视频": ("VIDEO",),
                "角色朝向": (["图片", "视频"], {"default": "图片"}),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("检测结果",)
    FUNCTION      = "check"
    CATEGORY      = "comfyui_o1key/KVideo"
    OUTPUT_NODE   = True

    def check(self, 参考视频, 角色朝向):
        character_orientation = "image" if 角色朝向 == "图片" else "video"
        duration = _get_video_duration(参考视频)

        if duration is None:
            result = "❌ 无法解析视频时长（格式不支持或文件损坏）"
            print(f"[K3 视频检测] {result}")
            return (result,)

        limit = 10 if character_orientation == "image" else 30
        orientation_label = 角色朝向
        ok = 3 <= duration <= limit

        if ok:
            result = (
                f"✅ 时长检测通过\n"
                f"视频时长: {duration:.2f}s\n"
                f"角色朝向: {orientation_label}（限制 3~{limit}s）"
            )
        else:
            result = (
                f"❌ 时长检测不通过\n"
                f"视频时长: {duration:.2f}s\n"
                f"角色朝向: {orientation_label}（限制 3~{limit}s）\n"
                f"请更换时长在 3~{limit}s 之间的视频。"
            )

        print(f"[K3 视频检测] {result}")
        return (result,)


# ── 节点注册 ──────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "K3MotionControl":     K3MotionControl,
    "K3MotionVideoCheck":  K3MotionVideoCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "K3MotionControl":    "动作控制 K3 自研",
    "K3MotionVideoCheck": "视频时长检测 K3",
}
