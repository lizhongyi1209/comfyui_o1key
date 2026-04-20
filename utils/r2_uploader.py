"""
R2 文件上传工具（通过 o1key 后端预签名接口）
- 插件内零 R2 凭证，仅使用用户的 O1KEY_API_KEY
- 流程：请求预签名 URL → PUT 直传 R2 → 返回公网 URL
"""

import io
import os
import uuid

import aiohttp

from .config import get_api_key_or_raise, get_api_base_url


async def _presign(filename: str, content_type: str) -> tuple:
    """向 o1key 后端请求预签名 URL，返回 (upload_url, public_url)"""
    api_key = get_api_key_or_raise()
    base_url = get_api_base_url()

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/v1/storage/presign",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"filename": filename, "content_type": content_type},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"预签名请求失败 ({resp.status}): {text}")
            data = await resp.json()

    return data["upload_url"], data["public_url"]


async def _put_upload(upload_url: str, data: bytes, content_type: str):
    """用预签名 URL 直传文件到 R2（不带 Authorization）"""
    async with aiohttp.ClientSession() as session:
        async with session.put(
            upload_url,
            data=data,
            headers={"Content-Type": content_type},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status not in (200, 204):
                text = await resp.text()
                raise RuntimeError(f"文件上传失败 ({resp.status}): {text}")


async def upload_video(video) -> str:
    """
    接受 ComfyUI VIDEO 对象，上传到 R2，返回公网 URL。
    支持 mp4 / mov 格式。
    """
    source = video.get_stream_source()

    if isinstance(source, io.BytesIO):
        source.seek(0)
        data = source.read()
        ext = "mp4"
    else:
        video_path = source
        if not video_path or not os.path.isfile(video_path):
            raise ValueError(f"无法获取参考视频文件路径（当前路径：{video_path}）")
        ext = os.path.splitext(video_path)[1].lower().lstrip(".")
        if ext not in ("mp4", "mov"):
            raise ValueError(f"参考视频格式须为 mp4 或 mov，当前为 .{ext}")
        with open(video_path, "rb") as f:
            data = f.read()

    content_type = "video/mp4" if ext == "mp4" else "video/quicktime"
    filename = f"{uuid.uuid4()}.{ext}"

    upload_url, public_url = await _presign(filename, content_type)
    await _put_upload(upload_url, data, content_type)

    print(f"[R2] 视频已上传: {public_url}")
    return public_url


async def upload_audio(audio) -> str:
    """
    接受 ComfyUI AUDIO dict（waveform tensor + sample_rate），
    编码为 WAV 后上传到 R2，返回公网 URL。
    """
    import struct
    import numpy as np

    waveform = audio["waveform"]      # shape: [B, C, N] or [C, N]
    sample_rate = int(audio["sample_rate"])

    if waveform.dim() == 3:
        waveform = waveform[0]

    wav_np = waveform.cpu().numpy()
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=0)
    wav_np = np.clip(wav_np, -1.0, 1.0)
    pcm = (wav_np * 32767).astype(np.int16)

    num_samples = len(pcm)
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate,
                          byte_rate, block_align, bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())

    data = buf.getvalue()
    filename = f"{uuid.uuid4()}.wav"

    upload_url, public_url = await _presign(filename, "audio/wav")
    await _put_upload(upload_url, data, "audio/wav")

    print(f"[R2] 音频已上传: {public_url}")
    return public_url
