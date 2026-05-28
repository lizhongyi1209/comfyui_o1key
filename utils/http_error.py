"""
统一 HTTP 错误处理 & 退避重试模块

使用方式：
  1. 对于 aiohttp 请求，用 async_request_with_retry() 包裹 POST/GET 调用
  2. 对于已拿到 status code 的场景，调用 raise_for_status() 抛出友好错误

新增生图/视频节点时，请统一使用本模块处理 HTTP 错误。
"""

import asyncio
import random
from typing import Optional

import aiohttp


# ═══════════════════════════════════════════════════════════════════════════════
# 状态码 → 用户友好文案
# ═══════════════════════════════════════════════════════════════════════════════

HTTP_ERROR_MESSAGES = {
    429: "模型速率超限或额度不足！",
    502: "网关超时。请重试或将网络切换为美国直连",
    503: "模型超载。请稍后重试！",
    504: "网关超时。请稍后重试。",
}

# 错误内容关键词 → 用户友好文案（优先于状态码匹配）
ERROR_CONTENT_MESSAGES = {
    "Your request was rejected by the safety system": "请求被安全系统拦截：请调整提示词，避免敏感、违规、血腥、色情、仇恨、未成年人或真实人物等高风险内容。",
    "safety system": "请求被安全系统拦截：请调整提示词，避免敏感、违规、血腥、色情、仇恨、未成年人或真实人物等高风险内容。",
    "unexpected end of JSON input": "通常重试能解决；反复出现就降低分辨率、数量或换网络线路。",
    "The current model has a high load": "模型过载，请稍后重试！",
    "system error": "系统错误，请稍后重试。",
}

# 可退避重试的状态码
RETRYABLE_STATUS_CODES = {429, 502, 503, 504, 524}

# 退避重试默认参数
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 2.0      # 首次重试等待秒数
DEFAULT_MAX_DELAY = 30.0      # 最大等待秒数
DEFAULT_BACKOFF_FACTOR = 2.0  # 指数退避因子


def get_friendly_message(status_code: int, raw_message: str = "") -> str:
    """根据状态码/错误内容返回友好文案，未匹配则返回原始信息"""
    if status_code == 524:
        return "Gateway timed out while waiting for upstream image generation. Please retry, lower resolution/count, or switch network route."
    if raw_message:
        raw_message_lower = raw_message.lower()
        for keyword, friendly_msg in ERROR_CONTENT_MESSAGES.items():
            if keyword.lower() in raw_message_lower:
                return friendly_msg
    if status_code == 500:
        return "服务器返回 500：上游生成失败或服务端临时异常。请稍后重试；如果多次出现，请降低分辨率/数量，或调整提示词。"
    friendly = HTTP_ERROR_MESSAGES.get(status_code)
    if friendly:
        return friendly
    return raw_message or f"请求失败 ({status_code})"


def raise_for_status(status_code: int, raw_message: str = "", prefix: str = ""):
    """根据状态码抛出带友好文案的 RuntimeError"""
    friendly = get_friendly_message(status_code, raw_message)
    full_msg = f"{prefix}{friendly}" if prefix else friendly
    raise RuntimeError(full_msg)


def is_retryable(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES


def _compute_delay(attempt: int, base_delay: float, max_delay: float, backoff_factor: float) -> float:
    """计算第 attempt 次重试的等待时间（含 jitter）"""
    delay = base_delay * (backoff_factor ** attempt)
    delay = min(delay, max_delay)
    jitter = random.uniform(0, delay * 0.3)
    return delay + jitter


async def async_request_with_retry(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    prefix: str = "",
    **request_kwargs,
) -> aiohttp.ClientResponse:
    """
    带退避重试的 aiohttp 请求。

    仅对 RETRYABLE_STATUS_CODES (429/502/503/504/524) 进行重试。
    超过最大重试次数后抛出友好 RuntimeError。
    成功时返回 response 对象（调用者需在 async with 外自行处理 body）。

    用法示例：
        resp = await async_request_with_retry(session, "POST", url, json=body, headers=headers)
        data = await resp.json()
    """
    last_status: Optional[int] = None
    last_message = ""

    for attempt in range(max_retries + 1):
        try:
            resp = await session.request(method, url, **request_kwargs)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < max_retries:
                delay = _compute_delay(attempt, base_delay, max_delay, backoff_factor)
                print(f"{prefix}网络错误，{delay:.1f}s 后重试 ({attempt+1}/{max_retries})...")
                await asyncio.sleep(delay)
                continue
            raise RuntimeError(f"{prefix}网络错误: {e}") from None

        if resp.status == 200:
            return resp

        last_status = resp.status
        try:
            last_message = await resp.text()
        except Exception:
            last_message = ""

        if is_retryable(resp.status) and attempt < max_retries:
            delay = _compute_delay(attempt, base_delay, max_delay, backoff_factor)
            friendly = get_friendly_message(resp.status)
            print(f"{prefix}{friendly} {delay:.1f}s 后重试 ({attempt+1}/{max_retries})...")
            await asyncio.sleep(delay)
            continue

        break

    if last_status and last_status in HTTP_ERROR_MESSAGES:
        raise_for_status(last_status, raw_message=last_message, prefix=prefix)

    friendly = get_friendly_message(last_status or 0, last_message)
    raise RuntimeError(f"{prefix}{friendly}")
