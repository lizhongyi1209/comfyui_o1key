import asyncio
from typing import Any, Dict


try:
    from comfy.model_management import processing_interrupted, InterruptProcessingException
    INTERRUPT_AVAILABLE = True
except Exception:
    INTERRUPT_AVAILABLE = False
    processing_interrupted = lambda: False

    class InterruptProcessingException(Exception):
        pass


SUCCESS_STATUSES = {"succeed", "succeeded", "success", "completed", "done", "finished"}
FAILURE_STATUSES = {
    "fail",
    "failed",
    "failure",
    "error",
    "expired",
    "timeout",
    "timed_out",
    "cancel",
    "canceled",
    "cancelled",
    "rejected",
}


def check_interrupt() -> None:
    if INTERRUPT_AVAILABLE and processing_interrupted():
        raise InterruptProcessingException()


async def interruptible_sleep(seconds: float, step: float = 0.2) -> None:
    elapsed = 0.0
    while elapsed < seconds:
        check_interrupt()
        delay = min(step, seconds - elapsed)
        await asyncio.sleep(delay)
        elapsed += delay
    check_interrupt()


async def run_with_interrupt(coro, step: float = 0.2):
    task = asyncio.ensure_future(coro)
    try:
        while not task.done():
            check_interrupt()
            await asyncio.wait({task}, timeout=step)
        check_interrupt()
        return await task
    except InterruptProcessingException:
        task.cancel()
        try:
            await task
        except BaseException:
            pass
        raise


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested_payloads(payload: Dict[str, Any]):
    root = _as_dict(payload)
    data = _as_dict(root.get("data"))
    inner = _as_dict(data.get("data"))
    return root, data, inner


def extract_status(payload: Dict[str, Any]) -> str:
    root, data, inner = _nested_payloads(payload)
    keys = ("status", "task_status", "state", "task_state")
    statuses = []
    for source in (data, inner, root):
        for key in keys:
            value = source.get(key)
            if value is not None and str(value).strip():
                statuses.append(str(value).strip().lower())
    for status in statuses:
        if status in FAILURE_STATUSES or any(
            token in status for token in ("fail", "error", "reject", "timeout", "cancel")
        ):
            return status
    for status in statuses:
        if status in SUCCESS_STATUSES:
            return status
    return statuses[0] if statuses else ""


def extract_progress(payload: Dict[str, Any]) -> int:
    root, data, inner = _nested_payloads(payload)
    for source in (data, inner, root):
        value = source.get("progress")
        if value is None:
            continue
        try:
            return max(0, min(100, int(float(str(value).strip().rstrip("%")))))
        except (TypeError, ValueError):
            return 0
    return 0


def extract_error_message(payload: Dict[str, Any], default: str = "未知错误") -> str:
    root, data, inner = _nested_payloads(payload)
    keys = (
        "fail_reason",
        "failure_reason",
        "task_status_msg",
        "status_msg",
        "error_message",
        "message",
        "msg",
        "reason",
        "detail",
        "details",
    )
    for source in (data, inner, root):
        error = source.get("error")
        if isinstance(error, dict):
            for key in ("message", "msg", "detail", "reason", "code"):
                value = error.get(key)
                if value:
                    return str(value)
        elif error:
            return str(error)

        for key in keys:
            value = source.get(key)
            if value:
                return str(value)
    return default


def extract_video_url(payload: Dict[str, Any]) -> str | None:
    root, data, inner = _nested_payloads(payload)
    for source in (data, inner, root):
        for key in ("video_url", "result_url", "url", "download_url"):
            value = source.get(key)
            if value:
                return str(value)

        result = _as_dict(source.get("result"))
        for key in ("video_url", "result_url", "url", "download_url"):
            value = result.get(key)
            if value:
                return str(value)

        content = _as_dict(source.get("content"))
        value = content.get("video_url") or content.get("url")
        if value:
            return str(value)

        task_result = _as_dict(source.get("task_result"))
        videos = task_result.get("videos")
        if isinstance(videos, list) and videos:
            first = _as_dict(videos[0])
            value = first.get("url") or first.get("video_url")
            if value:
                return str(value)
    return None


def is_success_status(status: str) -> bool:
    return status in SUCCESS_STATUSES


def is_failure_status(status: str, payload: Dict[str, Any] | None = None) -> bool:
    if status in FAILURE_STATUSES:
        return True
    if any(token in status for token in ("fail", "error", "reject", "timeout", "cancel")):
        return True
    if payload is None:
        return False
    root, data, inner = _nested_payloads(payload)
    failure_keys = ("error", "fail_reason", "failure_reason", "task_status_msg", "error_message")
    return any(any(source.get(key) for key in failure_keys) for source in (data, inner, root))
