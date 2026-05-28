"""
Nano Banana 节点 (V3)
ComfyUI 自定义节点，用于调用生图模型（OpenAI 兼容接口）
使用 V3 DynamicCombo 实现模型-宽高比-分辨率动态联动
"""

import io as _io
import re
import json
import time
import math
import base64
import random
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import torch
import numpy as np
from PIL import Image

from comfy_api.latest import io

from ..utils.image_utils import tensor_to_pil, pil_to_tensor, parse_batch_prompts, encode_images_for_image_size_limit
from ..utils.config import (
    NETWORK_ROUTE_OPTIONS,
    get_base_url_by_route,
    get_api_key_or_raise,
)
from ..utils.http_error import RETRYABLE_STATUS_CODES, HTTP_ERROR_MESSAGES, _compute_delay, DEFAULT_MAX_RETRIES, DEFAULT_BASE_DELAY, DEFAULT_MAX_DELAY, DEFAULT_BACKOFF_FACTOR
from ..clients.gemini_client import GeminiAPIClient

try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

try:
    from comfy.utils import ProgressBar
    PROGRESS_BAR_AVAILABLE = True
except ImportError:
    PROGRESS_BAR_AVAILABLE = False

try:
    from comfy.model_management import processing_interrupted, InterruptProcessingException
    INTERRUPT_AVAILABLE = True
except ImportError:
    INTERRUPT_AVAILABLE = False
    InterruptProcessingException = RuntimeError
    processing_interrupted = lambda: False

try:
    import psutil
    MEMORY_MONITOR_AVAILABLE = True
except ImportError:
    MEMORY_MONITOR_AVAILABLE = False

DEBUG_LOG_ENABLED = True
REQUEST_LOG_ENABLED = False

_NODE = "Nano Banana"
_ENDPOINT = "/v1/chat/completions"
_REQUEST_TIMEOUT = 900
_INTERRUPT_CHECK_INTERVAL = 0.2

_client_instance = None


def _get_client():
    global _client_instance
    if _client_instance is None:
        _client_instance = GeminiAPIClient()
    return _client_instance


async def _poll_interrupt():
    while True:
        await asyncio.sleep(_INTERRUPT_CHECK_INTERVAL)
        if INTERRUPT_AVAILABLE and processing_interrupted():
            return


async def _run_with_interrupt(coro):
    if not INTERRUPT_AVAILABLE:
        return await coro

    request_task = asyncio.ensure_future(coro)
    interrupt_task = asyncio.ensure_future(_poll_interrupt())

    done, pending = await asyncio.wait(
        [request_task, interrupt_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    if interrupt_task in done and request_task not in done:
        raise InterruptProcessingException()

    return request_task.result()


def _check_interrupt():
    if INTERRUPT_AVAILABLE and processing_interrupted():
        raise InterruptProcessingException()


def _images_to_tensor_safe(images: List[Image.Image], node_label: str) -> torch.Tensor:
    if not images:
        placeholder = Image.new('RGB', (512, 512), color=(128, 128, 128))
        return pil_to_tensor([placeholder])

    base_size = max(images, key=lambda img: img.size[0] * img.size[1]).size
    matched = [img for img in images if img.size == base_size]
    skipped = [img for img in images if img.size != base_size]

    if skipped:
        sizes_str = ", ".join(f"{img.size[0]}x{img.size[1]}" for img in skipped)
        print(
            f"{node_label}: 丢弃 {len(skipped)} 张较小尺寸的图 ({sizes_str})，"
            f"仅输出最大尺寸 {base_size[0]}x{base_size[1]} 的 {len(matched)} 张"
        )

    return pil_to_tensor(matched)


MODEL_ID_MAP = {
    "Nano Banana Pro": "nano-banana-pro",
    "Nano Banana 2": "nano-banana-2",
    "Nano Banana": "nano-banana",
}
RESOLUTION_KEY_MAP = {
    "512px": "0.5k",
    "1K": "1k",
    "2K": "2k",
    "4K": "4k",
}
BILLING_SPECIAL_ONLY = {"nano-banana"}


def _build_model_id(model_name: str, resolution: str, billing: str) -> str:
    base = MODEL_ID_MAP.get(model_name, "nano-banana-pro")

    if base == "nano-banana":
        if billing == "官方":
            raise ValueError(f"模型 \"{model_name}\" 仅支持特价计费")
        return "nano-banana"

    res_key = RESOLUTION_KEY_MAP.get(resolution, "2k")
    is_official = (billing == "官方")

    if base == "nano-banana-pro" and res_key == "1k" and not is_official:
        return "nano-banana-pro"

    if base == "nano-banana-2" and res_key == "0.5k":
        if is_official:
            raise ValueError("Nano Banana 2 的 512px 分辨率仅支持特价计费")
        return "nano-banana-2-0.5k"

    model_id = f"{base}-{res_key}"
    if is_official:
        model_id += "-official"
    return model_id

_IMAGE_RE = re.compile(r"!\[.*?\]\(data:image/(\w+);base64,([A-Za-z0-9+/=]+)\)")


def _get_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-cache, no-transform",
    }


def _build_request_body(
    prompt: str,
    model: str,
    aspect_ratio: str,
    resolution: str,
    images: Optional[List[Image.Image]] = None,
    enable_grounding: bool = False,
    thinking_level: Optional[str] = None,
) -> dict:
    google_config = {
        "image_config": {
            "image_size": resolution,
        }
    }
    if aspect_ratio and aspect_ratio != "智能":
        google_config["image_config"]["aspect_ratio"] = aspect_ratio
    if thinking_level:
        google_config["thinking_config"] = {
            "thinking_level": thinking_level.lower(),
            "include_thoughts": True,
        }

    def _make_body(encoded_images: Optional[List[tuple]] = None) -> dict:
        content_parts = [{"type": "text", "text": prompt}]
        if encoded_images:
            for mime_type, b64 in encoded_images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64}"}
                })

        body = {
            "model": model,
            "stream": True,
            "messages": [{"role": "user", "content": content_parts}],
            "extra_body": {"google": google_config},
        }

        if enable_grounding:
            body["extra_body"]["google_search"] = True
        return body

    encoded_images = None
    if images:
        encoded_images = encode_images_for_image_size_limit(images)

    return _make_body(encoded_images)


async def _generate_single(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    prompt: str,
    model: str,
    resolution: str,
    aspect_ratio: str,
    images: Optional[List[Image.Image]] = None,
    enable_grounding: bool = False,
    thinking_level: Optional[str] = None,
) -> List[Image.Image]:
    url = f"{base_url}{_ENDPOINT}"
    headers = _get_headers(api_key)
    body = _build_request_body(
        prompt=prompt,
        model=model,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        images=images,
        enable_grounding=enable_grounding,
        thinking_level=thinking_level,
    )

    if REQUEST_LOG_ENABLED:
        extra = json.dumps(body.get("extra_body", {}), ensure_ascii=False)
        print(f"[请求] POST {url} | model={model} | extra_body={extra}")

    last_status = None
    resp = None
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT, connect=30, sock_read=_REQUEST_TIMEOUT)
    for attempt in range(DEFAULT_MAX_RETRIES + 1):
        _check_interrupt()
        resp = await session.post(url, headers=headers, json=body, timeout=timeout)
        if resp.status == 200:
            break
        last_status = resp.status
        if resp.status in RETRYABLE_STATUS_CODES and attempt < DEFAULT_MAX_RETRIES:
            friendly = HTTP_ERROR_MESSAGES.get(resp.status, f"请求失败 ({resp.status})")
            delay = _compute_delay(attempt, DEFAULT_BASE_DELAY, DEFAULT_MAX_DELAY, DEFAULT_BACKOFF_FACTOR)
            print(f"Nano Banana: {friendly} {delay:.1f}s 后重试 ({attempt+1}/{DEFAULT_MAX_RETRIES})...")
            resp.close()
            await asyncio.sleep(delay)
            continue
        error_text = await resp.text()
        resp.close()
        if resp.status in HTTP_ERROR_MESSAGES:
            raise RuntimeError(HTTP_ERROR_MESSAGES[resp.status])
        try:
            err_json = json.loads(error_text)
            msg = err_json.get("error", {}).get("message", error_text[:200])
        except Exception:
            msg = error_text[:200]
        raise RuntimeError(f"API 错误 ({resp.status}): {msg}")
    else:
        if last_status and last_status in HTTP_ERROR_MESSAGES:
            raise RuntimeError(HTTP_ERROR_MESSAGES[last_status])
        raise RuntimeError(f"API 错误: 重试 {DEFAULT_MAX_RETRIES} 次后仍然失败")

    full_content = ""
    buffer = ""
    t_request = time.time()
    t_first_token = None
    try:
        async for raw_chunk in resp.content.iter_any():
            _check_interrupt()
            if t_first_token is None:
                t_first_token = time.time()
            buffer += raw_chunk.decode("utf-8")
            while "\n" in buffer:
                line_str, buffer = buffer.split("\n", 1)
                line_str = line_str.strip()
                if not line_str or not line_str.startswith("data:"):
                    continue
                data_str = line_str[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta:
                        full_content += delta["content"]
                except (json.JSONDecodeError, IndexError):
                    continue
    except aiohttp.ClientPayloadError as e:
        if full_content and _IMAGE_RE.search(full_content):
            print(f"Nano Banana: 响应流提前结束，但已收到完整图片，继续解析 ({e})")
        else:
            raise RuntimeError(f"响应流下载中断，请重试或检查网络/代理: {e}") from None
    finally:
        if resp is not None:
            resp.close()
    t_done = time.time()

    if not full_content:
        raise RuntimeError("API 未返回有效内容")

    # 思考模型可能输出多张临时图片，最终图片始终是最后一张
    matches = list(_IMAGE_RE.finditer(full_content))
    if not matches:
        raise RuntimeError(f"响应中未找到图片: {full_content[:100]}")

    last_match = matches[-1]
    img_data = base64.b64decode(last_match.group(2))
    final_image = Image.open(_io.BytesIO(img_data)).convert("RGB")

    first_token_ms = (t_first_token - t_request) * 1000 if t_first_token else 0
    download_ms = (t_done - t_first_token) * 1000 if t_first_token else 0

    return [final_image], first_token_ms, download_ms


async def _generate_single_task(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    prompt: str,
    model: str,
    resolution: str,
    aspect_ratio: str,
    images: Optional[List[Image.Image]],
    global_task_index: int,
    enable_grounding: bool = False,
    thinking_level: Optional[str] = None,
) -> dict:
    result = {
        "global_task_index": global_task_index,
        "prompt": prompt,
        "success": False,
        "generated_count": 0,
        "output_images": [],
        "error": None,
    }
    try:
        gen_images, first_token_ms, download_ms = await _generate_single(
            session=session,
            base_url=base_url,
            api_key=api_key,
            prompt=prompt,
            model=model,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            images=images if images else None,
            enable_grounding=enable_grounding,
            thinking_level=thinking_level,
        )
        result["output_images"] = gen_images
        result["success"] = True
        result["generated_count"] = len(gen_images)
    except InterruptProcessingException:
        raise
    except Exception as e:
        result["error"] = str(e)
    return result


async def _process_batch_async(
    base_url: str,
    api_key: str,
    prompts: List[str],
    model: str,
    resolution: str,
    aspect_ratio: str,
    images_per_prompt: int,
    input_images: Optional[List[Image.Image]],
    pbar=None,
    enable_grounding: bool = False,
    thinking_level: Optional[str] = None,
) -> List[dict]:
    tasks_def = []
    for p_idx, prompt in enumerate(prompts):
        for sub_idx in range(images_per_prompt):
            tasks_def.append((p_idx, sub_idx, prompt))

    total_tasks = len(tasks_def)
    max_concurrent = 50
    num_batches = math.ceil(total_tasks / max_concurrent)

    all_results = []
    completed = 0
    success_count = 0
    fail_count = 0

    connector = aiohttp.TCPConnector(ssl=False, limit=0, limit_per_host=0)

    async with aiohttp.ClientSession(connector=connector) as session:
        for batch_idx in range(num_batches):
            _check_interrupt()
            start_idx = batch_idx * max_concurrent
            end_idx = min(start_idx + max_concurrent, total_tasks)

            tasks = []
            for i in range(start_idx, end_idx):
                _check_interrupt()
                _, _, prompt = tasks_def[i]
                task = asyncio.create_task(
                    _generate_single_task(
                        session=session,
                        base_url=base_url,
                        api_key=api_key,
                        prompt=prompt,
                        model=model,
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                        images=input_images,
                        global_task_index=i,
                        enable_grounding=enable_grounding,
                        thinking_level=thinking_level,
                    )
                )
                tasks.append(task)

            batch_results = []
            for coro in asyncio.as_completed(tasks):
                _check_interrupt()
                result_data = None
                try:
                    result = await coro
                    if isinstance(result, Exception):
                        result_data = {"success": False, "error": str(result), "generated_count": 0, "output_images": [], "prompt": ""}
                    else:
                        result_data = result
                except InterruptProcessingException:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
                except Exception as e:
                    result_data = {"success": False, "error": str(e), "generated_count": 0, "output_images": [], "prompt": ""}

                batch_results.append(result_data)
                completed += 1
                prompt_snippet = (result_data.get("prompt", "") or "")[:30]

                if result_data and result_data.get("success", False):
                    success_count += 1
                    count = result_data.get("generated_count", 1)
                    print(f"Nano Banana: [{completed}/{total_tasks}] {prompt_snippet}{'...' if len(prompt_snippet) >= 30 else ''} → ✓成功({count}张)")
                else:
                    fail_count += 1
                    error_msg = result_data.get("error", "未知错误") if result_data else "未知错误"
                    print(f"Nano Banana: [{completed}/{total_tasks}] {prompt_snippet}{'...' if len(prompt_snippet) >= 30 else ''} → ✗失败: {error_msg}")

                if pbar is not None:
                    pbar.update(1)

            all_results.extend(batch_results)
            import gc; gc.collect()
            await asyncio.sleep(0.1)

    return all_results


class NanoBanana(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NanoBanana",
            display_name="Nano Banana",
            category="image/generation",
            inputs=[
                io.String.Input(
                    "prompt",
                    default="一个中国女子的OOTD",
                    multiline=True,
                ),
                io.DynamicCombo.Input("模型", options=[
                    io.DynamicCombo.Option("Nano Banana Pro", [
                        io.Combo.Input("宽高比", options=[
                            "智能", "1:1", "2:3", "3:2", "3:4", "4:3",
                            "4:5", "5:4", "9:16", "16:9", "21:9",
                        ], default="智能"),
                        io.Combo.Input("分辨率", options=["1K", "2K", "4K"], default="2K"),
                    ]),
                    io.DynamicCombo.Option("Nano Banana 2", [
                        io.Combo.Input("宽高比", options=[
                            "智能", "1:1", "1:4", "1:8", "2:3", "3:2", "3:4",
                            "4:1", "4:3", "4:5", "5:4", "8:1",
                            "9:16", "16:9", "21:9",
                        ], default="智能"),
                        io.Combo.Input("分辨率", options=["512px", "1K", "2K", "4K"], default="2K"),
                        io.Combo.Input("思考深度", options=["高", "低"], default="高"),
                    ]),
                    io.DynamicCombo.Option("Nano Banana", [
                        io.Combo.Input("宽高比", options=[
                            "智能", "1:1", "2:3", "3:2", "3:4", "4:3",
                            "4:5", "5:4", "9:16", "16:9", "21:9",
                        ], default="智能"),
                        io.Combo.Input("分辨率", options=["1K"], default="1K"),
                    ]),
                ]),
                io.Int.Input("生图数量", default=1, min=1, max=1000, step=1),
                io.Combo.Input("网络", options=NETWORK_ROUTE_OPTIONS, default="全球加速"),
                io.Combo.Input("计费", options=["特价", "官方"], default="特价"),
                io.Combo.Input("谷歌搜索", options=["关闭", "打开"], default="关闭"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                io.Image.Input("参考图1", optional=True),
                io.Image.Input("参考图2", optional=True),
                io.Image.Input("参考图3", optional=True),
                io.Image.Input("参考图4", optional=True),
                io.Image.Input("参考图5", optional=True),
                io.Image.Input("参考图6", optional=True),
                io.Image.Input("参考图7", optional=True),
                io.Image.Input("参考图8", optional=True),
                io.Image.Input("参考图9", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="输出图像"),
            ],
        )

    @classmethod
    def execute(cls, prompt, 模型, 生图数量, 计费, 网络, 谷歌搜索, seed, **kwargs) -> io.NodeOutput:
        start_time = time.time()
        was_interrupted = False

        model_name = 模型["模型"]
        宽高比 = 模型["宽高比"]
        分辨率 = 模型["分辨率"]
        思考深度 = 模型.get("思考深度")

        enable_grounding = (谷歌搜索 == "打开")

        thinking_level = None
        if model_name == "Nano Banana 2" and 思考深度:
            thinking_level = "High" if 思考深度 == "高" else "Low"

        actual_model = _build_model_id(model_name, 分辨率, 计费)

        api_key = get_api_key_or_raise("O1KEY_API_KEY")
        base_url = get_base_url_by_route(网络)

        pbar = ProgressBar(生图数量) if PROGRESS_BAR_AVAILABLE else None

        try:
            random.seed(seed)
            np.random.seed(seed % (2**32))

            input_images = []
            for i in range(1, 10):
                key = f"参考图{i}"
                if key in kwargs and kwargs[key] is not None:
                    pil_imgs = tensor_to_pil(kwargs[key])
                    input_images.extend(pil_imgs)

            if input_images and len(input_images) > 14:
                raise ValueError(f"输入图像数量 {len(input_images)} 超过限制 14 张")

            batch_prompts = parse_batch_prompts(prompt)

            grounding_str = " | 谷歌搜索接地" if enable_grounding else ""
            thinking_str = f" | 思考:{thinking_level}" if thinking_level else ""
            if batch_prompts:
                num_prompts = len(batch_prompts)
                total_images = num_prompts * 生图数量
                mode_str = f"批量提示词模式 ({num_prompts}个提示词)"
                if input_images:
                    mode_str += f" (输入{len(input_images)}张)"
                print(f"Nano Banana: {mode_str} | {分辨率} {宽高比} | 共{total_images}张{grounding_str}{thinking_str}")
            else:
                mode_str = f"图生图模式 (输入{len(input_images)}张)" if input_images else "文生图模式"
                print(f"Nano Banana: {mode_str} | {分辨率} {宽高比} | {生图数量}张{grounding_str}{thinking_str}")

            if batch_prompts or 生图数量 > 1:
                prompts = batch_prompts if batch_prompts else [prompt]
                images_per_prompt = 生图数量
                total_tasks = len(prompts) * images_per_prompt

                if pbar is not None:
                    pbar = ProgressBar(total_tasks)

                def run_async_in_thread():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            _run_with_interrupt(_process_batch_async(
                                base_url=base_url,
                                api_key=api_key,
                                prompts=prompts,
                                model=actual_model,
                                resolution=分辨率,
                                aspect_ratio=宽高比,
                                images_per_prompt=images_per_prompt,
                                input_images=input_images,
                                pbar=pbar,
                                enable_grounding=enable_grounding,
                                thinking_level=thinking_level,
                            ))
                        )
                    finally:
                        loop.close()

                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_async_in_thread)
                    try:
                        results = future.result(timeout=_REQUEST_TIMEOUT)
                    except TimeoutError:
                        raise RuntimeError(f"任务执行超时（{_REQUEST_TIMEOUT}秒）")

                success_count = sum(1 for r in results if r.get("success", False))
                fail_count = len(results) - success_count
                elapsed = time.time() - start_time
                time_str = f"{elapsed:.3f}s" if elapsed < 1 else f"{elapsed:.2f}s"
                print(f"完成！总耗时 {time_str} | 成功: {success_count}/{total_tasks} | 失败: {fail_count}")

                output_images = []
                for r in results:
                    output_images.extend(r.get("output_images", []))
                if not output_images:
                    placeholder = Image.new('RGB', (512, 512), color=(128, 128, 128))
                    output_images = [placeholder]

                output_tensor = _images_to_tensor_safe(output_images, _NODE)
                import gc; gc.collect()
                return io.NodeOutput(output_tensor)

            else:
                def run_single():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        async def _do():
                            connector = aiohttp.TCPConnector(ssl=False)
                            async with aiohttp.ClientSession(connector=connector) as session:
                                return await _generate_single(
                                    session=session,
                                    base_url=base_url,
                                    api_key=api_key,
                                    prompt=prompt,
                                    model=actual_model,
                                    resolution=分辨率,
                                    aspect_ratio=宽高比,
                                    images=input_images if input_images else None,
                                    enable_grounding=enable_grounding,
                                    thinking_level=thinking_level,
                                )
                        return loop.run_until_complete(_run_with_interrupt(_do()))
                    finally:
                        loop.close()

                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_single)
                    generated_images, first_token_ms, download_ms = future.result(timeout=_REQUEST_TIMEOUT)

                if pbar is not None:
                    pbar.update(1)

                output_tensor = _images_to_tensor_safe(generated_images, _NODE)
                elapsed = time.time() - start_time
                time_str = f"{elapsed:.3f}s" if elapsed < 1 else f"{elapsed:.2f}s"
                ft_str = f"{first_token_ms/1000:.2f}s"
                dl_str = f"{download_ms/1000:.2f}s"
                print(f"完成！总耗时 {time_str} | 首字 {ft_str} | 下载 {dl_str} | 成功 {len(generated_images)}张")

                import gc; gc.collect()
                return io.NodeOutput(output_tensor)

        except InterruptProcessingException:
            was_interrupted = True
            print("Nano Banana: 用户取消")
            raise
        except ValueError as e:
            if str(e) == "未授权！":
                print("请联系作者授权后方可使用！")
                raise ValueError("未授权！") from None
            raise ValueError(str(e)) from None
        except RuntimeError as e:
            raise RuntimeError(str(e)) from None
        except Exception as e:
            raise type(e)(str(e)) from None
        finally:
            if not was_interrupted:
                try:
                    client = _get_client()
                    client.base_url = base_url
                    balance_data = client.query_balance_sync()
                    balance_info = client.format_balance_info(balance_data)
                    print(f"Nano Banana: {balance_info}")
                except Exception:
                    pass
            import gc; gc.collect()
