"""
Comfyui_o1key - ComfyUI 自定义节点集合
通过 api.o1key.com 调用 AI 模型进行图像生成和文本生成

项目结构:
├── nodes/          # 节点实现
├── utils/          # 工具模块
├── clients/        # API 客户端
└── __init__.py     # 节点注册入口
"""


import ssl
import logging

# 屏蔽 ComfyUI 资产扫描的终端日志输出
_seeder_filter = lambda record: not any(
    kw in record.getMessage()
    for kw in ("Seeder start", "Asset scan", "Scan(", "Fast scan")
)
logging.getLogger().addFilter(_seeder_filter)

from .nodes import NanoBananaPro, BatchNanoBananaPro, GoogleGemini, LoadFile, ImageStitchPro, BatchCleanMetadata, VideoPreview, GoogleVeo, FluxImageEdit, UniversalLLMChat, KlingVideo, KlingFirstLastFrame, KlingMotionControlTest, AspectRatioPreset, BatchImagesO1key, Seedance, SeedanceMultiModal, StreamPreview, DoubaoImage, O1keyGPTImage, O1keyGrokImage, KVideoFirstLast, KVideoImage2Video
from .nodes import K3Video, K3VideoFirstLast, K3MotionControl, K3MotionVideoCheck, NanoBananaV2, NanoBananaV2Batch, SaveImageFormat
from .nodes import O1keySavePSD
from .nodes import O1keyRemoveBackground
from .nodes import O1keyColorRemoveBG

# 报错弹框友好文案（不修改原节点代码，仅在外层统一处理）
_MSG_TIMEOUT = "API 请求超时，请稍后重试或检查网络。"
_MSG_SSL_NETWORK = (
    "本地网络不太稳定!解决方案如下:\n"
    "1. 重启程序再试试看 (优先)\n"
    "2. 调整一下网络环境,如wifi或宽带等\n"
    "3. 切换VPN节点,或更换代理模式\n"
    "4. 关掉杀毒软件或防火墙\n"
    "5. 关掉浏览器VPN插件,避免冲突"
)

def _wrap_generate_for_error_display(cls, attr="generate"):
    original = getattr(cls, attr, None)
    if original is None:
        return
    def wrapped(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except TimeoutError as e:
            msg = (str(e) or "").strip()
            if not msg:
                msg = _MSG_TIMEOUT
            raise TimeoutError(msg) from None
        except (ssl.SSLError, OSError) as e:
            err_str = str(e)
            if "DECRYPTION_FAILED_OR_BAD_RECORD_MAC" in err_str or "decryption failed or bad record mac" in err_str.lower():
                raise RuntimeError(_MSG_SSL_NETWORK) from None
            raise
    setattr(cls, attr, wrapped)

_wrap_generate_for_error_display(NanoBananaPro)
_wrap_generate_for_error_display(BatchNanoBananaPro)
_wrap_generate_for_error_display(NanoBananaV2)
_wrap_generate_for_error_display(NanoBananaV2Batch)

# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "NanoBanana": NanoBananaPro,
    "BatchNanoBananaPro": BatchNanoBananaPro,
    "GoogleGemini": GoogleGemini,
    "LoadFile": LoadFile,
    "ImageStitchPro": ImageStitchPro,

    "BatchCleanMetadata": BatchCleanMetadata,
    "VideoPreview": VideoPreview,
    "GoogleVeo": GoogleVeo,
    "FluxImageEdit": FluxImageEdit,
    "UniversalLLMChat": UniversalLLMChat,
    "KlingVideo": KlingVideo,
    "KlingFirstLastFrame": KlingFirstLastFrame,
    "KlingMotionControlTest": KlingMotionControlTest,
    "AspectRatioPreset": AspectRatioPreset,

    "BatchImagesO1key": BatchImagesO1key,
    "Seedance": Seedance,
    "SeedanceMultiModal": SeedanceMultiModal,
    "StreamPreview": StreamPreview,
    "DoubaoImage": DoubaoImage,
    "O1keyGPTImage": O1keyGPTImage,
    "O1keyGrokImage": O1keyGrokImage,
    "KVideoFirstLast": KVideoFirstLast,
    "KVideoImage2Video": KVideoImage2Video,
    "K3Video": K3Video,
    "K3VideoFirstLast": K3VideoFirstLast,
    "K3MotionControl": K3MotionControl,
    "K3MotionVideoCheck": K3MotionVideoCheck,
    "NanoBananaV2": NanoBananaV2,
    "NanoBananaV2Batch": NanoBananaV2Batch,
    "SaveImageFormat": SaveImageFormat,
    "O1keySavePSD": O1keySavePSD,
    "O1keyRemoveBackground": O1keyRemoveBackground,
    "O1keyColorRemoveBG": O1keyColorRemoveBG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBanana": "Nano Banana",
    "BatchNanoBananaPro": "批量 Nano Banana",
    "GoogleGemini": "Google Gemini",
    "LoadFile": "加载文件",
    "ImageStitchPro": "图像拼接 Pro",

    "BatchCleanMetadata": "批量任务（防AI识别）",
    "VideoPreview": "预览视频",
    "GoogleVeo": "Google Veo - ab",
    "FluxImageEdit": "Flux2 图像编辑",
    "UniversalLLMChat": "全能LLM对话助手",
    "KlingVideo": "文/图生视频 自研模型",
    "KlingFirstLastFrame": "首尾帧生视频 自研模型",
    "KlingMotionControlTest": "动作控制 自研模型",
    "AspectRatioPreset": "图片宽高比预设",

    "BatchImagesO1key": "加载图像（批量）",
    "Seedance": "Seedance 视频生成",
    "SeedanceMultiModal": "Seedance 多模态参考生视频",
    "StreamPreview": "流式文本预览",
    "DoubaoImage": "豆包生图",
    "O1keyGPTImage": "o1key GPT Image",
    "O1keyGrokImage": "Grok Image",
    "KVideoFirstLast": "K26 图生视频（首尾帧）",
    "KVideoImage2Video": "K26 图生视频",
    "K3Video": "K3 图生视频 自研",
    "K3VideoFirstLast": "首尾帧 K3 自研",
    "K3MotionControl": "动作控制 K3 自研",
    "K3MotionVideoCheck": "视频时长检测 K3",
    "NanoBananaV2": "Nano Banana V2",
    "NanoBananaV2Batch": "Nano Banana V2（批量）",
    "SaveImageFormat": "保存图像（格式转换）",
    "O1keySavePSD": "保存 PSD（分层）",
    "O1keyRemoveBackground": "去背景（rembg）",
    "O1keyColorRemoveBG": "颜色去背景",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# 注册 /o1key/input_dir 接口，供前端文件上传按钮获取 input 目录绝对路径
try:
    from aiohttp import web
    from server import PromptServer
    import folder_paths
    from .utils.config import CONFIG_FILE, load_config, NETWORK_ROUTES

    @PromptServer.instance.routes.get("/o1key/input_dir")
    async def get_input_dir(request):
        import os
        path = os.path.abspath(folder_paths.get_input_directory())
        return web.json_response({"path": path})

    @PromptServer.instance.routes.get("/o1key/api_key")
    async def get_api_key_route(request):
        config = load_config()
        key = config.get("O1KEY_API_KEY", "")
        masked = ""
        if key:
            if len(key) > 8:
                masked = key[:3] + "****" + key[-4:]
            else:
                masked = "****"
        return web.json_response({"has_key": bool(key), "masked": masked})

    @PromptServer.instance.routes.post("/o1key/api_key")
    async def set_api_key_route(request):
        import os
        data = await request.json()
        new_key = data.get("api_key", "").strip()
        if not new_key:
            return web.json_response({"error": "API Key 不能为空"}, status=400)
        config = load_config()
        config["O1KEY_API_KEY"] = new_key
        lines = []
        for k, v in config.items():
            lines.append(f"{k}={v}")
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        return web.json_response({"success": True})

    @PromptServer.instance.routes.post("/o1key/test_key")
    async def test_api_key_route(request):
        import aiohttp as _aiohttp
        data = await request.json()
        test_key = data.get("api_key", "").strip()
        if not test_key:
            return web.json_response({"valid": False, "error": "密钥不能为空"})
        base_url = NETWORK_ROUTES.get("CF加速", "https://cf-api.o1key.com")
        url = f"{base_url}/v1/models"
        headers = {"Authorization": f"Bearer {test_key}"}
        try:
            async with _aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=_aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        return web.json_response({"valid": True})
                    elif resp.status == 401:
                        return web.json_response({"valid": False, "error": "密钥无效或已过期"})
                    else:
                        text = await resp.text()
                        return web.json_response({"valid": False, "error": f"验证失败 ({resp.status})"})
        except Exception as e:
            return web.json_response({"valid": False, "error": f"网络错误: {str(e)}"})

    @PromptServer.instance.routes.delete("/o1key/api_key")
    async def delete_api_key_route(request):
        config = load_config()
        config.pop("O1KEY_API_KEY", None)
        lines = []
        for k, v in config.items():
            lines.append(f"{k}={v}")
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        return web.json_response({"success": True})

    @PromptServer.instance.routes.get("/o1key/output_history")
    async def get_output_history(request):
        """读取 output 目录文件，按执行分组返回 /api/jobs 兼容格式"""
        import os, uuid, json as _json
        limit = int(request.query.get("limit", "200"))
        offset = int(request.query.get("offset", "0"))
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        meta_file = os.path.join(output_dir, ".o1key_history.json")
        meta = {}
        if os.path.isfile(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as mf:
                    meta = _json.load(mf)
            except Exception:
                pass
        supported_ext = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.mp4', '.webm'}
        # 收集所有文件并按 workflow_id 分组
        all_files = []
        for fname in os.listdir(output_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported_ext:
                continue
            fpath = os.path.join(output_dir, fname)
            if not os.path.isfile(fpath):
                continue
            mtime = os.path.getmtime(fpath)
            media = "images" if ext in {'.png','.jpg','.jpeg','.webp','.gif'} else "video"
            all_files.append({"name": fname, "mtime": mtime, "media": media})
        # 按 workflow_id 分组（同一次执行合并为一个 job）
        groups = {}
        ungrouped = []
        for f in all_files:
            m = meta.get(f["name"], {})
            wid = m.get("workflow_id")
            if wid:
                groups.setdefault(wid, []).append((f, m))
            else:
                ungrouped.append((f, {}))
        # 构建 job 列表
        jobs = []
        for wid, items in groups.items():
            items.sort(key=lambda x: x[0]["mtime"], reverse=True)
            latest = items[0]
            f, m = latest
            start_ms = int(m.get("start_time", f["mtime"]) * 1000)
            end_ms = int(m.get("end_time", f["mtime"]) * 1000)
            jobs.append({
                "id": wid,
                "status": "completed",
                "create_time": start_ms,
                "execution_start_time": start_ms,
                "execution_end_time": end_ms,
                "preview_output": {
                    "filename": f["name"],
                    "subfolder": "",
                    "type": "output",
                    "nodeId": "0",
                    "mediaType": f["media"],
                },
                "outputs_count": len(items),
                "execution_error": None,
                "workflow_id": wid,
            })
        # 无元数据的文件各自作为独立 job
        for f, m in ungrouped:
            mtime_ms = int(f["mtime"] * 1000)
            job_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f["name"]))
            jobs.append({
                "id": job_id,
                "status": "completed",
                "create_time": mtime_ms,
                "execution_start_time": mtime_ms,
                "execution_end_time": mtime_ms,
                "preview_output": {
                    "filename": f["name"],
                    "subfolder": "",
                    "type": "output",
                    "nodeId": "0",
                    "mediaType": f["media"],
                },
                "outputs_count": 1,
                "execution_error": None,
                "workflow_id": None,
            })
        # 按时间倒序排列，分页
        jobs.sort(key=lambda x: x["create_time"], reverse=True)
        total = len(jobs)
        page = jobs[offset:offset+limit]
        return web.json_response({
            "jobs": page,
            "pagination": {"offset": offset, "limit": limit, "total": total, "has_more": offset + limit < total}
        })

    @PromptServer.instance.routes.get("/o1key/output_workflow")
    async def get_output_workflow(request):
        """从 PNG 元数据中读取工作流，供前端恢复使用"""
        import os, struct, json as _json
        filename = request.query.get("filename", "")
        if not filename:
            return web.json_response({"error": "missing filename"}, status=400)
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        fpath = os.path.join(output_dir, filename)
        if not os.path.isfile(fpath) or not fpath.lower().endswith(".png"):
            return web.json_response({"error": "file not found"}, status=404)
        workflow = None
        prompt_data = None
        try:
            with open(fpath, "rb") as pf:
                pf.read(8)  # PNG signature
                while True:
                    raw = pf.read(8)
                    if len(raw) < 8:
                        break
                    length = struct.unpack(">I", raw[:4])[0]
                    chunk_type = raw[4:8]
                    data = pf.read(length)
                    pf.read(4)  # CRC
                    if chunk_type == b"tEXt":
                        key, val = data.split(b"\x00", 1)
                        k = key.decode("ascii", errors="replace")
                        if k == "workflow":
                            workflow = _json.loads(val)
                        elif k == "prompt":
                            prompt_data = _json.loads(val)
                    elif chunk_type == b"IEND":
                        break
        except Exception:
            pass
        return web.json_response({"workflow": workflow, "prompt": prompt_data})

    @PromptServer.instance.routes.get("/o1key/job_detail/{job_id}")
    async def get_job_detail(request):
        """根据 job_id (workflow_id 或 uuid5) 返回含工作流的 job 详情"""
        import os, uuid, struct, json as _json
        job_id = request.match_info["job_id"]
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        meta_file = os.path.join(output_dir, ".o1key_history.json")
        meta = {}
        if os.path.isfile(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as mf:
                    meta = _json.load(mf)
            except Exception:
                pass
        supported_ext = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.mp4', '.webm'}
        # 查找属于该 job 的所有文件（按 workflow_id 或 uuid5 匹配）
        matched_files = []
        for fname in os.listdir(output_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported_ext:
                continue
            m = meta.get(fname, {})
            if m.get("workflow_id") == job_id:
                matched_files.append(fname)
            elif str(uuid.uuid5(uuid.NAMESPACE_URL, fname)) == job_id:
                matched_files.append(fname)
        if not matched_files:
            return web.json_response({"error": "not found"}, status=404)
        # 用最新文件作为代表
        matched_files.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)
        target_file = matched_files[0]
        fpath = os.path.join(output_dir, target_file)
        m = meta.get(target_file, {})
        mtime = os.path.getmtime(fpath)
        start_ms = int(m.get("start_time", mtime) * 1000)
        end_ms = int(m.get("end_time", mtime) * 1000)
        ext = os.path.splitext(target_file)[1].lower()
        media = "images" if ext in {'.png','.jpg','.jpeg','.webp','.gif'} else "video"
        # 读取 PNG 工作流元数据
        workflow = None
        if ext == ".png":
            try:
                with open(fpath, "rb") as pf:
                    pf.read(8)
                    while True:
                        raw = pf.read(8)
                        if len(raw) < 8:
                            break
                        length = struct.unpack(">I", raw[:4])[0]
                        chunk_type = raw[4:8]
                        data = pf.read(length)
                        pf.read(4)
                        if chunk_type == b"tEXt":
                            key, val = data.split(b"\x00", 1)
                            k = key.decode("ascii", errors="replace")
                            if k == "workflow":
                                workflow = _json.loads(val)
                        elif chunk_type == b"IEND":
                            break
            except Exception:
                pass
        # 构建 outputs：包含该执行的所有文件
        outputs = {}
        for i, fname in enumerate(matched_files):
            e = os.path.splitext(fname)[1].lower()
            mt = "images" if e in {'.png','.jpg','.jpeg','.webp','.gif'} else "gifs"
            outputs.setdefault(str(i), {}).setdefault(mt, []).append(
                {"filename": fname, "subfolder": "", "type": "output"}
            )
        job_detail = {
            "id": job_id,
            "status": "completed",
            "create_time": start_ms,
            "execution_start_time": start_ms,
            "execution_end_time": end_ms,
            "preview_output": {
                "filename": target_file,
                "subfolder": "",
                "type": "output",
                "nodeId": "0",
                "mediaType": media,
            },
            "outputs_count": len(matched_files),
            "execution_error": None,
            "workflow_id": job_id,
            "workflow": {
                "extra_data": {
                    "extra_pnginfo": {"workflow": workflow}
                }
            } if workflow else None,
            "outputs": outputs,
        }
        return web.json_response(job_detail)

    @PromptServer.instance.routes.post("/o1key/delete_history")
    async def delete_history_item(request):
        """删除持久化历史记录及对应的输出文件"""
        import os, uuid, json as _json
        body = await request.json()
        job_ids = body.get("delete", [])
        if not job_ids:
            return web.json_response({"success": False, "error": "missing ids"}, status=400)
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        meta_file = os.path.join(output_dir, ".o1key_history.json")
        meta = {}
        if os.path.isfile(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as mf:
                    meta = _json.load(mf)
            except Exception:
                pass
        deleted_files = []
        for job_id in job_ids:
            files_to_remove = []
            for fname, m in list(meta.items()):
                if m.get("workflow_id") == job_id:
                    files_to_remove.append(fname)
                elif str(uuid.uuid5(uuid.NAMESPACE_URL, fname)) == job_id:
                    files_to_remove.append(fname)
            for fname in files_to_remove:
                meta.pop(fname, None)
                fpath = os.path.join(output_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        os.remove(fpath)
                        deleted_files.append(fname)
                    except Exception:
                        pass
        try:
            with open(meta_file, "w", encoding="utf-8") as mf:
                _json.dump(meta, mf, ensure_ascii=False)
        except Exception:
            pass
        return web.json_response({"success": True, "deleted": deleted_files})

    # === AI 聊天代理（流式 SSE 透传） ===
    @PromptServer.instance.routes.post("/o1key/restart")
    async def restart_server(request):
        import sys, os as _ros, subprocess, threading
        def _do_restart():
            import time
            time.sleep(1.5)
            skip = {"--auto-launch", "--auto_launch", "--launch", "--windows-standalone-build"}
            args = [a for a in sys.argv if a not in skip]
            args.append("--disable-auto-launch")
            subprocess.Popen([sys.executable] + args, cwd=_ros.getcwd())
            _ros._exit(0)
        threading.Thread(target=_do_restart, daemon=True).start()
        return web.json_response({"success": True, "message": "正在重启..."})

    # === AI 聊天代理（流式 SSE 透传） ===
    @PromptServer.instance.routes.post("/o1key/chat/completions")
    async def chat_completions_proxy(request):
        import aiohttp as _aiohttp
        import json as _cjson

        data = await request.json()
        config = load_config()
        api_key = config.get("O1KEY_API_KEY", "")
        if not api_key:
            return web.json_response({"error": "未配置 API Key"}, status=401)

        route = data.get("route", "CF加速")
        base_url = NETWORK_ROUTES.get(route, "https://cf-api.o1key.com")
        model = data.get("model", "gpt-5.5")
        messages = data.get("messages", [])

        url = f"{base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        body = {"model": model, "messages": messages, "stream": True}

        resp = web.StreamResponse(
            status=200, reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
        await resp.prepare(request)

        try:
            timeout = _aiohttp.ClientTimeout(total=120)
            async with _aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=body) as upstream:
                    if upstream.status != 200:
                        err = await upstream.text()
                        await resp.write(f"data: {_cjson.dumps({'error': err})}\n\n".encode())
                        await resp.write(b"data: [DONE]\n\n")
                        return resp
                    async for chunk in upstream.content.iter_any():
                        await resp.write(chunk)
        except Exception as e:
            await resp.write(f"data: {_cjson.dumps({'error': str(e)})}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")

        return resp

    # === 执行事件 Hook：持久化耗时元数据 ===
    import time as _time, json as _json2, os as _os
    _execution_tracker = {}

    _orig_send_sync = PromptServer.instance.send_sync

    def _patched_send_sync(event, data, *args, **kwargs):
        try:
            if event == "execution_start":
                pid = data.get("prompt_id", "")
                if pid:
                    _execution_tracker[pid] = {"start": _time.time(), "outputs": []}
            elif event == "executed":
                pid = data.get("prompt_id", "")
                output = data.get("output") or {}
                if pid and pid in _execution_tracker:
                    for img in output.get("images", []) + output.get("gifs", []):
                        if img.get("type") == "output" and img.get("filename"):
                            _execution_tracker[pid]["outputs"].append(img["filename"])
            elif event == "executing" and data.get("node") is None:
                pid = data.get("prompt_id", "")
                tracker = _execution_tracker.pop(pid, None)
                if tracker and tracker["outputs"]:
                    end_time = _time.time()
                    start_time = tracker["start"]
                    output_dir = _os.path.abspath(folder_paths.get_output_directory())
                    meta_file = _os.path.join(output_dir, ".o1key_history.json")
                    meta = {}
                    if _os.path.isfile(meta_file):
                        try:
                            with open(meta_file, "r", encoding="utf-8") as mf:
                                meta = _json2.load(mf)
                        except Exception:
                            pass
                    for fname in tracker["outputs"]:
                        meta[fname] = {
                            "start_time": start_time,
                            "end_time": end_time,
                            "outputs_count": len(tracker["outputs"]),
                            "workflow_id": pid,
                        }
                    try:
                        with open(meta_file, "w", encoding="utf-8") as mf:
                            _json2.dump(meta, mf, ensure_ascii=False)
                    except Exception:
                        pass
        except Exception:
            pass
        return _orig_send_sync(event, data, *args, **kwargs)

    PromptServer.instance.send_sync = _patched_send_sync

except Exception:
    pass
