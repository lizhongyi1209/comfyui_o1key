import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// 上传单个文件到 ComfyUI input 目录，返回服务端绝对路径
async function uploadFile(file) {
    const formData = new FormData();
    formData.append("image", file, file.name);
    const resp = await api.fetchApi("/upload/image", { method: "POST", body: formData });
    if (!resp.ok) throw new Error(`上传失败: ${file.name}`);
    const data = await resp.json();
    const inputDir = await getInputDir();
    // 拼成绝对路径（Windows 用反斜杠也可以，用正斜杠 Python 也认）
    return inputDir ? inputDir.replace(/\\/g, "/") + "/" + data.name : data.name;
}

// 获取 ComfyUI input 目录绝对路径（缓存）
let _inputDir = null;
async function getInputDir() {
    if (_inputDir !== null) return _inputDir;
    try {
        const resp = await api.fetchApi("/o1key/input_dir");
        if (resp.ok) _inputDir = (await resp.json()).path;
        else _inputDir = "";
    } catch { _inputDir = ""; }
    return _inputDir;
}

// 创建一个"选择文件"按钮，点击后弹出文件选择框
// onPaths(paths: string[]) 回调拿到上传后的路径列表
function makeUploadButton(label, accept, multiple, onPaths) {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.style.cssText =
        "width:100%;padding:4px 8px;cursor:pointer;margin-top:2px;" +
        "background:#3a5a3a;color:#ddd;border:1px solid #666;" +
        "border-radius:4px;font-size:12px;";

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.multiple = multiple;
    fileInput.accept = accept;
    fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    btn.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", async () => {
        const files = Array.from(fileInput.files);
        if (!files.length) return;
        btn.textContent = "⏳ 上传中...";
        btn.disabled = true;
        try {
            const paths = [];
            for (const f of files) paths.push(await uploadFile(f));
            onPaths(paths);
            btn.textContent = `✅ 已上传 ${files.length} 个`;
            setTimeout(() => { btn.textContent = label; }, 2000);
        } catch (e) {
            console.error("[o1key fileUpload]", e);
            btn.textContent = "❌ 上传失败";
            setTimeout(() => { btn.textContent = label; }, 2000);
        } finally {
            btn.disabled = false;
            fileInput.value = "";
        }
    });

    return btn;
}

const ACCEPT = ".pdf,.txt,.md,.csv,.json,.py,.js,.ts,.html,.xml,.docx,.xlsx,.pptx,.zip,.wav,.mp3,.png,.jpg,.jpeg,.webp";

app.registerExtension({
    name: "o1key.fileUpload",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LoadFile") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.call(this);

            const singleWidget = this.widgets?.find(w => w.name === "单文件路径");
            const folderWidget = this.widgets?.find(w => w.name === "文件夹路径");

            // "单文件路径"下方加按钮（支持多选，追加路径）
            if (singleWidget) {
                const btn = makeUploadButton("📂 选择文件（可多选）", ACCEPT, true, (paths) => {
                    const existing = singleWidget.value?.trim();
                    singleWidget.value = existing
                        ? existing + ", " + paths.join(", ")
                        : paths.join(", ");
                    singleWidget.callback?.(singleWidget.value);
                    app.graph.setDirtyCanvas(true);
                });
                this.addDOMWidget("upload_single_btn", "btn", btn, {
                    getValue() { return null; },
                    setValue() {},
                });
            }

            // 清空按钮：同时清空单文件路径和文件夹路径
            if (singleWidget || folderWidget) {
                const clearBtn = document.createElement("button");
                clearBtn.textContent = "🗑 清空文件路径";
                clearBtn.style.cssText =
                    "width:100%;padding:4px 8px;cursor:pointer;margin-top:2px;" +
                    "background:#5a3a3a;color:#ddd;border:1px solid #666;" +
                    "border-radius:4px;font-size:12px;";
                clearBtn.addEventListener("click", () => {
                    if (singleWidget) { singleWidget.value = ""; singleWidget.callback?.(""); }
                    if (folderWidget) { folderWidget.value = ""; folderWidget.callback?.(""); }
                    app.graph.setDirtyCanvas(true);
                });
                this.addDOMWidget("clear_paths_btn", "btn", clearBtn, {
                    getValue() { return null; },
                    setValue() {},
                });
            }
        };
    },
});
