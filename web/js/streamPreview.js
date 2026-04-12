import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// ── marked.js 懒加载 ──────────────────────────────────────────────────────────
let markedReady = null;
function loadMarked() {
    if (markedReady) return markedReady;
    markedReady = new Promise((resolve) => {
        if (window.marked) { resolve(window.marked); return; }
        const s = document.createElement("script");
        s.src = "https://cdn.jsdelivr.net/npm/marked/marked.min.js";
        s.onload = () => resolve(window.marked);
        s.onerror = () => resolve(null);
        document.head.appendChild(s);
    });
    return markedReady;
}

// ── 节点 UI 构建 ──────────────────────────────────────────────────────────────
function buildUI(node) {
    if (node._spContainer) return;

    const container = document.createElement("div");
    container.style.cssText =
        "width:100%;height:100%;box-sizing:border-box;padding:6px;" +
        "display:flex;flex-direction:column;gap:4px;";

    const toolbar = document.createElement("div");
    toolbar.style.cssText =
        "display:flex;justify-content:flex-end;gap:6px;align-items:center;";

    const mdToggle = document.createElement("button");
    mdToggle.textContent = "MD";
    mdToggle.title = "切换 Markdown / 纯文本";
    mdToggle.style.cssText =
        "font-size:10px;padding:2px 6px;border-radius:3px;cursor:pointer;" +
        "background:#2a5a2a;color:#ccc;border:1px solid #666;";

    const copyBtn = document.createElement("button");
    copyBtn.textContent = "复制";
    copyBtn.style.cssText =
        "font-size:10px;padding:2px 6px;border-radius:3px;cursor:pointer;" +
        "background:#444;color:#ccc;border:1px solid #666;";

    toolbar.appendChild(mdToggle);
    toolbar.appendChild(copyBtn);

    const isDedicatedPreview = node.comfyClass === "StreamPreview" || node.type === "StreamPreview";

    const content = document.createElement("div");
    if (isDedicatedPreview) {
        content.style.cssText =
            "flex:1;min-height:0;overflow:hidden;" +
            "background:#1a1a1a;border:1px solid #444;border-radius:4px;" +
            "padding:8px;box-sizing:border-box;font-size:13px;line-height:1.6;" +
            "color:#ddd;white-space:pre-wrap;word-break:break-word;";
    } else {
        content.style.cssText =
            "width:100%;min-height:60px;max-height:480px;overflow-y:auto;" +
            "background:#1a1a1a;border:1px solid #444;border-radius:4px;" +
            "padding:8px;box-sizing:border-box;font-size:13px;line-height:1.6;" +
            "color:#ddd;white-space:pre-wrap;word-break:break-word;";
    }

    const status = document.createElement("div");
    status.style.cssText =
        "font-size:10px;color:#888;text-align:right;min-height:14px;";

    container.appendChild(toolbar);
    container.appendChild(content);
    container.appendChild(status);

    node._spContainer = container;
    node._spContent = content;
    node._spStatus = status;
    node._spMdToggle = mdToggle;
    node._spRawText = "";
    node._spMarkdown = true;

    mdToggle.addEventListener("click", () => {
        node._spMarkdown = !node._spMarkdown;
        mdToggle.style.background = node._spMarkdown ? "#2a5a2a" : "#444";
        renderContent(node);
    });

    copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(node._spRawText).then(() => {
            copyBtn.textContent = "已复制";
            setTimeout(() => { copyBtn.textContent = "复制"; }, 1500);
        });
    });

    const widget = node.addDOMWidget("stream_preview_widget", "preview", container, {
        getValue() { return node._spRawText; },
        setValue(v) { },
    });
    widget.computeSize = (width) => {
        const isDedicatedPreview = node.comfyClass === "StreamPreview" || node.type === "StreamPreview";
        if (isDedicatedPreview) {
            const nodeHeight = node.size?.[1] ?? 320;
            const overhead = 60;
            return [width, Math.max(120, nodeHeight - overhead)];
        }
        return [width, 320];
    };

    loadMarked();
}

async function renderContent(node) {
    const text = node._spRawText;
    const el = node._spContent;
    if (!text) { el.innerHTML = ""; return; }

    if (node._spMarkdown) {
        const marked = await loadMarked();
        if (marked) {
            el.style.whiteSpace = "normal";
            el.innerHTML = marked.parse(text);
        } else {
            el.style.whiteSpace = "pre-wrap";
            el.textContent = text;
        }
    } else {
        el.style.whiteSpace = "pre-wrap";
        el.textContent = text;
    }
    const isPreview = node.comfyClass === "StreamPreview" || node.type === "StreamPreview";
    if (!isPreview) el.scrollTop = el.scrollHeight;
}

// ── 流式事件监听 ──────────────────────────────────────────────────────────────
api.addEventListener("o1key.stream_token", (event) => {
    const { node_id, token, done } = event.detail;
    const node = app.graph.getNodeById(parseInt(node_id));
    if (!node) return;

    buildUI(node);

    if (done) {
        node._spStreaming = false;
        node._spStatus.textContent = "生成完成";
        node._spStatus.style.color = "#4a4";
        return;
    }

    // 第一个 token 到来时清空上一次内容
    if (!node._spStreaming) {
        node._spStreaming = true;
        node._spRawText = "";
    }

    node._spRawText += token;
    node._spStatus.textContent = "生成中…";
    node._spStatus.style.color = "#a84";
    renderContent(node);
});

// ── 节点注册 ──────────────────────────────────────────────────────────────────
app.registerExtension({
    name: "comfyui_o1key.streamPreview",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "StreamPreview") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            buildUI(this);
        };

        nodeType.prototype.onResize = function () {
            this.setDirtyCanvas(true, false);
        };

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (origOnExecuted) origOnExecuted.apply(this, arguments);
            buildUI(this);

            const texts = message?.text;
            if (!texts || texts.length === 0) return;

            this._spRawText = texts[0];
            this._spStatus.textContent = "完成";
            this._spStatus.style.color = "#4a4";
            renderContent(this);
            this.setDirtyCanvas(true, true);
        };

        const origOnSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            if (origOnSerialize) origOnSerialize.apply(this, arguments);
            o.sp_text = this._spRawText || "";
            o.sp_markdown = this._spMarkdown !== false;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (o) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            buildUI(this);
            if (o.sp_text) {
                this._spRawText = o.sp_text;
                this._spMarkdown = o.sp_markdown !== false;
                this._spMdToggle.style.background = this._spMarkdown ? "#2a5a2a" : "#444";
                renderContent(this);
            }
        };
    },
});
