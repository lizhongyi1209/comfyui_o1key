import { app } from "../../../scripts/app.js";

// ─── State ───────────────────────────────────────────────────────────────────
const STORAGE_KEY = "o1key-chat-conversations";
let conversations = [];
let activeConvId = null;
let currentModel = "gpt-5.5";
let isStreaming = false;
let isThinking = false;
let abortController = null;
let pendingFiles = [];
let markedReady = false;
let chatContainer = null;
let showHistory = false;

const MAX_FILES = 4;
const MAX_FILE_SIZE = 20 * 1024 * 1024;
const ACCEPT_STRING = "image/*,video/*,audio/*,.pdf,.txt,.md,.csv,.json,.doc,.docx";

function classifyFile(mimeType) {
    if (mimeType.startsWith("image/")) return "image";
    if (mimeType.startsWith("video/")) return "video";
    if (mimeType.startsWith("audio/")) return "audio";
    return "document";
}

function getFileIcon(category) {
    switch (category) {
        case "video": return "\u{1F3AC}";
        case "audio": return "\u{1F3B5}";
        case "document": return "\u{1F4C4}";
        default: return "\u{1F4CE}";
    }
}

function classifyFileByName(filename) {
    const ext = (filename.split(".").pop() || "").toLowerCase();
    if (["png","jpg","jpeg","gif","webp","svg"].includes(ext)) return "image";
    if (["mp4","webm","mov","avi"].includes(ext)) return "video";
    if (["mp3","wav","ogg","m4a","flac"].includes(ext)) return "audio";
    return "document";
}

const MODELS = [
    "gpt-5.5",
    "gemini-3.1-pro-preview",
    "deepseek-v4-pro",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "gemini-3.5-flash",
    "doubao-seed-2.0-pro",
];

// ─── Persistence ─────────────────────────────────────────────────────────────
function loadConversations() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) conversations = JSON.parse(raw);
    } catch {}
}
function saveConversations() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations)); } catch {}
}
function getActiveConv() {
    return conversations.find(c => c.id === activeConvId) || null;
}
function getMessages() {
    const conv = getActiveConv();
    return conv ? conv.messages : [];
}
function genId() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}
function formatTime(ts) {
    const d = new Date(ts);
    const pad = n => String(n).padStart(2, "0");
    return `${pad(d.getMonth()+1)}/${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// ─── Marked.js lazy load ─────────────────────────────────────────────────────
function ensureMarked() {
    if (markedReady || window.marked) { markedReady = true; return; }
    const s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/marked/marked.min.js";
    s.onload = () => { markedReady = true; };
    document.head.appendChild(s);
}

// ─── Styles ──────────────────────────────────────────────────────────────────
const STYLE_ID = "o1key-chat-styles";
const CSS = `
#o1key-chat-root{display:flex;flex-direction:column;height:100%;min-height:0;color:#ddd;background:var(--comfy-menu-bg,#202020);overflow:hidden;font-family:inherit;position:absolute;inset:0}
#o1key-chat-header{padding:12px 14px 0;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
#o1key-chat-header .chat-title{font-size:14px;font-weight:600;color:#eee;letter-spacing:.3px}
#o1key-chat-header .hdr-btns{display:flex;gap:4px}
#o1key-chat-header .chat-hdr-btn{width:28px;height:28px;display:flex;align-items:center;justify-content:center;background:transparent;color:#888;border:1px solid rgba(255,255,255,.08);border-radius:6px;cursor:pointer;transition:all .15s}
#o1key-chat-header .chat-hdr-btn:hover{color:#ddd;background:rgba(255,255,255,.08);border-color:rgba(255,255,255,.15)}
#o1key-chat-header .chat-hdr-btn.active{color:#ddd;background:rgba(255,255,255,.1);border-color:rgba(255,255,255,.2)}
#o1key-chat-toolbar{padding:10px 14px;display:flex;gap:6px;flex-shrink:0}
#o1key-chat-toolbar select{flex:1;background:rgba(255,255,255,.08);color:#ddd;border:1px solid rgba(255,255,255,.1);border-radius:8px;padding:8px 32px 8px 12px;font-size:13px;font-weight:500;outline:none;cursor:pointer;transition:all .15s;appearance:none;-webkit-appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='7'%3E%3Cpath d='M1 1l5 5 5-5' fill='none' stroke='%23999' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 10px center}
#o1key-chat-toolbar select:hover{border-color:rgba(255,255,255,.22);background:rgba(255,255,255,.1)}
#o1key-chat-toolbar select:focus{border-color:rgba(255,255,255,.3);background:rgba(255,255,255,.1)}
#o1key-chat-toolbar select option{background:#2a2a2a;color:#ddd;padding:8px}
#o1key-chat-messages{flex:1;overflow-y:auto;padding:12px 14px;display:flex;flex-direction:column;gap:2px;scroll-behavior:smooth;min-height:0}
#o1key-chat-messages::-webkit-scrollbar{width:4px}
#o1key-chat-messages::-webkit-scrollbar-track{background:transparent}
#o1key-chat-messages::-webkit-scrollbar-thumb{background:rgba(255,255,255,.08);border-radius:2px}
.o1k-msg-wrap{display:flex;flex-direction:column;gap:2px;position:relative;padding:4px 0}
.o1k-msg-wrap:hover .o1k-msg-actions{opacity:1}
.o1k-msg-wrap.user{align-items:flex-end}
.o1k-msg-wrap.assistant{align-items:flex-start}
.o1k-msg{max-width:90%;padding:10px 13px;border-radius:14px;font-size:13px;line-height:1.6;word-break:break-word;animation:o1k-fade .2s ease}
.o1k-msg-wrap.user .o1k-msg{background:rgba(255,255,255,.1);color:#eee;border-bottom-right-radius:4px}
.o1k-msg-wrap.assistant .o1k-msg{background:transparent;color:#ddd;border:1px solid rgba(255,255,255,.06);border-bottom-left-radius:4px}
.o1k-msg pre{background:rgba(0,0,0,.25);padding:10px 12px;border-radius:8px;overflow-x:auto;margin:8px 0;border:1px solid rgba(255,255,255,.05)}
.o1k-msg code{font-size:12px;font-family:"JetBrains Mono","Fira Code",monospace;color:#ddd}
.o1k-msg p{margin:4px 0}
.o1k-msg a{color:#7eb8f7;text-decoration:none}
.o1k-msg img{max-width:180px;max-height:140px;border-radius:8px;margin:4px 0;cursor:pointer;border:1px solid rgba(255,255,255,.06)}
.o1k-msg-time{font-size:10px;color:#555;padding:0 4px;user-select:none}
.o1k-msg-actions{position:relative;top:auto;opacity:0;transition:opacity .15s;display:flex;gap:2px;padding:2px 4px}
.o1k-msg-wrap.user .o1k-msg-actions{justify-content:flex-end}
.o1k-msg-wrap.assistant .o1k-msg-actions{justify-content:flex-start}
.o1k-msg-actions button{width:22px;height:22px;border:none;border-radius:4px;background:rgba(255,255,255,.06);color:#888;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .12s}
.o1k-msg-actions button:hover{background:rgba(255,255,255,.15);color:#ddd}
.o1k-msg-actions button.o1k-act-del:hover{background:rgba(220,60,60,.6);color:#fff}
.o1k-edit-area{display:flex;flex-direction:column;gap:6px;width:100%}
.o1k-edit-area textarea{width:100%;background:rgba(255,255,255,.06);color:#e0e0e0;border:1px solid rgba(255,255,255,.15);border-radius:8px;padding:8px 10px;font-size:13px;resize:none;outline:none;min-height:40px;max-height:150px;line-height:1.5;font-family:inherit}
.o1k-edit-area textarea:focus{border-color:rgba(255,255,255,.3)}
.o1k-edit-btns{display:flex;gap:6px;justify-content:flex-end}
.o1k-edit-btns button{padding:4px 12px;border:none;border-radius:6px;font-size:12px;cursor:pointer;transition:all .12s}
.o1k-edit-btns .o1k-edit-cancel{background:rgba(255,255,255,.08);color:#aaa}
.o1k-edit-btns .o1k-edit-cancel:hover{background:rgba(255,255,255,.12);color:#ddd}
.o1k-edit-btns .o1k-edit-save{background:rgba(126,184,247,.2);color:#7eb8f7}
.o1k-edit-btns .o1k-edit-save:hover{background:rgba(126,184,247,.35);color:#fff}
.o1k-msg .typing-dot{display:inline-block;width:5px;height:5px;background:#777;border-radius:50%;margin:0 2px;animation:o1k-blink 1.4s infinite}
.o1k-msg .typing-dot:nth-child(2){animation-delay:.2s}
.o1k-msg .typing-dot:nth-child(3){animation-delay:.4s}
@keyframes o1k-blink{0%,80%,100%{opacity:.2}40%{opacity:1}}
@keyframes o1k-fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
.o1k-thinking{display:flex;align-items:center;gap:8px;padding:2px 0}
.o1k-thinking-icon{width:16px;height:16px;border:2px solid rgba(255,255,255,.15);border-top-color:#7eb8f7;border-radius:50%;animation:o1k-spin .8s linear infinite}
.o1k-thinking-text{font-size:12px;color:#888;animation:o1k-pulse 1.5s ease-in-out infinite}
@keyframes o1k-spin{to{transform:rotate(360deg)}}
@keyframes o1k-pulse{0%,100%{opacity:.6}50%{opacity:1}}
.o1k-reasoning{margin:0 0 8px;border:1px solid rgba(255,255,255,.06);border-radius:8px;overflow:hidden}
.o1k-reasoning summary{display:flex;align-items:center;gap:6px;padding:6px 10px;font-size:11px;color:#888;cursor:pointer;user-select:none;list-style:none;transition:color .12s}
.o1k-reasoning summary::-webkit-details-marker{display:none}
.o1k-reasoning summary::before{content:"";display:inline-block;width:0;height:0;border-style:solid;border-width:4px 0 4px 6px;border-color:transparent transparent transparent #666;transition:transform .15s}
.o1k-reasoning[open] summary::before{transform:rotate(90deg)}
.o1k-reasoning summary:hover{color:#bbb}
.o1k-reasoning .o1k-reasoning-body{padding:6px 10px 8px;font-size:12px;color:#777;line-height:1.5;border-top:1px solid rgba(255,255,255,.04);max-height:200px;overflow-y:auto}
.o1k-reasoning .o1k-reasoning-body::-webkit-scrollbar{width:3px}
.o1k-reasoning .o1k-reasoning-body::-webkit-scrollbar-thumb{background:rgba(255,255,255,.08);border-radius:2px}
#o1key-chat-history{flex:1;overflow-y:auto;padding:8px 14px;display:flex;flex-direction:column;gap:4px;min-height:0}
.o1k-conv-item{display:flex;align-items:center;gap:8px;padding:10px 12px;border-radius:8px;cursor:pointer;transition:all .12s;border:1px solid transparent}
.o1k-conv-item:hover{background:rgba(255,255,255,.06);border-color:rgba(255,255,255,.08)}
.o1k-conv-item.active{background:rgba(255,255,255,.08);border-color:rgba(255,255,255,.12)}
.o1k-conv-item .conv-info{flex:1;min-width:0}
.o1k-conv-item .conv-title{font-size:12px;color:#ddd;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.o1k-conv-item .conv-time{font-size:10px;color:#666;margin-top:2px}
.o1k-conv-item .conv-rename,.o1k-conv-item .conv-del{width:22px;height:22px;border:none;border-radius:4px;background:transparent;color:#666;cursor:pointer;display:flex;align-items:center;justify-content:center;opacity:0;transition:all .12s;flex-shrink:0}
.o1k-conv-item:hover .conv-rename,.o1k-conv-item:hover .conv-del{opacity:1}
.o1k-conv-item .conv-rename:hover{background:rgba(126,184,247,.2);color:#7eb8f7}
.o1k-conv-item .conv-del:hover{background:rgba(220,60,60,.6);color:#fff}
#o1key-chat-input-area{flex-shrink:0;padding:10px 14px 14px;background:transparent}
#o1key-chat-previews{display:flex;gap:6px;padding:0 0 8px;flex-wrap:wrap}
#o1key-chat-previews .preview-thumb{position:relative;width:40px;height:40px;border-radius:6px;overflow:hidden;border:1px solid rgba(255,255,255,.1)}
#o1key-chat-previews .preview-thumb img,#o1key-chat-previews .preview-thumb video{width:100%;height:100%;object-fit:cover}
#o1key-chat-previews .preview-thumb .remove-btn{position:absolute;top:-2px;right:-2px;width:14px;height:14px;background:rgba(200,60,60,.85);color:#fff;border:none;border-radius:50%;font-size:9px;cursor:pointer;display:flex;align-items:center;justify-content:center;line-height:1}
#o1key-chat-previews .preview-thumb.video-preview{width:60px}
#o1key-chat-previews .preview-thumb.video-preview::after{content:"";position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.3);pointer-events:none}
#o1key-chat-previews .preview-thumb.video-preview .play-badge{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:16px;height:16px;background:rgba(255,255,255,.85);border-radius:50%;display:flex;align-items:center;justify-content:center;pointer-events:none}
#o1key-chat-previews .preview-thumb.video-preview .play-badge::after{content:"";border-style:solid;border-width:4px 0 4px 7px;border-color:transparent transparent transparent #333;margin-left:1px}
#o1key-chat-previews .preview-thumb.file-preview{display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,.06);width:auto;min-width:40px;padding:0 8px;gap:4px}
#o1key-chat-previews .preview-thumb.file-preview .file-icon{font-size:16px;line-height:1}
#o1key-chat-previews .preview-thumb.file-preview .file-name{font-size:9px;color:#aaa;max-width:60px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.o1k-msg .file-chip{display:inline-flex;align-items:center;gap:4px;padding:4px 8px;border-radius:6px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.08);font-size:11px;color:#bbb;margin:3px 2px}
.o1k-msg .file-chip .chip-icon{font-size:14px}
.o1k-msg .file-chip .chip-name{max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.o1k-msg .video-thumb{position:relative;display:inline-block;max-width:160px;border-radius:8px;overflow:hidden;margin:4px 0;border:1px solid rgba(255,255,255,.08)}
.o1k-msg .video-thumb img{width:100%;display:block;border-radius:8px}
.o1k-msg .video-thumb .play-overlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.25)}
.o1k-msg .video-thumb .play-overlay::after{content:"";border-style:solid;border-width:8px 0 8px 14px;border-color:transparent transparent transparent rgba(255,255,255,.9)}
#o1key-chat-input-box{display:flex;align-items:flex-end;gap:0;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:14px;padding:6px 6px 6px 12px;transition:border-color .2s}
#o1key-chat-input-box:focus-within{border-color:rgba(255,255,255,.22)}
#o1key-chat-input-box textarea{flex:1;background:transparent;color:#e0e0e0;border:none;padding:6px 0;font-size:13px;resize:none;outline:none;min-height:20px;max-height:100px;line-height:1.5}
#o1key-chat-input-box textarea::placeholder{color:#555}
#o1key-chat-input-box button{width:30px;height:30px;border:none;border-radius:8px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .15s;flex-shrink:0;background:transparent;color:#666}
#o1key-chat-input-box button:hover{color:#ccc;background:rgba(255,255,255,.08)}
#o1key-chat-input-box .o1k-btn-send{color:#999;background:rgba(255,255,255,.06)}
#o1key-chat-input-box .o1k-btn-send:hover{color:#fff;background:rgba(255,255,255,.15)}
#o1key-chat-input-box .o1k-btn-stop{color:#fff;background:rgba(220,70,55,.75)}
#o1key-chat-input-box .o1k-btn-stop:hover{background:rgba(220,70,55,.95)}
.o1k-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:#555;font-size:13px;gap:10px;user-select:none}
.o1k-empty svg{width:36px;height:36px;opacity:.2;stroke:#888}
`;

function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const el = document.createElement("style");
    el.id = STYLE_ID;
    el.textContent = CSS;
    document.head.appendChild(el);
}

// ─── Extension Registration ──────────────────────────────────────────────────
app.registerExtension({
    name: "o1key.chatPanel",
    async setup() {
        ensureMarked();
        loadConversations();
        app.extensionManager.registerSidebarTab({
            id: "o1key-chat",
            title: "对话",
            icon: "pi pi-comments",
            type: "custom",
            render: (container) => {
                injectStyles();
                renderChatPanel(container);
            },
        });
    },
});

// ─── Render Chat Panel ───────────────────────────────────────────────────────
function renderChatPanel(container) {
    container.innerHTML = "";
    container.style.position = "relative";
    container.style.height = "100%";
    container.style.overflow = "hidden";
    const root = document.createElement("div");
    root.id = "o1key-chat-root";
    root.innerHTML = `
        <div id="o1key-chat-header">
            <span class="chat-title">AI 对话</span>
            <div class="hdr-btns">
                <button class="chat-hdr-btn" id="o1k-history-btn" title="对话记录">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 8v4l3 3"/><circle cx="12" cy="12" r="9"/></svg>
                </button>
                <button class="chat-hdr-btn" id="o1k-new-chat" title="新对话">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>
                </button>
            </div>
        </div>
        <div id="o1key-chat-toolbar">
            <select id="o1k-model-sel">${MODELS.map(m => `<option value="${m}"${m === currentModel ? " selected" : ""}>${m}${m === "gemini-3.1-pro-preview" ? " (支持视频)" : ""}</option>`).join("")}</select>
        </div>
        <div id="o1key-chat-messages"></div>
        <div id="o1key-chat-history" style="display:none"></div>
        <div id="o1key-chat-input-area">
            <div id="o1key-chat-previews"></div>
            <div id="o1key-chat-input-box">
                <button id="o1k-attach" title="上传文件">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>
                </button>
                <textarea id="o1k-input" rows="1" placeholder="发送消息..."></textarea>
                <button class="o1k-btn-send" id="o1k-send" title="发送">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg>
                </button>
            </div>
        </div>
        <input type="file" id="o1k-file-input" accept="${ACCEPT_STRING}" multiple style="display:none">
    `;
    container.appendChild(root);
    chatContainer = root;
    bindEvents(root);
    renderMessages();
}

// ─── Event Binding ───────────────────────────────────────────────────────────
function bindEvents(root) {
    const modelSel = root.querySelector("#o1k-model-sel");
    const newBtn = root.querySelector("#o1k-new-chat");
    const histBtn = root.querySelector("#o1k-history-btn");
    const input = root.querySelector("#o1k-input");
    const sendBtn = root.querySelector("#o1k-send");
    const attachBtn = root.querySelector("#o1k-attach");
    const fileInput = root.querySelector("#o1k-file-input");

    modelSel.addEventListener("change", () => { currentModel = modelSel.value; });
    newBtn.addEventListener("click", startNewConversation);
    histBtn.addEventListener("click", toggleHistory);

    input.addEventListener("input", () => autoResize(input));
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
    });
    input.addEventListener("paste", (e) => {
        const items = e.clipboardData?.items;
        if (!items) return;
        for (const item of items) {
            if (item.kind === "file") {
                const file = item.getAsFile();
                if (file) { e.preventDefault(); addFile(file); }
            }
        }
    });
    sendBtn.addEventListener("click", handleSend);
    attachBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
        for (const f of fileInput.files) addFile(f);
        fileInput.value = "";
    });
}

// ─── Conversation Management ─────────────────────────────────────────────────
function startNewConversation() {
    if (showHistory) toggleHistory();
    const conv = { id: genId(), title: "新对话", messages: [], createdAt: Date.now(), updatedAt: Date.now() };
    conversations.unshift(conv);
    activeConvId = conv.id;
    saveConversations();
    pendingFiles = [];
    renderMessages();
    renderPreviews();
}

function toggleHistory() {
    showHistory = !showHistory;
    const msgBox = chatContainer.querySelector("#o1key-chat-messages");
    const histBox = chatContainer.querySelector("#o1key-chat-history");
    const histBtn = chatContainer.querySelector("#o1k-history-btn");
    if (showHistory) {
        msgBox.style.display = "none";
        histBox.style.display = "flex";
        histBtn.classList.add("active");
        renderHistory();
    } else {
        msgBox.style.display = "flex";
        histBox.style.display = "none";
        histBtn.classList.remove("active");
    }
}

function renderHistory() {
    const box = chatContainer.querySelector("#o1key-chat-history");
    if (conversations.length === 0) {
        box.innerHTML = `<div class="o1k-empty"><span>暂无对话记录</span></div>`;
        return;
    }
    box.innerHTML = conversations.map(c => {
        const active = c.id === activeConvId ? " active" : "";
        const title = c.title || "新对话";
        const time = formatTime(c.updatedAt || c.createdAt);
        return `<div class="o1k-conv-item${active}" data-id="${c.id}">
            <div class="conv-info"><div class="conv-title">${escapeHtml(title)}</div><div class="conv-time">${time}</div></div>
            <button class="conv-rename" data-id="${c.id}" title="重命名"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg></button>
            <button class="conv-del" data-id="${c.id}" title="删除对话"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg></button>
        </div>`;
    }).join("");
    box.querySelectorAll(".o1k-conv-item").forEach(el => {
        el.addEventListener("click", (e) => {
            if (e.target.closest(".conv-del") || e.target.closest(".conv-rename")) return;
            activeConvId = el.dataset.id;
            showHistory = false;
            const msgBox = chatContainer.querySelector("#o1key-chat-messages");
            const histBox = chatContainer.querySelector("#o1key-chat-history");
            msgBox.style.display = "flex";
            histBox.style.display = "none";
            chatContainer.querySelector("#o1k-history-btn").classList.remove("active");
            renderMessages();
        });
    });
    box.querySelectorAll(".conv-rename").forEach(btn => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            const id = btn.dataset.id;
            const conv = conversations.find(c => c.id === id);
            if (!conv) return;
            const newTitle = prompt("重命名对话", conv.title || "");
            if (newTitle === null || !newTitle.trim()) return;
            conv.title = newTitle.trim();
            conv.updatedAt = Date.now();
            saveConversations();
            renderHistory();
        });
    });
    box.querySelectorAll(".conv-del").forEach(btn => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            if (!confirm("确定删除这个对话？")) return;
            const id = btn.dataset.id;
            conversations = conversations.filter(c => c.id !== id);
            if (activeConvId === id) activeConvId = conversations[0]?.id || null;
            saveConversations();
            renderHistory();
        });
    });
}

function deleteMessage(idx) {
    if (!confirm("确定删除这条消息？")) return;
    const conv = getActiveConv();
    if (!conv) return;
    conv.messages.splice(idx, 1);
    conv.updatedAt = Date.now();
    saveConversations();
    renderMessages();
}

function copyMessage(idx) {
    const conv = getActiveConv();
    if (!conv) return;
    const msg = conv.messages[idx];
    if (!msg) return;
    const text = typeof msg.content === "string" ? msg.content : extractTextFromContent(msg.content);
    navigator.clipboard.writeText(text).then(() => {
        const btn = chatContainer.querySelector(`.o1k-msg-wrap[data-idx="${idx}"] [data-act="copy"]`);
        if (btn) { btn.title = "已复制"; setTimeout(() => { btn.title = "复制"; }, 1500); }
    });
}

function enterEditMode(idx) {
    if (isStreaming) return;
    const conv = getActiveConv();
    if (!conv) return;
    const msg = conv.messages[idx];
    if (!msg || msg.role !== "user") return;
    const text = extractTextFromContent(msg.content);
    const box = chatContainer.querySelector("#o1key-chat-messages");
    const wrap = box.querySelector(`.o1k-msg-wrap[data-idx="${idx}"]`);
    if (!wrap) return;
    const msgEl = wrap.querySelector(".o1k-msg");
    msgEl.innerHTML = `<div class="o1k-edit-area">
        <textarea class="o1k-edit-input">${escapeHtml(text)}</textarea>
        <div class="o1k-edit-btns">
            <button class="o1k-edit-cancel">取消</button>
            <button class="o1k-edit-save">保存并重新生成</button>
        </div>
    </div>`;
    const ta = msgEl.querySelector(".o1k-edit-input");
    ta.style.height = Math.min(ta.scrollHeight, 150) + "px";
    ta.focus();
    ta.setSelectionRange(ta.value.length, ta.value.length);
    msgEl.querySelector(".o1k-edit-cancel").addEventListener("click", () => renderMessages());
    msgEl.querySelector(".o1k-edit-save").addEventListener("click", () => {
        const newText = ta.value.trim();
        if (!newText) return;
        conv.messages.splice(idx);
        conv.updatedAt = Date.now();
        saveConversations();
        sendMessage(newText, []);
    });
}

function retryMessage(idx) {
    if (isStreaming) return;
    const conv = getActiveConv();
    if (!conv) return;
    const msg = conv.messages[idx];
    if (!msg || msg.role !== "user") return;
    const text = extractTextFromContent(msg.content);
    conv.messages.splice(idx);
    conv.updatedAt = Date.now();
    saveConversations();
    sendMessage(text, []);
}

function extractTextFromContent(content) {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
        const textPart = content.find(p => p.type === "text");
        return textPart?.text || "";
    }
    return "";
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
function autoResize(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
}

function captureImageThumb(dataUrl) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            try {
                const maxW = 160;
                const w = Math.min(img.width, maxW);
                const h = Math.round(w * img.height / img.width);
                const canvas = document.createElement("canvas");
                canvas.width = w;
                canvas.height = h;
                canvas.getContext("2d").drawImage(img, 0, 0, w, h);
                resolve(canvas.toDataURL("image/jpeg", 0.5));
            } catch { resolve(null); }
        };
        img.onerror = () => resolve(null);
        img.src = dataUrl;
    });
}

function captureVideoThumb(dataUrl) {
    return new Promise((resolve) => {
        const timeout = setTimeout(() => resolve(null), 5000);
        const video = document.createElement("video");
        video.preload = "auto";
        video.muted = true;
        video.playsInline = true;
        video.crossOrigin = "anonymous";
        const done = () => {
            clearTimeout(timeout);
            try {
                const canvas = document.createElement("canvas");
                const w = Math.min(video.videoWidth || 320, 320);
                const h = video.videoHeight ? Math.round(w * video.videoHeight / video.videoWidth) : 180;
                canvas.width = w;
                canvas.height = h;
                canvas.getContext("2d").drawImage(video, 0, 0, w, h);
                resolve(canvas.toDataURL("image/jpeg", 0.6));
            } catch { resolve(null); }
            video.src = "";
            video.load();
        };
        video.onloadeddata = () => {
            if (video.readyState >= 2) done();
        };
        video.onerror = () => { clearTimeout(timeout); resolve(null); };
        video.src = dataUrl;
        video.load();
    });
}

function addFile(file) {
    if (!file || pendingFiles.length >= MAX_FILES) return;
    if (file.size > MAX_FILE_SIZE) {
        alert(`文件 "${file.name}" 超过 20MB 限制`);
        return;
    }
    const mimeType = file.type || "application/octet-stream";
    const category = classifyFile(mimeType);
    const reader = new FileReader();
    reader.onload = (e) => {
        const entry = { id: genId(), name: file.name, mimeType, category, dataUrl: e.target.result, thumb: null };
        pendingFiles.push(entry);
        renderPreviews();
        if (category === "video") {
            captureVideoThumb(e.target.result).then(thumb => {
                entry.thumb = thumb;
                renderPreviews();
            });
        }
    };
    reader.readAsDataURL(file);
}

function renderPreviews() {
    const box = chatContainer.querySelector("#o1key-chat-previews");
    box.innerHTML = pendingFiles.map((f, i) => {
        if (f.category === "image") {
            return `<div class="preview-thumb"><img src="${f.dataUrl}"><button class="remove-btn" data-idx="${i}">&times;</button></div>`;
        }
        if (f.category === "video") {
            const src = f.thumb || f.dataUrl;
            const tag = f.thumb ? `<img src="${f.thumb}">` : `<video src="${f.dataUrl}" muted preload="metadata"></video>`;
            return `<div class="preview-thumb video-preview">${tag}<span class="play-badge"></span><button class="remove-btn" data-idx="${i}">&times;</button></div>`;
        }
        const icon = getFileIcon(f.category);
        return `<div class="preview-thumb file-preview"><span class="file-icon">${icon}</span><span class="file-name" title="${escapeHtml(f.name)}">${escapeHtml(f.name)}</span><button class="remove-btn" data-idx="${i}">&times;</button></div>`;
    }).join("");
    box.querySelectorAll(".remove-btn").forEach(btn => {
        btn.addEventListener("click", () => { pendingFiles.splice(+btn.dataset.idx, 1); renderPreviews(); });
    });
}

function handleSend() {
    if (isStreaming) { if (abortController) abortController.abort(); return; }
    const input = chatContainer.querySelector("#o1k-input");
    const text = input.value.trim();
    if (!text && pendingFiles.length === 0) return;
    input.value = "";
    autoResize(input);
    const filesToSend = [...pendingFiles];
    pendingFiles = [];
    renderPreviews();
    if (!activeConvId) startNewConversation();
    sendMessage(text, filesToSend);
}

// ─── Send Message (Streaming) ────────────────────────────────────────────────
function stripFileDataForStorage(conv) {
    for (const msg of conv.messages) {
        if (!Array.isArray(msg.content)) continue;
        msg.content = msg.content.map(part => {
            if (part.type === "image_url" && part._video && part.image_url?.url?.startsWith("data:")) {
                return { type: "image_url", _video: true, _stripped: true, _thumb: part._thumb || null, image_url: { url: "[video]" } };
            }
            if (part.type === "image_url" && part.image_url?.url?.startsWith("data:")) {
                return { type: "image_url", image_url: { url: "[image]" }, _stripped: true, _thumb: part._thumb || null };
            }
            if (part.type === "input_audio" && part.input_audio?.data) {
                return { type: "input_audio", input_audio: { format: part.input_audio.format }, _stripped: true };
            }
            if (part.type === "file" && part.file?.file_data) {
                return { type: "file", file: { filename: part.file.filename }, _stripped: true };
            }
            return part;
        });
    }
}

function cleanContentForApi(content) {
    if (typeof content === "string") return content;
    if (!Array.isArray(content)) return content;
    const cleaned = content.filter(p => !p._stripped).map(part => {
        if (part.type === "image_url") {
            return { type: "image_url", image_url: { url: part.image_url.url } };
        }
        if (part.type === "input_audio") {
            return { type: "input_audio", input_audio: part.input_audio };
        }
        if (part.type === "file") {
            return { type: "file", file: part.file };
        }
        if (part.type === "text") {
            return { type: "text", text: part.text };
        }
        return part;
    });
    if (cleaned.length === 0) return "(附件已省略)";
    const textOnly = cleaned.filter(p => p.type === "text");
    if (cleaned.length === textOnly.length && textOnly.length === 1) return textOnly[0].text;
    return cleaned;
}

function buildFileContentPart(file) {
    if (file.category === "image") {
        return { type: "image_url", image_url: { url: file.dataUrl }, _thumb: file.thumb || null };
    }
    if (file.category === "video") {
        return { type: "image_url", image_url: { url: file.dataUrl }, _video: true, _thumb: file.thumb || null };
    }
    if (file.category === "audio") {
        const base64 = file.dataUrl.split(",")[1] || "";
        const ext = (file.name.split(".").pop() || "").toLowerCase();
        const format = ext === "wav" ? "wav" : "mp3";
        return { type: "input_audio", input_audio: { data: base64, format } };
    }
    return { type: "file", file: { filename: file.name, file_data: file.dataUrl } };
}

async function sendMessage(text, files) {
    const conv = getActiveConv();
    if (!conv) return;
    for (const f of files) {
        if (f.category === "video" && !f.thumb && f.dataUrl) {
            f.thumb = await captureVideoThumb(f.dataUrl);
        }
        if (f.category === "image" && !f.thumb && f.dataUrl) {
            f.thumb = await captureImageThumb(f.dataUrl);
        }
    }
    let content;
    if (files.length > 0) {
        content = [];
        for (const f of files) content.push(buildFileContentPart(f));
        if (text) content.push({ type: "text", text });
    } else { content = text; }

    conv.messages.push({ role: "user", content, time: Date.now() });
    conv.messages.push({ role: "assistant", content: "", time: Date.now() });
    if (conv.title === "新对话" && text) conv.title = text.slice(0, 30);
    conv.updatedAt = Date.now();

    const reqMsgs = conv.messages.slice(0, -1).map(m => ({ role: m.role, content: cleanContentForApi(m.content) }));
    stripFileDataForStorage(conv);
    saveConversations();

    const sendBtn = chatContainer.querySelector("#o1k-send");
    sendBtn.classList.remove("o1k-btn-send");
    sendBtn.classList.add("o1k-btn-stop");
    sendBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>`;
    isStreaming = true;
    isThinking = true;
    abortController = new AbortController();
    renderMessages();
    scrollToBottom();

    try {
        const resp = await fetch("/o1key/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: abortController.signal,
            body: JSON.stringify({ model: currentModel, messages: reqMsgs, stream: true }),
        });
        if (!resp.ok) {
            isThinking = false;
            let errMsg = `请求失败 (${resp.status})`;
            try { const d = await resp.json(); if (d.error) errMsg = d.error; } catch {}
            conv.messages[conv.messages.length - 1].content = errMsg;
            saveConversations(); updateLastMessage(); finishStream(); return;
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();
            for (const line of lines) {
                if (!line.startsWith("data:")) continue;
                const data = line.slice(5).trim();
                if (data === "[DONE]") break;
                try {
                    const chunk = JSON.parse(data);
                    if (chunk.error) { conv.messages[conv.messages.length-1].content += `\n\n**错误:** ${chunk.error}`; updateLastMessage(); break; }
                    const delta = chunk.choices?.[0]?.delta;
                    const finish = chunk.choices?.[0]?.finish_reason;
                    const lastMsg = conv.messages[conv.messages.length-1];
                    if (delta?.reasoning_content) {
                        isThinking = false;
                        lastMsg._reasoning = (lastMsg._reasoning || "") + delta.reasoning_content;
                        updateLastMessage();
                    }
                    if (delta?.content) {
                        isThinking = false;
                        lastMsg.content += delta.content;
                        updateLastMessage();
                    }
                    if (finish === "content_filter") {
                        lastMsg.content += "\n\n⚠️ 内容被安全过滤，未能完整输出。";
                        updateLastMessage();
                    }
                } catch {}
            }
        }
    } catch (e) {
        if (e.name !== "AbortError") { conv.messages[conv.messages.length-1].content += `\n\n**请求失败:** ${e.message}`; updateLastMessage(); }
    }
    if (!conv.messages[conv.messages.length-1].content) {
        conv.messages[conv.messages.length-1].content = "（模型未返回任何内容）";
        updateLastMessage();
    }
    conv.messages[conv.messages.length-1].time = Date.now();
    saveConversations();
    finishStream();
}

function finishStream() {
    isStreaming = false; isThinking = false; abortController = null;
    const sendBtn = chatContainer.querySelector("#o1k-send");
    sendBtn.classList.remove("o1k-btn-stop");
    sendBtn.classList.add("o1k-btn-send");
    sendBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg>`;
    renderLastAsMarkdown();
}

// ─── Message Rendering ───────────────────────────────────────────────────────
function renderMessages() {
    const box = chatContainer.querySelector("#o1key-chat-messages");
    const msgs = getMessages();
    if (msgs.length === 0) {
        box.innerHTML = `<div class="o1k-empty"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 21a9 9 0 1 0-9-9c0 1.5.4 2.9 1 4.2L3 21l4.8-1c1.3.6 2.7 1 4.2 1z"/><path d="M8 12h.01M12 12h.01M16 12h.01"/></svg><span>开始新对话</span></div>`;
        return;
    }
    box.innerHTML = msgs.map((msg, i) => {
        const cls = msg.role === "user" ? "user" : "assistant";
        const html = formatMsgContent(msg, i, msgs);
        const time = msg.time ? `<div class="o1k-msg-time">${formatTime(msg.time)}</div>` : "";
        let actions = "";
        if (msg.role === "user") {
            actions = `<div class="o1k-msg-actions">
                <button data-idx="${i}" data-act="edit" title="编辑"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg></button>
                <button data-idx="${i}" data-act="retry" title="重新生成"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 11-2.12-9.36L23 10"/></svg></button>
                <button data-idx="${i}" data-act="del" class="o1k-act-del" title="删除"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg></button>
            </div>`;
        } else {
            actions = `<div class="o1k-msg-actions">
                <button data-idx="${i}" data-act="copy" title="复制"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg></button>
                <button data-idx="${i}" data-act="del" class="o1k-act-del" title="删除"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg></button>
            </div>`;
        }
        return `<div class="o1k-msg-wrap ${cls}" data-idx="${i}"><div class="o1k-msg">${html}</div>${time}${actions}</div>`;
    }).join("");
    box.querySelectorAll(".o1k-msg-actions button").forEach(btn => {
        const idx = +btn.dataset.idx;
        const act = btn.dataset.act;
        btn.addEventListener("click", () => {
            if (act === "del") deleteMessage(idx);
            else if (act === "edit") enterEditMode(idx);
            else if (act === "retry") retryMessage(idx);
            else if (act === "copy") copyMessage(idx);
        });
    });
    scrollToBottom();
}

function formatMsgContent(msg, idx, msgs) {
    const c = msg.content;
    let reasoningHtml = "";
    if (msg.role === "assistant" && msg._reasoning) {
        const reasoningText = isStreaming && idx === msgs.length - 1
            ? escapeHtml(msg._reasoning).replace(/\n/g, "<br>")
            : renderMd(msg._reasoning);
        reasoningHtml = `<details class="o1k-reasoning"><summary>思考过程</summary><div class="o1k-reasoning-body">${reasoningText}</div></details>`;
    }
    if (typeof c === "string") {
        if (msg.role === "assistant" && !isStreaming) return reasoningHtml + renderMd(c);
        if (msg.role === "assistant" && c === "" && idx === msgs.length - 1) {
            if (isThinking) {
                return `<div class="o1k-thinking"><div class="o1k-thinking-icon"></div><span class="o1k-thinking-text">思考中...</span></div>`;
            }
            if (msg._reasoning) {
                return reasoningHtml;
            }
            return `<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>`;
        }
        if (msg.role === "assistant") return reasoningHtml + escapeHtml(c).replace(/\n/g, "<br>");
        return escapeHtml(c).replace(/\n/g, "<br>");
    }
    if (Array.isArray(c)) {
        let html = "";
        for (const part of c) {
            if (part.type === "text") html += `<p>${escapeHtml(part.text)}</p>`;
            else if (part.type === "image_url") {
                if (part._video) {
                    if (part._thumb) {
                        html += `<div class="video-thumb"><img src="${part._thumb}"><div class="play-overlay"></div></div>`;
                    } else {
                        html += `<span class="file-chip"><span class="chip-icon">\u{1F3AC}</span><span class="chip-name">视频</span></span>`;
                    }
                } else if (part._stripped) {
                    if (part._thumb) {
                        html += `<img src="${part._thumb}" loading="lazy">`;
                    } else {
                        html += `<span class="file-chip"><span class="chip-icon">\u{1F5BC}</span><span class="chip-name">图片</span></span>`;
                    }
                } else {
                    html += `<img src="${part.image_url.url}" loading="lazy">`;
                }
            }
            else if (part.type === "video_url") {
                if (part._thumb) {
                    html += `<div class="video-thumb"><img src="${part._thumb}"><div class="play-overlay"></div></div>`;
                } else {
                    html += `<span class="file-chip"><span class="chip-icon">\u{1F3AC}</span><span class="chip-name">视频</span></span>`;
                }
            }
            else if (part.type === "input_audio") {
                html += `<span class="file-chip"><span class="chip-icon">\u{1F3B5}</span><span class="chip-name">音频</span></span>`;
            }
            else if (part.type === "file" && part.file) {
                const name = part.file.filename || "file";
                const cat = classifyFileByName(name);
                const icon = getFileIcon(cat);
                html += `<span class="file-chip"><span class="chip-icon">${icon}</span><span class="chip-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span></span>`;
            }
        }
        return html;
    }
    return "";
}

function updateLastMessage() {
    const box = chatContainer.querySelector("#o1key-chat-messages");
    const msgs = getMessages();
    const last = box.querySelector(`.o1k-msg-wrap[data-idx="${msgs.length - 1}"] .o1k-msg`);
    if (last) last.innerHTML = escapeHtml(msgs[msgs.length - 1].content).replace(/\n/g, "<br>");
    scrollToBottom();
}

function renderLastAsMarkdown() {
    const box = chatContainer.querySelector("#o1key-chat-messages");
    const msgs = getMessages();
    const last = box.querySelector(`.o1k-msg-wrap[data-idx="${msgs.length - 1}"] .o1k-msg`);
    if (last && msgs[msgs.length - 1]?.role === "assistant") {
        last.innerHTML = renderMd(msgs[msgs.length - 1].content);
    }
}

function scrollToBottom() {
    const box = chatContainer.querySelector("#o1key-chat-messages");
    if (box) box.scrollTop = box.scrollHeight;
}

function renderMd(text) {
    if (!text) return "";
    if (window.marked) { try { return window.marked.parse(text); } catch {} }
    return escapeHtml(text).replace(/\n/g, "<br>");
}

function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}