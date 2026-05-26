import { app } from "../../../scripts/app.js";

// ─── State ───────────────────────────────────────────────────────────────────
const STORAGE_KEY = "o1key-chat-conversations";
let conversations = [];
let activeConvId = null;
let currentModel = "gpt-5.5";
let currentReasoning = "medium";
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

function truncateFilename(name, maxLen = 20) {
    if (!name || name.length <= maxLen) return name || "";
    const ext = name.lastIndexOf(".") > 0 ? name.slice(name.lastIndexOf(".")) : "";
    const base = name.slice(0, name.length - ext.length);
    const keep = maxLen - ext.length - 2;
    if (keep <= 3) return name.slice(0, maxLen - 2) + "..." + ext;
    return base.slice(0, keep) + "..." + ext;
}

const MODELS = [
    "gpt-5.5",
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "deepseek-v4-pro",
    "claude-opus-4-7",
    "claude-opus-4-6",
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

function formatDuration(seconds) {
    const s = Math.round(seconds);
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return m > 0 ? `${m}:${String(sec).padStart(2, "0")}` : `0:${String(sec).padStart(2, "0")}`;
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
#o1key-chat-toolbar select:hover{border-color:rgba(255,255,255,.22);background-color:rgba(255,255,255,.1)}
#o1key-chat-toolbar select:focus{border-color:rgba(255,255,255,.3);background-color:rgba(255,255,255,.1)}
#o1key-chat-toolbar #o1k-reasoning-sel{flex:none;width:auto;padding-right:28px}
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
.o1k-msg-model{font-size:10px;color:#666;padding:0 4px 2px;user-select:none;font-weight:500}
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
.o1k-conv-item .conv-rename-input{flex:1;min-width:0;padding:2px 6px;border:1px solid rgba(126,184,247,.5);border-radius:4px;background:rgba(0,0,0,.3);color:#ddd;font-size:12px;outline:none}
.o1k-conv-item .conv-rename-input:focus{border-color:#7eb8f7}
.o1k-conv-confirm{display:flex;align-items:center;gap:6px;flex:1;min-width:0}
.o1k-conv-confirm span{font-size:11px;color:#ccc;white-space:nowrap}
.o1k-conv-confirm button{padding:2px 8px;border:none;border-radius:4px;font-size:11px;cursor:pointer;transition:all .12s}
.o1k-conv-confirm .confirm-yes{background:rgba(220,60,60,.7);color:#fff}
.o1k-conv-confirm .confirm-yes:hover{background:rgba(220,60,60,.9)}
.o1k-conv-confirm .confirm-no{background:rgba(255,255,255,.1);color:#aaa}
.o1k-conv-confirm .confirm-no:hover{background:rgba(255,255,255,.15);color:#ddd}
.o1k-msg-del-confirm{display:flex;align-items:center;gap:6px;padding:4px 8px;margin-top:4px;border-radius:6px;background:rgba(220,60,60,.08);border:1px solid rgba(220,60,60,.2)}
.o1k-msg-del-confirm span{font-size:11px;color:#ccc}
.o1k-msg-del-confirm button{padding:2px 8px;border:none;border-radius:4px;font-size:11px;cursor:pointer;transition:all .12s}
.o1k-msg-del-confirm .confirm-yes{background:rgba(220,60,60,.7);color:#fff}
.o1k-msg-del-confirm .confirm-yes:hover{background:rgba(220,60,60,.9)}
.o1k-msg-del-confirm .confirm-no{background:rgba(255,255,255,.1);color:#aaa}
.o1k-msg-del-confirm .confirm-no:hover{background:rgba(255,255,255,.15);color:#ddd}
#o1key-chat-input-area{flex-shrink:0;padding:10px 14px 14px;background:transparent}
#o1key-chat-previews{display:flex;gap:6px;padding:0 0 8px;flex-wrap:wrap}
#o1key-chat-previews .preview-thumb{position:relative;width:40px;height:40px;border-radius:6px;overflow:hidden;border:1px solid rgba(255,255,255,.1)}
#o1key-chat-previews .preview-thumb img,#o1key-chat-previews .preview-thumb video{width:100%;height:100%;object-fit:cover}
#o1key-chat-previews .preview-thumb .remove-btn{position:absolute;top:-2px;right:-2px;width:14px;height:14px;background:rgba(200,60,60,.85);color:#fff;border:none;border-radius:50%;font-size:9px;cursor:pointer;display:flex;align-items:center;justify-content:center;line-height:1}
#o1key-chat-previews .preview-thumb.video-preview{width:auto;height:auto;max-height:none}
#o1key-chat-previews .preview-thumb.video-preview img,#o1key-chat-previews .preview-thumb.video-preview video{width:auto;height:50px;max-height:none;object-fit:contain}
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
.o1k-msg .video-thumb img{width:100%;max-width:none;max-height:none;display:block;border:none;margin:0;border-radius:0}
.o1k-msg .video-thumb .play-overlay{position:absolute;inset:0;bottom:24px;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.25)}
.o1k-msg .video-thumb .play-overlay::after{content:"";border-style:solid;border-width:8px 0 8px 14px;border-color:transparent transparent transparent rgba(255,255,255,.9)}
.o1k-msg .video-thumb .video-duration{position:absolute;bottom:28px;right:4px;padding:1px 5px;font-size:10px;color:#fff;background:rgba(0,0,0,.7);border-radius:3px;line-height:1.4}
.o1k-msg .video-thumb .video-filename{position:relative;padding:3px 6px;font-size:10px;color:#bbb;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;background:rgba(0,0,0,.45);border-radius:0 0 8px 8px}
.o1k-msg .video-thumb.video-thumb-placeholder{width:120px;height:80px;background:rgba(255,255,255,.06);display:flex;flex-direction:column;align-items:center;justify-content:center}
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
            <select id="o1k-model-sel">${MODELS.map(m => `<option value="${m}"${m === currentModel ? " selected" : ""}>${m}</option>`).join("")}</select>
            <select id="o1k-reasoning-sel"><option value="low"${currentReasoning === "low" ? " selected" : ""}>思考:低</option><option value="medium"${currentReasoning === "medium" ? " selected" : ""}>思考:中</option><option value="high"${currentReasoning === "high" ? " selected" : ""}>思考:高</option></select>
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
    const reasoningSel = root.querySelector("#o1k-reasoning-sel");
    const newBtn = root.querySelector("#o1k-new-chat");
    const histBtn = root.querySelector("#o1k-history-btn");
    const input = root.querySelector("#o1k-input");
    const sendBtn = root.querySelector("#o1k-send");
    const attachBtn = root.querySelector("#o1k-attach");
    const fileInput = root.querySelector("#o1k-file-input");

    modelSel.addEventListener("change", () => { currentModel = modelSel.value; });
    reasoningSel.addEventListener("change", () => { currentReasoning = reasoningSel.value; });
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
            const item = btn.closest(".o1k-conv-item");
            const infoEl = item.querySelector(".conv-info");
            const oldHtml = infoEl.innerHTML;
            infoEl.innerHTML = `<input class="conv-rename-input" value="${escapeHtml(conv.title || "")}" />`;
            const input = infoEl.querySelector(".conv-rename-input");
            input.focus();
            input.select();
            const commit = () => {
                const val = input.value.trim();
                if (val && val !== conv.title) {
                    conv.title = val;
                    conv.updatedAt = Date.now();
                    saveConversations();
                }
                renderHistory();
            };
            input.addEventListener("keydown", (ev) => {
                if (ev.key === "Enter") { ev.preventDefault(); commit(); }
                if (ev.key === "Escape") { ev.preventDefault(); renderHistory(); }
            });
            input.addEventListener("blur", commit);
        });
    });
    box.querySelectorAll(".conv-del").forEach(btn => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            const id = btn.dataset.id;
            const item = btn.closest(".o1k-conv-item");
            const infoEl = item.querySelector(".conv-info");
            infoEl.innerHTML = `<div class="o1k-conv-confirm"><span>确定删除？</span><button class="confirm-yes">删除</button><button class="confirm-no">取消</button></div>`;
            item.querySelector(".conv-rename").style.display = "none";
            btn.style.display = "none";
            infoEl.querySelector(".confirm-yes").addEventListener("click", (ev) => {
                ev.stopPropagation();
                conversations = conversations.filter(c => c.id !== id);
                if (activeConvId === id) activeConvId = conversations[0]?.id || null;
                saveConversations();
                renderHistory();
            });
            infoEl.querySelector(".confirm-no").addEventListener("click", (ev) => {
                ev.stopPropagation();
                renderHistory();
            });
        });
    });
}

function deleteMessage(idx) {
    const wrap = chatContainer.querySelector(`.o1k-msg-wrap[data-idx="${idx}"]`);
    if (!wrap || wrap.querySelector(".o1k-msg-del-confirm")) return;
    const confirmEl = document.createElement("div");
    confirmEl.className = "o1k-msg-del-confirm";
    confirmEl.innerHTML = `<span>确定删除？</span><button class="confirm-yes">删除</button><button class="confirm-no">取消</button>`;
    wrap.appendChild(confirmEl);
    confirmEl.querySelector(".confirm-yes").addEventListener("click", () => {
        const conv = getActiveConv();
        if (!conv) return;
        conv.messages.splice(idx, 1);
        conv.updatedAt = Date.now();
        saveConversations();
        renderMessages();
    });
    confirmEl.querySelector(".confirm-no").addEventListener("click", () => {
        confirmEl.remove();
    });
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
        const originalFiles = extractFilesFromContent(msg.content);
        conv.messages.splice(idx);
        conv.updatedAt = Date.now();
        saveConversations();
        sendMessage(newText, originalFiles);
    });
}

function retryMessage(idx) {
    if (isStreaming) return;
    const conv = getActiveConv();
    if (!conv) return;
    const msg = conv.messages[idx];
    if (!msg || msg.role !== "user") return;
    const originalContent = msg.content;
    conv.messages.splice(idx);
    conv.updatedAt = Date.now();
    saveConversations();
    sendMessage(null, [], originalContent);
}

function extractTextFromContent(content) {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
        const textPart = content.find(p => p.type === "text");
        return textPart?.text || "";
    }
    return "";
}

function extractFilesFromContent(content) {
    if (!Array.isArray(content)) return [];
    const files = [];
    for (const part of content) {
        if (part._stripped) continue;
        if (part.type === "image_url" && part.image_url) {
            const url = part.image_url.url;
            if (!url || url === "[image]" || url === "[video]") continue;
            files.push({ category: part._video ? "video" : "image", dataUrl: url, thumb: part._thumb || null, name: part._name || null });
        } else if (part.type === "input_audio" && part.input_audio?.data) {
            files.push({ category: "audio", dataUrl: "data:audio/" + (part.input_audio.format || "mp3") + ";base64," + part.input_audio.data, name: "audio." + (part.input_audio.format || "mp3") });
        } else if (part.type === "file" && part.file?.file_data) {
            files.push({ category: "file", dataUrl: part.file.file_data, name: part.file.filename });
        }
    }
    return files;
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

function compressImageForApi(dataUrl) {
    const MAX_DIM = 1568;
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            try {
                let w = img.width, h = img.height;
                const maxDim = Math.max(w, h);
                if (maxDim > MAX_DIM) {
                    const scale = MAX_DIM / maxDim;
                    w = Math.round(w * scale);
                    h = Math.round(h * scale);
                }
                const canvas = document.createElement("canvas");
                canvas.width = w;
                canvas.height = h;
                canvas.getContext("2d").drawImage(img, 0, 0, w, h);
                resolve(canvas.toDataURL("image/jpeg", 0.85));
            } catch { resolve(dataUrl); }
        };
        img.onerror = () => resolve(dataUrl);
        img.src = dataUrl;
    });
}

function extractVideoFrames(dataUrl, fileName) {
    const MAX_FRAME_DIM = 768;
    const TIMEOUT_MS = 30000;
    return new Promise(async (resolve) => {
        // Convert data URL to Blob and upload to server for same-origin playback
        let videoSrc = null;
        try {
            const resp = await fetch(dataUrl);
            const blob = await resp.blob();
            const formData = new FormData();
            formData.append("image", blob, fileName || "video.mp4");
            formData.append("type", "temp");
            formData.append("overwrite", "true");
            const uploadResp = await fetch("/upload/image", { method: "POST", body: formData });
            if (uploadResp.ok) {
                const data = await uploadResp.json();
                videoSrc = `/view?filename=${encodeURIComponent(data.name)}&type=temp${data.subfolder ? "&subfolder=" + encodeURIComponent(data.subfolder) : ""}`;
            }
        } catch (e) { console.warn("[VideoFrames] upload error:", e); }
        if (!videoSrc) { resolve(null); return; }

        const timeout = setTimeout(() => {
            console.warn("[VideoFrames] timeout");
            cleanup();
            resolve(null);
        }, TIMEOUT_MS);
        const video = document.createElement("video");
        video.preload = "auto";
        video.muted = true;
        video.playsInline = true;
        const cleanup = () => {
            clearTimeout(timeout);
            video.onloadedmetadata = null;
            video.onseeked = null;
            video.onerror = null;
            video.src = "";
            video.load();
        };
        video.onerror = () => { cleanup(); resolve(null); };
        video.onloadedmetadata = () => {
            const duration = video.duration;
            if (!duration || !isFinite(duration)) { cleanup(); resolve(null); return; }
            if (duration > 300) { cleanup(); resolve(null); return; }
            let interval;
            if (duration <= 20) interval = 0.25;
            else if (duration <= 60) interval = 1;
            else if (duration <= 120) interval = 2;
            else interval = 5;
            const times = [];
            for (let t = 0; t < duration; t += interval) times.push(t);
            if (times.length === 0) times.push(0);
            if (times.length > 60) times.length = 60;
            const frames = [];
            let idx = 0;
            const captureNext = () => {
                if (idx >= times.length) {
                    cleanup();
                    resolve({ frames, duration, count: frames.length });
                    return;
                }
                video.currentTime = times[idx];
            };
            video.onseeked = () => {
                try {
                    const vw = video.videoWidth || 640;
                    const vh = video.videoHeight || 360;
                    let w = vw, h = vh;
                    const maxDim = Math.max(w, h);
                    if (maxDim > MAX_FRAME_DIM) {
                        const scale = MAX_FRAME_DIM / maxDim;
                        w = Math.round(w * scale);
                        h = Math.round(h * scale);
                    }
                    const canvas = document.createElement("canvas");
                    canvas.width = w;
                    canvas.height = h;
                    canvas.getContext("2d").drawImage(video, 0, 0, w, h);
                    const frameDataUrl = canvas.toDataURL("image/jpeg", 0.7);
                    frames.push(frameDataUrl);
                } catch (e) {
                    console.warn("[VideoFrames] frame error:", e);
                }
                idx++;
                captureNext();
            };
            captureNext();
        };
        video.src = videoSrc;
        video.load();
    });
}

function captureVideoThumb(source) {
    console.log("[VideoThumb] start, source type:", source instanceof File ? "File" : source instanceof Blob ? "Blob" : "string", source instanceof File ? source.name : "");
    return new Promise(async (resolve) => {
        let serverFile = null;
        // Upload to ComfyUI server to get a same-origin URL (bypass CSP)
        if (source instanceof File || source instanceof Blob) {
            try {
                const formData = new FormData();
                formData.append("image", source, source.name || "video.mp4");
                formData.append("type", "temp");
                formData.append("overwrite", "true");
                const resp = await fetch("/upload/image", { method: "POST", body: formData });
                if (resp.ok) {
                    const data = await resp.json();
                    serverFile = `/view?filename=${encodeURIComponent(data.name)}&type=temp${data.subfolder ? "&subfolder=" + encodeURIComponent(data.subfolder) : ""}`;
                    console.log("[VideoThumb] uploaded, server URL:", serverFile);
                } else {
                    console.warn("[VideoThumb] upload failed:", resp.status);
                }
            } catch (e) { console.warn("[VideoThumb] upload error:", e); }
        }
        const videoSrc = serverFile || (typeof source === "string" ? source : null);
        if (!videoSrc) { resolve(null); return; }

        const timeout = setTimeout(() => { console.warn("[VideoThumb] timeout after 8s"); cleanup(); resolve(null); }, 8000);
        const video = document.createElement("video");
        video.preload = "auto";
        video.muted = true;
        video.playsInline = true;
        const cleanup = () => {
            clearTimeout(timeout);
            video.onloadedmetadata = null;
            video.onseeked = null;
            video.onerror = null;
            video.src = "";
            video.load();
        };
        const capture = () => {
            console.log("[VideoThumb] capture: videoWidth=", video.videoWidth, "videoHeight=", video.videoHeight, "currentTime=", video.currentTime);
            try {
                const vw = video.videoWidth || 320;
                const vh = video.videoHeight || 180;
                const w = Math.min(vw, 320);
                const h = Math.round(w * vh / vw);
                const canvas = document.createElement("canvas");
                canvas.width = w;
                canvas.height = h;
                canvas.getContext("2d").drawImage(video, 0, 0, w, h);
                const result = canvas.toDataURL("image/jpeg", 0.6);
                console.log("[VideoThumb] success, canvas:", w, "x", h);
                cleanup();
                resolve(result);
            } catch (e) { console.error("[VideoThumb] capture error:", e); cleanup(); resolve(null); }
        };
        video.onloadedmetadata = () => {
            console.log("[VideoThumb] loadedmetadata: duration=", video.duration, "size=", video.videoWidth, "x", video.videoHeight);
            const seekTo = Math.min(video.duration * 0.1, 0.5);
            video.currentTime = seekTo > 0 ? seekTo : 0.01;
        };
        video.onseeked = () => { console.log("[VideoThumb] seeked"); capture(); };
        video.onerror = (e) => { console.error("[VideoThumb] video error:", video.error?.message, video.error?.code); cleanup(); resolve(null); };
        video.src = videoSrc;
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
            captureVideoThumb(file).then(thumb => {
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
    let lastUserIdx = -1;
    for (let i = conv.messages.length - 1; i >= 0; i--) {
        if (conv.messages[i].role === "user" && Array.isArray(conv.messages[i].content)) {
            lastUserIdx = i; break;
        }
    }
    for (let i = 0; i < conv.messages.length; i++) {
        const msg = conv.messages[i];
        if (!Array.isArray(msg.content)) continue;
        if (i === lastUserIdx) continue;
        msg.content = msg.content.map(part => {
            if (part.type === "image_url" && part._video && part.image_url?.url?.startsWith("data:")) {
                return { type: "image_url", _video: true, _stripped: true, _thumb: part._thumb || null, _name: part._name || null, _duration: part._duration || null, image_url: { url: "[video]" } };
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
        return { type: "image_url", image_url: { url: file.dataUrl }, _video: true, _thumb: file.thumb || null, _name: file.name || null };
    }
    if (file.category === "audio") {
        const base64 = file.dataUrl.split(",")[1] || "";
        const ext = (file.name.split(".").pop() || "").toLowerCase();
        const format = ext === "wav" ? "wav" : "mp3";
        return { type: "input_audio", input_audio: { data: base64, format } };
    }
    return { type: "file", file: { filename: file.name, file_data: file.dataUrl } };
}

async function sendMessage(text, files, rawContent) {
    const conv = getActiveConv();
    if (!conv) return;
    let content;
    if (rawContent !== undefined) {
        content = rawContent;
    } else {
        for (const f of files) {
            if (f.category === "video" && !f.thumb && f.dataUrl) {
                f.thumb = await captureVideoThumb(f.dataUrl);
            }
            if (f.category === "image" && !f.thumb && f.dataUrl) {
                f.thumb = await captureImageThumb(f.dataUrl);
            }
        }
        if (files.length > 0) {
            content = [];
            for (const f of files) {
                if (f.category === "video") {
                    const result = await extractVideoFrames(f.dataUrl, f.name);
                    if (result && result.frames.length > 0) {
                        content.push({ type: "text", text: `[视频: ${f.name || "video"}, 时长${Math.round(result.duration)}秒, ${result.count}帧]`, _hidden: true });
                        const firstFrameIdx = content.length;
                        for (let fi = 0; fi < result.frames.length; fi++) {
                            content.push({ type: "image_url", image_url: { url: result.frames[fi], detail: "low" }, _hidden: fi > 0 });
                        }
                        content[firstFrameIdx]._video = true;
                        content[firstFrameIdx]._thumb = f.thumb || null;
                        content[firstFrameIdx]._name = f.name || null;
                        content[firstFrameIdx]._videoFrames = true;
                        content[firstFrameIdx]._duration = result.duration;
                        content[firstFrameIdx]._hidden = false;
                    } else {
                        content.push(buildFileContentPart(f));
                    }
                } else {
                    content.push(buildFileContentPart(f));
                }
            }
            if (text) content.push({ type: "text", text });
        } else { content = text; }
    }

    const title = typeof content === "string" ? content : extractTextFromContent(content);

    conv.messages.push({ role: "user", content, time: Date.now() });
    conv.messages.push({ role: "assistant", content: "", time: Date.now(), model: currentModel });
    if (conv.title === "新对话" && title) conv.title = title.slice(0, 30);
    conv.updatedAt = Date.now();

    const allMsgs = conv.messages.slice(0, -1).map(m => ({ role: m.role, content: cleanContentForApi(m.content) }));
    // 过滤掉失败的对话对（assistant 为空响应的及其前面的 user 消息）
    const reqMsgs = [];
    for (let i = 0; i < allMsgs.length; i++) {
        if (allMsgs[i].role === "assistant" && allMsgs[i].content === "（模型未返回任何内容）") {
            reqMsgs.pop(); // 移除前面对应的 user 消息
            continue;
        }
        reqMsgs.push(allMsgs[i]);
    }
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
            body: JSON.stringify({ model: currentModel, messages: reqMsgs, stream: true, reasoning_effort: currentReasoning }),
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
        const modelLabel = msg.role === "assistant" && msg.model ? `<div class="o1k-msg-model">${escapeHtml(msg.model)}</div>` : "";
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
        return `<div class="o1k-msg-wrap ${cls}" data-idx="${i}">${modelLabel}<div class="o1k-msg">${html}</div>${time}${actions}</div>`;
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
        const isCurrentStreaming = isStreaming && idx === msgs.length - 1;
        const reasoningText = isCurrentStreaming
            ? escapeHtml(msg._reasoning).replace(/\n/g, "<br>")
            : renderMd(msg._reasoning);
        const openAttr = isCurrentStreaming ? " open" : "";
        reasoningHtml = `<details class="o1k-reasoning"${openAttr}><summary>思考过程</summary><div class="o1k-reasoning-body">${reasoningText}</div></details>`;
    }
    if (typeof c === "string") {
        if (msg.role === "assistant" && !isStreaming) return reasoningHtml + renderMd(c);
        if (msg.role === "assistant" && c === "" && idx === msgs.length - 1) {
            if (msg._reasoning) {
                return reasoningHtml;
            }
            if (isThinking) {
                return `<div class="o1k-thinking"><div class="o1k-thinking-icon"></div><span class="o1k-thinking-text">思考中...</span></div>`;
            }
            return `<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>`;
        }
        if (msg.role === "assistant") return reasoningHtml + escapeHtml(c).replace(/\n/g, "<br>");
        return escapeHtml(c).replace(/\n/g, "<br>");
    }
    if (Array.isArray(c)) {
        let html = "";
        for (const part of c) {
            if (part._hidden) continue;
            if (part.type === "text") html += `<p>${escapeHtml(part.text)}</p>`;
            else if (part.type === "image_url") {
                if (part._video) {
                    const vName = part._name ? truncateFilename(part._name) : "";
                    const dur = part._duration ? formatDuration(part._duration) : "";
                    const durBadge = dur ? `<span class="video-duration">${dur}</span>` : "";
                    if (part._thumb) {
                        html += `<div class="video-thumb"><img src="${part._thumb}"><div class="play-overlay"></div>${durBadge}${vName ? `<div class="video-filename" title="${escapeHtml(part._name)}">${escapeHtml(vName)}</div>` : ""}</div>`;
                    } else {
                        html += `<div class="video-thumb video-thumb-placeholder"><div class="play-overlay"></div>${durBadge}${vName ? `<div class="video-filename" title="${escapeHtml(part._name)}">${escapeHtml(vName)}</div>` : `<span class="file-chip"><span class="chip-icon">\u{1F3AC}</span><span class="chip-name">视频</span></span>`}</div>`;
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
                const vName = part._name ? truncateFilename(part._name) : "";
                if (part._thumb) {
                    html += `<div class="video-thumb"><img src="${part._thumb}"><div class="play-overlay"></div>${vName ? `<div class="video-filename" title="${escapeHtml(part._name)}">${escapeHtml(vName)}</div>` : ""}</div>`;
                } else {
                    html += `<div class="video-thumb video-thumb-placeholder"><div class="play-overlay"></div>${vName ? `<div class="video-filename" title="${escapeHtml(part._name)}">${escapeHtml(vName)}</div>` : `<span class="file-chip"><span class="chip-icon">\u{1F3AC}</span><span class="chip-name">视频</span></span>`}</div>`;
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
    const msg = msgs[msgs.length - 1];
    const last = box.querySelector(`.o1k-msg-wrap[data-idx="${msgs.length - 1}"] .o1k-msg`);
    if (!last) return;
    // 增量更新 reasoning body，避免整体重渲染导致抖动
    if (msg._reasoning && msg.content === "") {
        const body = last.querySelector(".o1k-reasoning-body");
        if (body) {
            body.innerHTML = escapeHtml(msg._reasoning).replace(/\n/g, "<br>");
            scrollToBottom();
            return;
        }
    }
    last.innerHTML = formatMsgContent(msg, msgs.length - 1, msgs);
    scrollToBottom();
}

function renderLastAsMarkdown() {
    const box = chatContainer.querySelector("#o1key-chat-messages");
    const msgs = getMessages();
    const msg = msgs[msgs.length - 1];
    const last = box.querySelector(`.o1k-msg-wrap[data-idx="${msgs.length - 1}"] .o1k-msg`);
    if (last && msg?.role === "assistant") {
        last.innerHTML = formatMsgContent(msg, msgs.length - 1, msgs);
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