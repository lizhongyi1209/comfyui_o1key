import { app } from "../../../scripts/app.js";

const STORAGE_KEY = "o1key-notes";
const SEEDED_KEY = "o1key-notes-seeded-v2";
const STYLE_ID = "o1key-notes-styles";

let notes = [];
let searchText = "";
let activeFilter = "all";
let noteContainer = null;
let editingNoteId = null;
let draftNote = null;
let pendingDelete = false;

const SAMPLE_NOTES = [
    {
        title: "产品主图：高级玻璃质感",
        tags: ["产品图", "玻璃", "灯光"],
        content: `Clean studio lighting, translucent glass material, subtle caustics, soft shadow, premium product photography, 85mm lens, minimal background, high detail.

使用方式：
1. 把产品图作为参考图输入
2. 保留主体轮廓，只调整材质和灯光
3. 如果玻璃过亮，降低 "caustics" 权重`
    },
    {
        title: "Nano Banana 参考图经验",
        tags: ["Nano Banana", "参考图"],
        content: `参考图越多越容易跑偏，主体一致性优先用 1-3 张图。

复杂场景建议分两步：
1. 先生成主体和构图
2. 再用局部或参考图做材质、背景、文字等精修

如果提示词和参考图冲突，模型通常会优先参考图。`
    },
    {
        title: "电商模特换装模板",
        tags: ["电商", "模特", "换装"],
        content: `Keep face identity and original pose. Replace the outfit with: [服装描述].
Realistic fabric texture, natural folds, accurate seams, studio e-commerce photography, clean background, consistent lighting.

Avoid changing body shape, face, hairstyle, camera angle, or hand position.`
    },
    {
        title: "常用负向词",
        tags: ["负向词", "通用"],
        content: "low quality, blurry, deformed hands, extra fingers, bad anatomy, distorted text, watermark, logo, oversaturated, plastic skin, broken geometry"
    },
    {
        title: "图片批量命名经验",
        tags: ["批量", "工作流"],
        content: `批量生图前先确认保存路径和命名规则。

推荐流程：
1. 小批量跑 2-3 张确认风格
2. 固定提示词和参考图
3. 再放大批量数量

这样失败成本最低，也更容易定位是哪一环导致跑偏。`
    }
];

const CSS = `
#o1key-notes-root{position:absolute;inset:0;display:flex;flex-direction:column;min-height:0;color:#ddd;background:var(--comfy-menu-bg,#202020);overflow:hidden;font-family:inherit;--o1n-blue:#4f8cff;--o1n-blue-2:#7eb8f7;--o1n-blue-soft:rgba(79,140,255,.14)}
#o1key-notes-header{padding:12px 14px 10px;border-bottom:1px solid rgba(255,255,255,.06);flex-shrink:0}
.o1n-title-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.o1n-title{font-size:14px;font-weight:700;color:#eee;letter-spacing:.2px}
.o1n-head-actions{display:flex;gap:5px}
.o1n-icon-btn{width:28px;height:28px;border:1px solid rgba(255,255,255,.09);border-radius:7px;background:rgba(255,255,255,.035);color:#888;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:all .15s;flex-shrink:0}
.o1n-icon-btn:hover{color:#ddd;background:rgba(255,255,255,.08);border-color:rgba(255,255,255,.16)}
.o1n-icon-btn.primary{background:var(--o1n-blue);color:#fff;border-color:transparent}
.o1n-icon-btn.primary:hover{background:#6ca0ff;color:#fff}
#o1n-search-box{height:34px;border:1px solid rgba(255,255,255,.1);border-radius:8px;background:rgba(255,255,255,.045);display:flex;align-items:center;gap:8px;padding:0 10px;transition:border-color .15s}
#o1n-search-box:focus-within{border-color:rgba(79,140,255,.42)}
#o1n-search{flex:1;background:transparent;border:0;outline:0;color:#ddd;font-size:12px;min-width:0}
#o1n-search::placeholder{color:#666}
#o1n-panel-status{height:16px;margin-top:7px;color:#777;font-size:11px;line-height:16px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.o1n-tag-strip{display:flex;gap:6px;margin-top:10px;overflow-x:auto;padding-bottom:1px}
.o1n-tag-strip::-webkit-scrollbar{display:none}
.o1n-tag-filter{height:26px;padding:0 9px;border-radius:7px;border:1px solid rgba(255,255,255,.09);background:transparent;color:#929292;font-size:12px;display:flex;align-items:center;gap:5px;cursor:pointer;white-space:nowrap;transition:all .15s}
.o1n-tag-filter:hover{color:#cfcfcf;border-color:rgba(255,255,255,.16)}
.o1n-tag-filter.active{color:var(--o1n-blue-2);border-color:rgba(79,140,255,.36);background:var(--o1n-blue-soft)}
#o1key-notes-list{flex:1;min-height:0;overflow:auto;padding:8px 10px 12px;display:flex;flex-direction:column;gap:6px}
#o1key-notes-list::-webkit-scrollbar,#o1n-content::-webkit-scrollbar{width:4px}
#o1key-notes-list::-webkit-scrollbar-thumb,#o1n-content::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:2px}
.o1n-item{border:1px solid transparent;border-radius:8px;padding:10px;background:transparent;cursor:pointer;transition:all .12s}
.o1n-item:hover{background:rgba(255,255,255,.04);border-color:rgba(255,255,255,.07)}
.o1n-item.active{background:rgba(255,255,255,.065);border-color:rgba(255,255,255,.12);box-shadow:inset 2px 0 0 var(--o1n-blue)}
.o1n-item-top{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.o1n-item-title{font-size:13px;color:#e0e0e0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;min-width:0}
.o1n-item-edit{font-size:11px;color:#777;flex-shrink:0}
.o1n-item-text{font-size:12px;line-height:1.45;color:#8b8b8b;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;word-break:break-word}
.o1n-item-meta{margin-top:8px;display:flex;align-items:center;justify-content:space-between;gap:8px;color:#666;font-size:10px}
.o1n-item-tags{display:flex;gap:4px;overflow:hidden;min-width:0}
.o1n-tag{color:var(--o1n-blue-2);background:rgba(79,140,255,.1);border:1px solid rgba(79,140,255,.2);border-radius:5px;padding:2px 5px;white-space:nowrap;max-width:120px;overflow:hidden;text-overflow:ellipsis}
.o1n-empty{height:100%;display:flex;align-items:center;justify-content:center;text-align:center;color:#666;font-size:13px;line-height:1.6;padding:20px}
.o1n-editor-backdrop{position:absolute;inset:0;background:rgba(0,0,0,.34);display:flex;align-items:stretch;justify-content:flex-end;z-index:30;animation:o1n-fade .12s ease}
.o1n-editor{width:100%;height:100%;background:var(--comfy-menu-bg,#202020);border-left:1px solid rgba(79,140,255,.28);box-shadow:0 22px 70px rgba(0,0,0,.46);display:flex;flex-direction:column;min-height:0}
.o1n-editor-top{height:48px;padding:0 14px;border-bottom:1px solid rgba(255,255,255,.08);display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
.o1n-editor-title{font-size:13px;font-weight:700;color:#eee}
.o1n-editor-body{padding:12px;display:flex;flex-direction:column;gap:9px;flex:1;min-height:0}
#o1n-title-input,#o1n-content{width:100%;border:1px solid rgba(255,255,255,.1);border-radius:8px;background:rgba(255,255,255,.045);color:#ddd;outline:0;font-family:inherit;transition:border-color .15s}
#o1n-title-input{height:34px;padding:0 10px;font-size:13px;font-weight:700;flex-shrink:0}
#o1n-content{flex:1;min-height:160px;resize:none;padding:11px 12px;font-size:13px;line-height:1.58}
#o1n-title-input:focus,#o1n-content:focus{border-color:rgba(79,140,255,.42)}
.o1n-tag-editor{min-height:74px;border:1px solid rgba(255,255,255,.1);border-radius:8px;background:rgba(255,255,255,.035);padding:7px;flex-shrink:0}
.o1n-tag-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.o1n-tag-token{height:24px;padding:0 7px;border-radius:6px;border:1px solid rgba(79,140,255,.25);background:rgba(79,140,255,.1);color:var(--o1n-blue-2);display:inline-flex;align-items:center;gap:6px;font-size:12px}
.o1n-tag-remove{color:#8baee8;font-size:13px;line-height:1;cursor:pointer}
.o1n-tag-input{height:24px;min-width:104px;flex:1;border:1px dashed rgba(255,255,255,.14);border-radius:6px;background:transparent;color:#ddd;padding:0 7px;font-size:12px;outline:0}
.o1n-tag-input::placeholder{color:#777}
.o1n-tag-hint{margin-top:7px;color:#666;font-size:11px;line-height:1.35}
.o1n-editor-actions{display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px;padding:12px;border-top:1px solid rgba(255,255,255,.08);flex-shrink:0}
.o1n-action{height:32px;border:1px solid rgba(255,255,255,.1);border-radius:7px;background:rgba(255,255,255,.035);color:#aaa;font-size:12px;display:flex;align-items:center;justify-content:center;gap:6px;cursor:pointer;transition:all .15s}
.o1n-action:hover{color:#e5e5e5;background:rgba(255,255,255,.075);border-color:rgba(255,255,255,.16)}
.o1n-action.primary{background:var(--o1n-blue);color:#fff;border-color:transparent;font-weight:700}
.o1n-action.primary:hover{background:#6ca0ff;color:#fff}
.o1n-action.accent{background:#d8b45b;color:#171717;border-color:transparent;font-weight:700}
.o1n-action.accent:hover{background:#e3c474;color:#111}
.o1n-action.danger:hover{background:rgba(215,101,101,.18);border-color:rgba(215,101,101,.32);color:#f0a0a0}
.o1n-delete-confirm{grid-column:1 / -1;display:none;align-items:center;gap:7px;padding:7px 8px;border:1px solid rgba(215,101,101,.24);border-radius:7px;background:rgba(215,101,101,.08);color:#ccc;font-size:12px}
.o1n-delete-confirm.show{display:flex}
.o1n-delete-confirm span{flex:1;min-width:0}
.o1n-mini-btn{height:24px;border:1px solid rgba(255,255,255,.1);border-radius:6px;background:rgba(255,255,255,.06);color:#bbb;padding:0 8px;font-size:11px;cursor:pointer}
.o1n-mini-btn.danger{background:rgba(215,101,101,.5);border-color:rgba(215,101,101,.3);color:#fff}
#o1n-status{grid-column:1 / -1;height:16px;color:#777;font-size:11px;line-height:16px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#o1n-import-file{display:none}
@keyframes o1n-fade{from{opacity:0}to{opacity:1}}
`;

function genId() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function now() {
    return Date.now();
}

function parseTags(value) {
    if (Array.isArray(value)) return value.map(String).map(s => s.trim()).filter(Boolean);
    return String(value || "")
        .split(/[,，、\s]+/)
        .map(s => s.trim())
        .filter(Boolean);
}

function makeNote(data = {}) {
    const ts = now();
    return {
        id: data.id || genId(),
        title: String(data.title || "未命名笔记"),
        tags: parseTags(data.tags || data.category || ""),
        content: String(data.content || ""),
        createdAt: data.createdAt || ts,
        updatedAt: data.updatedAt || ts,
    };
}

function cloneNote(note) {
    return {
        ...note,
        tags: [...(note.tags || [])],
    };
}

function formatDate(ts) {
    const d = new Date(ts || now());
    const today = new Date();
    const pad = n => String(n).padStart(2, "0");
    if (d.toDateString() === today.toDateString()) {
        return `今天 ${pad(d.getHours())}:${pad(d.getMinutes())}`;
    }
    return `${pad(d.getMonth() + 1)}/${pad(d.getDate())}`;
}

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function icon(name, size = 14) {
    const icons = {
        plus: `<path d="M12 5v14M5 12h14"/>`,
        search: `<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>`,
        copy: `<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>`,
        insert: `<path d="M12 5v14"/><path d="M19 12H5"/>`,
        save: `<path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><path d="M17 21v-8H7v8"/><path d="M7 3v5h8"/>`,
        trash: `<path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/>`,
        download: `<path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/>`,
        upload: `<path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><path d="M17 8l-5-5-5 5"/><path d="M12 3v12"/>`,
        node: `<rect x="3" y="4" width="7" height="7" rx="1"/><rect x="14" y="13" width="7" height="7" rx="1"/><path d="M10 7.5h3a4 4 0 014 4V13"/>`,
        close: `<path d="M18 6L6 18M6 6l12 12"/>`,
    };
    return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${icons[name] || icons.plus}</svg>`;
}

function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const el = document.createElement("style");
    el.id = STYLE_ID;
    el.textContent = CSS;
    document.head.appendChild(el);
}

function loadNotes() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        notes = raw ? JSON.parse(raw).map(makeNote) : [];
    } catch {
        notes = [];
    }

    if (!localStorage.getItem(SEEDED_KEY)) {
        const existingTitles = new Set(notes.map(n => n.title));
        const samples = SAMPLE_NOTES.map(makeNote).filter(n => !existingTitles.has(n.title));
        notes = [...samples, ...notes];
        localStorage.setItem(SEEDED_KEY, "1");
        saveNotes();
    }
}

function saveNotes() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(notes));
    } catch {}
}

function allTags() {
    const counts = new Map();
    for (const note of notes) {
        for (const tag of note.tags || []) counts.set(tag, (counts.get(tag) || 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12);
}

function filteredNotes() {
    const q = searchText.trim().toLowerCase();
    return notes.filter(note => {
        if (activeFilter.startsWith("tag:") && !(note.tags || []).includes(activeFilter.slice(4))) return false;
        if (!q) return true;
        return [note.title, note.content, ...(note.tags || [])].join("\n").toLowerCase().includes(q);
    }).sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
}

app.registerExtension({
    name: "o1key.notePanel",
    async setup() {
        loadNotes();
        app.extensionManager.registerSidebarTab({
            id: "o1key-notes",
            title: "笔记",
            icon: "pi pi-pencil",
            type: "custom",
            render: (container) => {
                injectStyles();
                renderNotePanel(container);
            },
        });
    },
});

function renderNotePanel(container) {
    container.innerHTML = "";
    container.style.position = "relative";
    container.style.height = "100%";
    container.style.overflow = "hidden";

    const root = document.createElement("div");
    root.id = "o1key-notes-root";
    root.innerHTML = `
        <div id="o1key-notes-header">
            <div class="o1n-title-row">
                <div class="o1n-title">笔记</div>
                <div class="o1n-head-actions">
                    <button class="o1n-icon-btn" id="o1n-import" title="导入 JSON">${icon("upload")}</button>
                    <button class="o1n-icon-btn" id="o1n-export" title="导出 JSON">${icon("download")}</button>
                    <button class="o1n-icon-btn primary" id="o1n-new" title="新建笔记">${icon("plus")}</button>
                </div>
            </div>
            <div id="o1n-search-box">${icon("search", 13)}<input id="o1n-search" placeholder="搜索笔记内容或标签" value="${escapeHtml(searchText)}"></div>
            <div class="o1n-tag-strip" id="o1n-tag-strip"></div>
            <div id="o1n-panel-status"></div>
        </div>
        <div id="o1key-notes-list"></div>
        <input id="o1n-import-file" type="file" accept=".json,application/json">
    `;
    container.appendChild(root);
    noteContainer = root;
    bindPanelEvents(root);
    renderTagFilters();
    renderList();
    renderEditor();
}

function bindPanelEvents(root) {
    root.querySelector("#o1n-new").addEventListener("click", createNote);
    root.querySelector("#o1n-export").addEventListener("click", exportNotes);
    root.querySelector("#o1n-import").addEventListener("click", () => root.querySelector("#o1n-import-file").click());
    root.querySelector("#o1n-import-file").addEventListener("change", importNotes);
    root.querySelector("#o1n-search").addEventListener("input", (e) => {
        searchText = e.target.value;
        renderList();
    });
}

function renderTagFilters() {
    const row = noteContainer?.querySelector("#o1n-tag-strip");
    if (!row) return;
    const tags = allTags();
    row.innerHTML = `<button class="o1n-tag-filter${activeFilter === "all" ? " active" : ""}" data-filter="all">全部 <span>${notes.length}</span></button>` +
        tags.map(([tag, count]) => {
            const filter = `tag:${tag}`;
            return `<button class="o1n-tag-filter${activeFilter === filter ? " active" : ""}" data-filter="${escapeHtml(filter)}">#${escapeHtml(tag)} <span>${count}</span></button>`;
        }).join("");

    row.querySelectorAll("[data-filter]").forEach(btn => {
        btn.addEventListener("click", () => {
            activeFilter = btn.dataset.filter;
            renderTagFilters();
            renderList();
        });
    });
}

function renderList() {
    const list = noteContainer.querySelector("#o1key-notes-list");
    const visible = filteredNotes();

    if (!visible.length) {
        list.innerHTML = `<div class="o1n-empty">没有匹配的笔记</div>`;
        return;
    }

    list.innerHTML = visible.map(note => {
        const tags = (note.tags || []).slice(0, 3).map(tag => `<span class="o1n-tag">#${escapeHtml(tag)}</span>`).join("");
        return `<div class="o1n-item${note.id === editingNoteId ? " active" : ""}" data-id="${note.id}">
            <div class="o1n-item-top">
                <div class="o1n-item-title">${escapeHtml(note.title || "未命名笔记")}</div>
                <div class="o1n-item-edit">编辑</div>
            </div>
            <div class="o1n-item-text">${escapeHtml(note.content || "空笔记")}</div>
            <div class="o1n-item-meta"><div class="o1n-item-tags">${tags || `<span class="o1n-tag">#未分类</span>`}</div><span>${formatDate(note.updatedAt)}</span></div>
        </div>`;
    }).join("");

    list.querySelectorAll(".o1n-item").forEach(item => {
        item.addEventListener("click", () => openEditor(item.dataset.id));
    });
}

function renderEditor() {
    noteContainer.querySelector(".o1n-editor-backdrop")?.remove();
    if (!draftNote) return;
    pendingDelete = false;

    const overlay = document.createElement("div");
    overlay.className = "o1n-editor-backdrop";
    overlay.innerHTML = `
        <div class="o1n-editor">
            <div class="o1n-editor-top">
                <div class="o1n-editor-title">${editingNoteId ? "编辑笔记" : "新建笔记"}</div>
                <button class="o1n-icon-btn" id="o1n-close-editor" title="关闭">${icon("close", 13)}</button>
            </div>
            <div class="o1n-editor-body">
                <input id="o1n-title-input" value="${escapeHtml(draftNote.title)}" placeholder="笔记标题">
                <div class="o1n-tag-editor">
                    <div class="o1n-tag-row">
                        ${(draftNote.tags || []).map(tag => `<span class="o1n-tag-token">${escapeHtml(tag)} <span class="o1n-tag-remove" data-tag="${escapeHtml(tag)}">×</span></span>`).join("")}
                        <input class="o1n-tag-input" id="o1n-tag-input" placeholder="＋ 添加标签">
                    </div>
                    <div class="o1n-tag-hint">输入标签后按 Enter，或用逗号分隔多个标签。</div>
                </div>
                <textarea id="o1n-content" placeholder="记录提示词、参数经验、踩坑结论...">${escapeHtml(draftNote.content)}</textarea>
            </div>
            <div class="o1n-editor-actions">
                <button class="o1n-action" id="o1n-cancel">取消</button>
                <button class="o1n-action" id="o1n-copy">${icon("copy", 13)}复制</button>
                <button class="o1n-action accent" id="o1n-insert">${icon("insert", 13)}插入</button>
                <button class="o1n-action" id="o1n-save-from-node">${icon("node", 13)}从节点保存</button>
                <button class="o1n-action danger" id="o1n-delete">${icon("trash", 13)}删除</button>
                <button class="o1n-action primary" id="o1n-save">${icon("save", 13)}保存</button>
                <div class="o1n-delete-confirm" id="o1n-delete-confirm">
                    <span>确定删除这条笔记？</span>
                    <button class="o1n-mini-btn danger" id="o1n-delete-yes">删除</button>
                    <button class="o1n-mini-btn" id="o1n-delete-no">取消</button>
                </div>
                <div id="o1n-status"></div>
            </div>
        </div>
    `;
    noteContainer.appendChild(overlay);
    bindEditorEvents(overlay);
}

function bindEditorEvents(overlay) {
    overlay.querySelector("#o1n-close-editor").addEventListener("click", closeEditor);
    overlay.querySelector("#o1n-cancel").addEventListener("click", closeEditor);
    overlay.querySelector("#o1n-title-input").addEventListener("input", updateDraftFromEditor);
    overlay.querySelector("#o1n-content").addEventListener("input", updateDraftFromEditor);
    overlay.querySelectorAll(".o1n-tag-remove").forEach(btn => {
        btn.addEventListener("click", () => removeDraftTag(btn.dataset.tag));
    });
    overlay.querySelector("#o1n-tag-input").addEventListener("keydown", handleTagInputKeydown);
    overlay.querySelector("#o1n-tag-input").addEventListener("blur", commitTagInput);
    overlay.querySelector("#o1n-copy").addEventListener("click", copyDraftContent);
    overlay.querySelector("#o1n-insert").addEventListener("click", insertDraftContent);
    overlay.querySelector("#o1n-save-from-node").addEventListener("click", saveFromCurrentNode);
    overlay.querySelector("#o1n-delete").addEventListener("click", requestDeleteEditingNote);
    overlay.querySelector("#o1n-delete-yes").addEventListener("click", deleteEditingNote);
    overlay.querySelector("#o1n-delete-no").addEventListener("click", hideDeleteConfirm);
    overlay.querySelector("#o1n-save").addEventListener("click", saveDraft);
}

function openEditor(id) {
    const note = notes.find(n => n.id === id);
    if (!note) return;
    editingNoteId = id;
    draftNote = cloneNote(note);
    renderList();
    renderEditor();
}

function closeEditor() {
    editingNoteId = null;
    draftNote = null;
    pendingDelete = false;
    renderList();
    renderEditor();
}

function createNote() {
    editingNoteId = null;
    draftNote = makeNote({ title: "新笔记", tags: ["未分类"], content: "" });
    renderList();
    renderEditor();
    noteContainer.querySelector("#o1n-title-input")?.focus();
}

function updateDraftFromEditor() {
    if (!draftNote) return;
    draftNote.title = noteContainer.querySelector("#o1n-title-input")?.value.trim() || "未命名笔记";
    draftNote.content = noteContainer.querySelector("#o1n-content")?.value || "";
    draftNote.updatedAt = now();
}

function addTagsFromText(value) {
    if (!draftNote) return false;
    const incoming = parseTags(value);
    if (!incoming.length) return false;
    const existing = new Set(draftNote.tags || []);
    for (const tag of incoming) existing.add(tag);
    draftNote.tags = [...existing];
    draftNote.updatedAt = now();
    renderDraftTags();
    return true;
}

function commitTagInput() {
    const input = noteContainer?.querySelector("#o1n-tag-input");
    if (!input) return;
    if (addTagsFromText(input.value)) input.value = "";
}

function handleTagInputKeydown(e) {
    if (e.key !== "Enter" && e.key !== "," && e.key !== "，") return;
    e.preventDefault();
    commitTagInput();
}

function removeDraftTag(tag) {
    if (!draftNote) return;
    draftNote.tags = (draftNote.tags || []).filter(t => t !== tag);
    draftNote.updatedAt = now();
    renderDraftTags();
}

function renderDraftTags() {
    const row = noteContainer?.querySelector(".o1n-tag-row");
    if (!row || !draftNote) return;
    row.innerHTML = `
        ${(draftNote.tags || []).map(tag => `<span class="o1n-tag-token">${escapeHtml(tag)} <span class="o1n-tag-remove" data-tag="${escapeHtml(tag)}">×</span></span>`).join("")}
        <input class="o1n-tag-input" id="o1n-tag-input" placeholder="＋ 添加标签">
    `;
    row.querySelectorAll(".o1n-tag-remove").forEach(btn => {
        btn.addEventListener("click", () => removeDraftTag(btn.dataset.tag));
    });
    const input = row.querySelector("#o1n-tag-input");
    input.addEventListener("keydown", handleTagInputKeydown);
    input.addEventListener("blur", commitTagInput);
    input.focus();
}

function saveDraft() {
    if (!draftNote) return;
    updateDraftFromEditor();
    draftNote.tags = draftNote.tags?.length ? draftNote.tags : ["未分类"];
    if (editingNoteId) {
        const idx = notes.findIndex(n => n.id === editingNoteId);
        if (idx >= 0) notes[idx] = cloneNote(draftNote);
    } else {
        draftNote.id = genId();
        draftNote.createdAt = now();
        draftNote.updatedAt = now();
        notes.unshift(cloneNote(draftNote));
        editingNoteId = draftNote.id;
    }
    saveNotes();
    renderTagFilters();
    closeEditor();
}

function requestDeleteEditingNote() {
    if (!editingNoteId) {
        closeEditor();
        return;
    }
    pendingDelete = true;
    const box = noteContainer?.querySelector("#o1n-delete-confirm");
    box?.classList.add("show");
    setStatus("");
}

function hideDeleteConfirm() {
    pendingDelete = false;
    noteContainer?.querySelector("#o1n-delete-confirm")?.classList.remove("show");
}

function deleteEditingNote() {
    if (!editingNoteId) {
        closeEditor();
        return;
    }
    const note = notes.find(n => n.id === editingNoteId);
    if (!note) return;
    notes = notes.filter(n => n.id !== editingNoteId);
    saveNotes();
    renderTagFilters();
    closeEditor();
}

async function copyDraftContent() {
    updateDraftFromEditor();
    try {
        await navigator.clipboard.writeText(draftNote?.content || "");
        setStatus("已复制");
    } catch {
        setStatus("复制失败，请手动选择内容");
    }
}

function insertDraftContent() {
    updateDraftFromEditor();
    const text = draftNote?.content || "";
    if (!text.trim()) {
        setStatus("当前笔记内容为空");
        return;
    }
    if (insertIntoFocusedInput(text) || insertIntoSelectedNode(text)) {
        setStatus("已插入");
        return;
    }
    navigator.clipboard?.writeText(text).catch(() => {});
    setStatus("未找到可插入位置，已复制内容");
}

function setStatus(message) {
    const el = noteContainer?.querySelector("#o1n-status");
    if (!el) return;
    el.textContent = message;
    clearTimeout(setStatus._timer);
    setStatus._timer = setTimeout(() => {
        if (el.textContent === message) el.textContent = "";
    }, 2200);
}

function setPanelStatus(message) {
    const el = noteContainer?.querySelector("#o1n-panel-status");
    if (!el) return;
    el.textContent = message;
    clearTimeout(setPanelStatus._timer);
    setPanelStatus._timer = setTimeout(() => {
        if (el.textContent === message) el.textContent = "";
    }, 2600);
}

function insertIntoFocusedInput(text) {
    const el = document.activeElement;
    if (!el || noteContainer.contains(el)) return false;
    if (!(el instanceof HTMLTextAreaElement || el instanceof HTMLInputElement)) return false;
    const start = el.selectionStart ?? el.value.length;
    const end = el.selectionEnd ?? el.value.length;
    const before = el.value.slice(0, start);
    const after = el.value.slice(end);
    const spacer = before && !before.endsWith("\n") ? "\n" : "";
    el.value = before + spacer + text + after;
    const pos = (before + spacer + text).length;
    el.setSelectionRange(pos, pos);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
}

function findSelectedNode() {
    const selectedNodes = app.canvas?.selected_nodes;
    if (selectedNodes) {
        const values = Array.isArray(selectedNodes) ? selectedNodes : Object.values(selectedNodes);
        if (values.length) return values[0];
    }
    return app.canvas?.selected_node || null;
}

function isPromptWidget(widget) {
    const name = String(widget?.name || "").toLowerCase();
    const value = widget?.value;
    if (typeof value !== "string") return false;
    return (
        name.includes("prompt") ||
        name.includes("提示词") ||
        name.includes("正向") ||
        name === "text" ||
        name === "文本"
    );
}

function insertIntoSelectedNode(text) {
    const node = findSelectedNode();
    if (!node?.widgets?.length) return false;
    const widget = node.widgets.find(isPromptWidget) || node.widgets.find(w => typeof w.value === "string");
    if (!widget) return false;
    const current = widget.value || "";
    const spacer = current && !current.endsWith("\n") ? "\n" : "";
    widget.value = current + spacer + text;
    widget.callback?.(widget.value, app.canvas, node, widget);
    app.graph?.setDirtyCanvas?.(true, true);
    return true;
}

function getPromptWidgetsFromSelectedNode() {
    const node = findSelectedNode();
    if (!node?.widgets?.length) return [];
    return node.widgets.filter(w => typeof w.value === "string" && String(w.value).trim());
}

function saveFromCurrentNode() {
    const widgets = getPromptWidgetsFromSelectedNode();
    if (!widgets.length) {
        setStatus("当前没有选中包含文本的节点");
        return;
    }

    const preferred = widgets.find(isPromptWidget) || widgets[0];
    const node = findSelectedNode();
    draftNote = makeNote({
        title: `${node?.title || node?.type || "节点"}：${preferred.name || "提示词"}`,
        tags: [node?.title || node?.type || "节点"],
        content: preferred.value,
    });
    editingNoteId = null;
    renderEditor();
    setStatus("已读取当前节点内容，保存后写入笔记");
}

function exportNotes() {
    const blob = new Blob([JSON.stringify(notes, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `o1key-notes-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

async function importNotes(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    try {
        const imported = JSON.parse(await file.text());
        const list = Array.isArray(imported) ? imported : imported.notes;
        if (!Array.isArray(list)) throw new Error("Invalid notes file");
        const existing = new Set(notes.map(n => n.id));
        const normalized = list.map(makeNote).map(n => {
            if (existing.has(n.id)) n.id = genId();
            return n;
        });
        notes = [...normalized, ...notes];
        saveNotes();
        renderTagFilters();
        renderList();
        setPanelStatus(`已导入 ${normalized.length} 条笔记`);
    } catch {
        setPanelStatus("导入失败，请确认 JSON 格式");
    }
}
