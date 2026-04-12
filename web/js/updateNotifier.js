import { api } from "../../../scripts/api.js";

api.addEventListener("o1key.update_available", (event) => {
    const message = event.detail?.message || "欢迎使用o1key工作流";

    const style = document.createElement("style");
    style.textContent = `
        @keyframes o1key-fadein {
            from { opacity: 0; transform: translateY(16px) scale(0.97); }
            to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        .o1key-toast-close:hover { color: #fff !important; }
    `;
    document.head.appendChild(style);

    const toast = document.createElement("div");
    toast.style.cssText = `
        position: fixed;
        bottom: 28px;
        left: 28px;
        background: linear-gradient(135deg, #0d3320 0%, #145a32 60%, #1e8449 100%);
        color: #d5f5e3;
        border: 1px solid #27ae60;
        border-radius: 12px;
        padding: 18px 20px 16px 20px;
        font-size: 15px;
        z-index: 99999;
        box-shadow: 0 6px 24px rgba(39,174,96,0.35), 0 2px 8px rgba(0,0,0,0.5);
        max-width: 340px;
        animation: o1key-fadein 0.4s cubic-bezier(.22,.68,0,1.2);
    `;

    const header = document.createElement("div");
    header.style.cssText = `
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    `;

    const title = document.createElement("span");
    title.textContent = "🐴 o1key 工作流";
    title.style.cssText = `font-weight: bold; font-size: 13px; color: #82e0aa; letter-spacing: 0.5px;`;

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "×";
    closeBtn.className = "o1key-toast-close";
    closeBtn.style.cssText = `
        background: none;
        border: none;
        color: #82e0aa;
        font-size: 20px;
        cursor: pointer;
        padding: 0;
        line-height: 1;
    `;
    closeBtn.onclick = () => toast.remove();

    header.appendChild(title);
    header.appendChild(closeBtn);

    const divider = document.createElement("div");
    divider.style.cssText = `height: 1px; background: rgba(39,174,96,0.3); margin-bottom: 10px;`;

    const text = document.createElement("div");
    text.textContent = message;
    text.style.cssText = `line-height: 1.7; color: #d5f5e3;`;

    toast.appendChild(header);
    toast.appendChild(divider);
    toast.appendChild(text);
    document.body.appendChild(toast);
});
