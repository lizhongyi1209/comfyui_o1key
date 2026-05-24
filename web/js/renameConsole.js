import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "o1key.renameConsole",
    async setup() {
        const rename = () => {
            document.querySelectorAll(".side-bar-button, [class*='sidebar'] button, .p-togglebutton, button").forEach(btn => {
                const label = btn.getAttribute("aria-label") || "";
                const text = btn.textContent || "";
                if (label === "控制台" || label === "Console" || text.trim() === "控制台" || text.trim() === "Console") {
                    if (label === "控制台" || label === "Console") {
                        btn.setAttribute("aria-label", "日志");
                    }
                    const span = btn.querySelector("span");
                    if (span && (span.textContent.trim() === "控制台" || span.textContent.trim() === "Console")) {
                        span.textContent = "日志";
                    } else if (!span && (btn.textContent.trim() === "控制台" || btn.textContent.trim() === "Console")) {
                        btn.textContent = "日志";
                    }
                }
            });
        };

        const observer = new MutationObserver(rename);
        observer.observe(document.body, { childList: true, subtree: true });
        setTimeout(rename, 1000);
        setTimeout(rename, 3000);
    },
});
