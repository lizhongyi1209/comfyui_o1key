import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "o1key.restartButton",
    async setup() {
        let injected = false;

        function inject() {
            if (injected) return;
            if (document.querySelector("#o1k-restart-btn")) { injected = true; return; }

            const allBtns = document.querySelectorAll("button, .p-togglebutton, .side-bar-button");
            let logBtn = null;
            for (const btn of allBtns) {
                const label = (btn.getAttribute("aria-label") || "") + (btn.textContent || "");
                if (label.includes("日志") || label.includes("Console") || label.includes("控制台") || label.includes("Logs")) {
                    logBtn = btn;
                    break;
                }
            }
            if (!logBtn || !logBtn.parentNode) return;

            const btn = logBtn.cloneNode(false);
            btn.id = "o1k-restart-btn";
            btn.setAttribute("aria-label", "重启");
            btn.title = "重启 ComfyUI";

            const logStyle = window.getComputedStyle(logBtn);
            btn.style.display = "flex";
            btn.style.flexDirection = "column";
            btn.style.alignItems = "center";
            btn.style.justifyContent = "center";
            btn.style.gap = logStyle.gap || "4px";

            const iconSpan = document.createElement("span");
            iconSpan.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 2v6h-6"/><path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M3 22v-6h6"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/></svg>`;
            const textSpan = document.createElement("span");
            textSpan.textContent = "重启";
            btn.appendChild(iconSpan);
            btn.appendChild(textSpan);

            btn.addEventListener("click", async () => {
                if (!confirm("确定要重启 ComfyUI 吗？")) return;
                btn.style.opacity = "0.5";
                btn.style.pointerEvents = "none";
                try { await fetch("/o1key/restart", { method: "POST" }); } catch {}
                pollUntilReady();
            });

            logBtn.parentNode.insertBefore(btn, logBtn);
            injected = true;
        }

        function pollUntilReady() {
            let attempts = 0;
            const maxAttempts = 40;
            const interval = setInterval(async () => {
                attempts++;
                if (attempts > maxAttempts) { clearInterval(interval); forceReload(); return; }
                try {
                    const r = await fetch("/api/system_stats", { signal: AbortSignal.timeout(2000) });
                    if (r.ok) { clearInterval(interval); forceReload(); }
                } catch {}
            }, 1500);
        }

        function forceReload() {
            window.onbeforeunload = null;
            Object.defineProperty(BeforeUnloadEvent.prototype, "returnValue", {
                get() { return ""; },
                set() {}
            });
            location.reload();
        }

        const observer = new MutationObserver(inject);
        observer.observe(document.body, { childList: true, subtree: true });
        setTimeout(inject, 2000);
        setTimeout(inject, 4000);
        setTimeout(inject, 8000);
    },
});
