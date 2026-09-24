import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "o1key.restartButton",
    async setup() {
        function inject() {
            if (document.querySelector("#o1k-restart-btn") && document.querySelector("#o1k-update-btn")) return;

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

            function makeButton(id, label, title, icon) {
                const btn = logBtn.cloneNode(false);
                btn.id = id;
                btn.setAttribute("aria-label", label);
                btn.title = title;
                const logStyle = window.getComputedStyle(logBtn);
                btn.style.display = "flex";
                btn.style.flexDirection = "column";
                btn.style.alignItems = "center";
                btn.style.justifyContent = "center";
                btn.style.gap = logStyle.gap || "4px";
                const iconSpan = document.createElement("span");
                iconSpan.innerHTML = icon;
                const textSpan = document.createElement("span");
                textSpan.textContent = label;
                btn.append(iconSpan, textSpan);
                return btn;
            }

            let restartBtn = document.querySelector("#o1k-restart-btn");
            if (!restartBtn) {
                restartBtn = makeButton("o1k-restart-btn", "重启", "重启 ComfyUI",
                    `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 2v6h-6"/><path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M3 22v-6h6"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/></svg>`);
                restartBtn.addEventListener("click", async () => {
                    if (!confirm("确定要重启 ComfyUI 吗？")) return;
                    restartBtn.style.opacity = "0.5";
                    restartBtn.style.pointerEvents = "none";
                    await disableExperimentalAssetApi();
                    try { await fetch("/o1key/restart", { method: "POST" }); } catch {}
                    pollUntilReady();
                });
                logBtn.parentNode.insertBefore(restartBtn, logBtn);
            }

            if (!document.querySelector("#o1k-update-btn")) {
                const updateBtn = makeButton("o1k-update-btn", "更新", "更新 comfyui_o1key 节点包",
                    `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M4 18v3h16v-3"/></svg>`);
                let updating = false;
                updateBtn.addEventListener("click", async () => {
                    if (updating) return;
                    if (!confirm("从 origin/main 拉取 comfyui_o1key 最新版本？")) return;
                    updating = true;
                    updateBtn.disabled = true;
                    updateBtn.style.opacity = "0.5";
                    updateBtn.title = "正在更新...";
                    try {
                        const response = await fetch("/o1key/update", {
                            method: "POST",
                            headers: { "X-O1Key-Update": "1" },
                        });
                        const result = await response.json();
                        if (!response.ok) throw new Error(result.error || "更新失败");
                        if (!result.updated) {
                            alert(`已是最新版本（${result.version}）。`);
                        } else {
                            const dependencies = result.requirements_changed
                                ? "\n依赖列表已变化，请先在 ComfyUI 的 Python 环境中执行 pip install -r requirements.txt。"
                                : "";
                            alert(`更新完成（${result.version}）。${dependencies}\n请点击“重启”使新版本生效。`);
                        }
                    } catch (error) {
                        alert(`更新失败：${error.message}`);
                    } finally {
                        updating = false;
                        updateBtn.disabled = false;
                        updateBtn.style.opacity = "";
                        updateBtn.title = "更新 comfyui_o1key 节点包";
                    }
                });
                restartBtn.after(updateBtn);
            }
        }

        async function disableExperimentalAssetApi() {
            if (!(await shouldDisableExperimentalAssetApi())) return;
            try {
                await fetch("/api/settings/Comfy.Assets.UseAssetAPI", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(false),
                    signal: AbortSignal.timeout(2000),
                });
            } catch {}
        }

        async function shouldDisableExperimentalAssetApi() {
            try {
                const r = await fetch("/api/settings/Comfy.Assets.UseAssetAPI", {
                    cache: "no-store",
                    signal: AbortSignal.timeout(2000),
                });
                if (!r.ok || !(await r.json())) return false;
            } catch {
                return false;
            }
            return !(await fetchOk("/api/assets/seed/status", 2000));
        }

        async function fetchOk(url, timeout = 2500) {
            try {
                const r = await fetch(url, {
                    cache: "no-store",
                    signal: AbortSignal.timeout(timeout),
                });
                return r.ok;
            } catch {
                return false;
            }
        }

        async function comfyReady() {
            const [statsOk, modelFoldersOk] = await Promise.all([
                fetchOk("/api/system_stats"),
                fetchOk("/api/experiment/models"),
            ]);
            return statsOk && modelFoldersOk;
        }

        function pollUntilReady() {
            let attempts = 0;
            const maxAttempts = 80;
            const minRestartWaitMs = 5000;
            const startedAt = Date.now();
            let sawUnavailable = false;
            const interval = setInterval(async () => {
                attempts++;
                if (attempts > maxAttempts) { clearInterval(interval); forceReload(); return; }
                const ready = await comfyReady();
                if (!ready) {
                    sawUnavailable = true;
                    return;
                }
                if (!sawUnavailable && Date.now() - startedAt < minRestartWaitMs) return;

                clearInterval(interval);
                await disableExperimentalAssetApi();
                setTimeout(forceReload, 800);
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
