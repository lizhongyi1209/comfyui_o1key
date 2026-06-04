import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "o1key.hideSidebarItems",
    async setup() {
        const hide = () => {
            // 隐藏侧边栏的"说明"、"应用"、"模型"、"节点"、"模板"按钮
            const hiddenLabels = ["说明", "帮助", "help", "应用", "apps", "模型", "models", "节点", "nodes", "模板", "templates", "template"];
            document.querySelectorAll(".side-bar-button, [class*='sidebar'] button, .p-togglebutton").forEach(btn => {
                const label = [
                    btn.getAttribute("aria-label"),
                    btn.getAttribute("title"),
                    btn.getAttribute("data-title"),
                    btn.getAttribute("data-label"),
                    btn.textContent,
                ].filter(Boolean).join(" ").toLowerCase();
                if (hiddenLabels.some(k => label.includes(k))) {
                    btn.style.display = "none";
                }
            });
            // 隐藏左上角下拉菜单中的"帮助"项
            document.querySelectorAll(".p-menuitem, .p-menu-item, [class*='menu'] li, [class*='Menu'] li").forEach(item => {
                const text = item.textContent || "";
                if (text.trim() === "帮助" || text.trim() === "Help") {
                    item.style.display = "none";
                }
            });
            // 隐藏登录/注册弹框（Google/Github 登录对话框）
            document.querySelectorAll("[class*='dialog'], [class*='Dialog'], [class*='modal'], [class*='Modal']").forEach(dialog => {
                const text = dialog.textContent || "";
                if ((text.includes("Google") || text.includes("Github")) &&
                    (text.includes("登录") || text.includes("注册"))) {
                    dialog.style.display = "none";
                    const mask = dialog.previousElementSibling;
                    if (mask && mask.className && mask.className.includes("mask")) {
                        mask.style.display = "none";
                    }
                }
            });
            // 在右侧内容区隐藏"登录/注册"按钮并注入 API Key
            injectApiKeyPanel();
        };

        async function injectApiKeyPanel() {
            // 找到右侧内容区中包含"我的用户设置"的区域
            let contentArea = null;
            document.querySelectorAll("h1, h2, h3, h4, span, div").forEach(el => {
                const t = (el.textContent || "").trim();
                if (t === "我的用户设置" || t === "My User Settings") {
                    contentArea = el.closest("div");
                }
            });
            if (!contentArea) return;

            // 隐藏"登录/注册"按钮和"登录您的账户"文字
            contentArea.querySelectorAll("button, a, span, p, div").forEach(el => {
                const t = (el.textContent || "").trim();
                if (t.includes("登录") || t.includes("注册") || t === "Sign In" || t === "Sign Up" || t.includes("登录您的账户") || t.includes("Log in")) {
                    if (el.tagName === "BUTTON" || el.tagName === "A" || t.includes("登录您的账户")) {
                        el.style.display = "none";
                    }
                }
            });

            // 检查是否已注入（DOM 中已存在则跳过）
            if (document.querySelector("#o1key-apikey-box")) return;

            // 创建 API Key 输入区域
            const box = document.createElement("div");
            box.id = "o1key-apikey-box";
            box.style.cssText = "margin-top:24px;padding:20px;border:1px solid #444;border-radius:8px;background:#1e1e1e;";
            box.innerHTML = `
                <div style="font-weight:bold;font-size:15px;margin-bottom:6px;color:#eee;">O1Key API 密钥</div>
                <div style="font-size:12px;color:#999;margin-bottom:14px;">输入您的 API 密钥，测试通过后方可保存</div>
                <div style="display:flex;gap:8px;align-items:center;">
                    <input id="o1key-apikey-input" type="text" placeholder="请输入 API 密钥"
                        autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false"
                        data-form-type="other" data-lpignore="true" name="o1key-key-field"
                        style="flex:1;padding:8px 12px;border:1px solid #555;border-radius:4px;background:#111;color:#eee;font-size:13px;" />
                    <button id="o1key-apikey-test"
                        style="padding:8px 14px;border:none;border-radius:4px;background:#47a;color:#fff;cursor:pointer;font-size:13px;white-space:nowrap;">测试令牌</button>
                    <button id="o1key-apikey-save" disabled
                        style="padding:8px 14px;border:none;border-radius:4px;background:#555;color:#999;cursor:not-allowed;font-size:13px;white-space:nowrap;">保存</button>
                    <button id="o1key-apikey-clear"
                        style="padding:8px 14px;border:none;border-radius:4px;background:#a44;color:#fff;cursor:pointer;font-size:13px;white-space:nowrap;">清空密钥</button>
                </div>
                <div id="o1key-apikey-status" style="margin-top:10px;font-size:12px;color:#999;"></div>
            `;
            contentArea.appendChild(box);

            // 加载当前状态
            try {
                const resp = await fetch("/o1key/api_key");
                const data = await resp.json();
                const status = box.querySelector("#o1key-apikey-status");
                if (data.has_key) {
                    status.textContent = "当前密钥: " + data.masked;
                    status.style.color = "#3b8";
                } else {
                    status.textContent = "尚未配置 API 密钥";
                    status.style.color = "#a84";
                }
            } catch (e) {}

            const saveBtn = box.querySelector("#o1key-apikey-save");
            const testBtn = box.querySelector("#o1key-apikey-test");
            let testPassed = false;

            // 输入变化时重置测试状态
            box.querySelector("#o1key-apikey-input").addEventListener("input", () => {
                testPassed = false;
                saveBtn.disabled = true;
                saveBtn.style.background = "#555";
                saveBtn.style.color = "#999";
                saveBtn.style.cursor = "not-allowed";
            });

            // 测试令牌按钮
            testBtn.addEventListener("click", async () => {
                const input = box.querySelector("#o1key-apikey-input");
                const status = box.querySelector("#o1key-apikey-status");
                const key = input.value.trim();
                if (!key) { status.textContent = "请输入密钥"; status.style.color = "#a44"; return; }
                testBtn.disabled = true;
                testBtn.textContent = "验证中...";
                status.textContent = "正在验证密钥...";
                status.style.color = "#999";
                try {
                    const resp = await fetch("/o1key/test_key", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({api_key: key})
                    });
                    const data = await resp.json();
                    if (data.valid) {
                        testPassed = true;
                        status.textContent = "验证通过，可以保存";
                        status.style.color = "#3b8";
                        saveBtn.disabled = false;
                        saveBtn.style.background = "#3b8";
                        saveBtn.style.color = "#fff";
                        saveBtn.style.cursor = "pointer";
                    } else {
                        testPassed = false;
                        status.textContent = data.error || "验证失败";
                        status.style.color = "#a44";
                    }
                } catch (e) {
                    status.textContent = "网络错误";
                    status.style.color = "#a44";
                }
                testBtn.disabled = false;
                testBtn.textContent = "测试令牌";
            });

            // 保存按钮（仅测试通过后可用）
            saveBtn.addEventListener("click", async () => {
                if (!testPassed) return;
                const input = box.querySelector("#o1key-apikey-input");
                const status = box.querySelector("#o1key-apikey-status");
                const key = input.value.trim();
                try {
                    const resp = await fetch("/o1key/api_key", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({api_key: key})
                    });
                    const data = await resp.json();
                    if (data.success) {
                        status.textContent = "密钥已保存";
                        status.style.color = "#3b8";
                        input.value = "";
                        testPassed = false;
                        saveBtn.disabled = true;
                        saveBtn.style.background = "#555";
                        saveBtn.style.color = "#999";
                        saveBtn.style.cursor = "not-allowed";
                    } else {
                        status.textContent = data.error || "保存失败";
                        status.style.color = "#a44";
                    }
                } catch (e) {
                    status.textContent = "网络错误";
                    status.style.color = "#a44";
                }
            });

            // 清空密钥按钮
            const clearBtn = box.querySelector("#o1key-apikey-clear");
            clearBtn.addEventListener("click", async () => {
                if (!confirm("确定要清空 API 密钥吗？")) return;
                const status = box.querySelector("#o1key-apikey-status");
                try {
                    const resp = await fetch("/o1key/api_key", { method: "DELETE" });
                    const data = await resp.json();
                    if (data.success) {
                        status.textContent = "API 密钥已清空";
                        status.style.color = "#a84";
                        box.querySelector("#o1key-apikey-input").value = "";
                        testPassed = false;
                        saveBtn.disabled = true;
                        saveBtn.style.background = "#555";
                        saveBtn.style.color = "#999";
                        saveBtn.style.cursor = "not-allowed";
                    } else {
                        status.textContent = data.error || "清空失败";
                        status.style.color = "#a44";
                    }
                } catch (e) {
                    status.textContent = "网络错误";
                    status.style.color = "#a44";
                }
            });
        }

        const observer = new MutationObserver(hide);
        observer.observe(document.body, { childList: true, subtree: true });
        setTimeout(hide, 1000);
        setTimeout(hide, 3000);
    },
});
