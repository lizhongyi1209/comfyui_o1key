import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "o1key.assetToggle",
    settings: [
        {
            id: "o1key.AssetSave",
            name: "资产保存",
            tooltip: "持久性保存生图记录",
            type: "boolean",
            defaultValue: true,
        },
    ],
    async init() {
        const enabled = () => {
            try {
                return app.ui.settings.getSettingValue("o1key.AssetSave", true);
            } catch {
                return true;
            }
        };

        // --- Patch getHistory: 合并真实 jobs 与持久化历史 ---
        const _origGetHistory = api.getHistory.bind(api);
        api.getHistory = async function (maxItems = 200, opts = {}) {
            const real = await _origGetHistory(maxItems, opts);
            if (!enabled()) return real;
            try {
                const offset = opts?.offset || 0;
                const resp = await fetch(
                    `/o1key/output_history?limit=${maxItems}&offset=${offset}`
                );
                if (!resp.ok) return real;
                const data = await resp.json();
                const persisted = data.jobs || [];
                if (!persisted.length) return real;
                if (!real || !Array.isArray(real) || !real.length) {
                    // 仅持久化数据时也补充 priority
                    const t = data.pagination?.total || persisted.length;
                    return persisted.map((j, i) => ({
                        ...j,
                        priority: j.priority ?? t - i,
                    }));
                }
                // 合并去重：以 id 为 key，真实 jobs 优先
                const seen = new Set(real.map((j) => j.id));
                const merged = [...real];
                for (const job of persisted) {
                    if (!seen.has(job.id)) {
                        merged.push(job);
                    }
                }
                // 按时间倒序
                merged.sort(
                    (a, b) => (b.create_time || 0) - (a.create_time || 0)
                );
                const result = merged.slice(0, maxItems);
                // 补充 priority 字段（队列面板依赖此字段排序）
                const total = data.pagination?.total || result.length;
                for (let idx = 0; idx < result.length; idx++) {
                    if (result[idx].priority == null) {
                        result[idx] = {
                            ...result[idx],
                            priority: total - idx,
                        };
                    }
                }
                return result;
            } catch (e) {
                return real;
            }
        };

        // --- Patch getJobDetail: 真实 API 失败时回退到本地路由 ---
        const _origGetJobDetail = api.getJobDetail.bind(api);
        api.getJobDetail = async function (jobId) {
            const real = await _origGetJobDetail(jobId);
            if (real) return real;
            if (!enabled()) return undefined;
            try {
                const resp = await fetch(
                    `/o1key/job_detail/${encodeURIComponent(jobId)}`
                );
                if (!resp.ok) return undefined;
                return await resp.json();
            } catch (e) {
                return undefined;
            }
        };
    },
});
