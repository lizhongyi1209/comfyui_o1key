import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "comfyui_o1key.videoPreview",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "VideoPreview") return;

        const origOnExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onExecuted = function (message) {
            if (origOnExecuted) {
                origOnExecuted.apply(this, arguments);
            }

            const videos = message?.videos;
            if (!videos || videos.length === 0) return;

            const videoInfo = videos[0];
            const params = new URLSearchParams();
            params.set("filename", videoInfo.filename);
            if (videoInfo.subfolder) params.set("subfolder", videoInfo.subfolder);
            params.set("type", videoInfo.type || "output");

            const videoUrl = api.apiURL(`/view?${params.toString()}`);

            // ── 首次创建 DOM 结构 ──────────────────────────────
            if (!this._videoContainer) {
                this._videoContainer = document.createElement("div");
                this._videoContainer.style.cssText =
                    "width:100%;display:flex;flex-direction:column;align-items:center;" +
                    "padding:4px;box-sizing:border-box;";

                this._videoEl = document.createElement("video");
                this._videoEl.controls = true;
                this._videoEl.loop = true;
                this._videoEl.autoplay = true;
                this._videoEl.muted = true;
                this._videoEl.playsInline = true;
                // 宽度铺满容器，高度由 object-fit 自适应，不限制 max-height
                this._videoEl.style.cssText =
                    "width:100%;display:block;border-radius:4px;" +
                    "background:#000;object-fit:contain;";

                this._videoLabel = document.createElement("div");
                this._videoLabel.style.cssText =
                    "font-size:10px;color:#aaa;margin-top:2px;" +
                    "text-align:center;word-break:break-all;";

                this._videoResLabel = document.createElement("div");
                this._videoResLabel.style.cssText =
                    "font-size:10px;color:#888;margin-top:1px;" +
                    "text-align:center;";

                this._videoContainer.appendChild(this._videoEl);
                this._videoContainer.appendChild(this._videoLabel);
                this._videoContainer.appendChild(this._videoResLabel);

                // ── 视频元数据加载后，根据真实宽高比重新调整节点大小 ──
                this._videoEl.addEventListener("loadedmetadata", () => {
                    const vw = this._videoEl.videoWidth;
                    const vh = this._videoEl.videoHeight;
                    if (!vw || !vh) return;

                    // 存储宽高比（高/宽），供 computeSize 使用
                    this._videoAspectRatio = vh / vw;

                    // 显示分辨率
                    if (this._videoResLabel) {
                        this._videoResLabel.textContent = `${vw} × ${vh}`;
                    }

                    // 用真实比例重新计算节点高度
                    this._resizeToVideo();
                });
            }

            this._videoEl.src = videoUrl;
            this._videoLabel.textContent = videoInfo.filename;

            // ── 注册 DOM Widget（仅第一次）──────────────────────
            if (!this.widgets?.find((w) => w.name === "video_preview_widget")) {
                const self = this;
                const widget = this.addDOMWidget(
                    "video_preview_widget",
                    "div",
                    this._videoContainer,
                    { serialize: false, hideOnZoom: false }
                );

                // computeSize 在 LiteGraph 布局时被调用，返回 [宽, 高]
                widget.computeSize = function (width) {
                    const w = width ?? self.size?.[0] ?? 300;
                    if (self._videoAspectRatio) {
                        const innerW = Math.max(w - 16, 10); // 减去左右 padding
                        const videoH = Math.round(innerW * self._videoAspectRatio);
                        return [w, videoH + 40]; // +40 = 文件名 + 分辨率标签高度
                    }
                    // 元数据未就绪时给一个合理默认值
                    return [w, 260];
                };
            }

            // 初次渲染（元数据尚未加载）给出合理初始尺寸
            if (!this._videoAspectRatio) {
                const w = Math.max(this.size[0], 320);
                const h = Math.max(this.size[1], 300);
                this.setSize([w, h]);
            }

            this.setDirtyCanvas(true, true);
        };

        // ── 辅助方法：按视频真实比例自适应节点大小 ──────────────
        nodeType.prototype._resizeToVideo = function () {
            if (!this._videoAspectRatio) return;

            const nodeWidth = Math.max(this.size[0], 320);
            const innerW = nodeWidth - 16;
            const videoH = Math.round(innerW * this._videoAspectRatio);
            const labelH = 40; // 文件名 + 分辨率两行

            // 节点头部 + 其他 widget 的高度
            // LiteGraph 节点头部约 30px，每个普通 widget 约 24px
            const NON_VIDEO_WIDGETS = (this.widgets?.filter(
                (w) => w.name !== "video_preview"
            ).length ?? 0);
            const headerH = 58 + NON_VIDEO_WIDGETS * 24;

            const totalH = headerH + videoH + labelH;

            this.setSize([nodeWidth, totalH]);
            this.setDirtyCanvas(true, true);
        };

        // ── 节点手动缩放时同步更新视频高度 ─────────────────────
        const origOnResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            if (origOnResize) origOnResize.apply(this, arguments);
            if (this._videoAspectRatio && this._videoEl) {
                // 用新宽度重新计算正确高度，避免拉伸/压缩
                const innerW = Math.max(size[0] - 16, 10);
                const videoH = Math.round(innerW * this._videoAspectRatio);
                this._videoEl.style.height = videoH + "px";
            }
        };
    },
});
