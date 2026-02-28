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

            if (!this._videoContainer) {
                this._videoContainer = document.createElement("div");
                this._videoContainer.style.cssText =
                    "width:100%;display:flex;flex-direction:column;align-items:center;padding:4px;box-sizing:border-box;";

                this._videoEl = document.createElement("video");
                this._videoEl.controls = true;
                this._videoEl.loop = true;
                this._videoEl.autoplay = true;
                this._videoEl.muted = true;
                this._videoEl.playsInline = true;
                this._videoEl.style.cssText =
                    "width:100%;max-height:400px;border-radius:4px;background:#000;object-fit:contain;";

                this._videoLabel = document.createElement("div");
                this._videoLabel.style.cssText =
                    "font-size:10px;color:#aaa;margin-top:2px;text-align:center;word-break:break-all;";

                this._videoContainer.appendChild(this._videoEl);
                this._videoContainer.appendChild(this._videoLabel);
            }

            this._videoEl.src = videoUrl;
            this._videoLabel.textContent = videoInfo.filename;

            const existingWidget = this.widgets?.find((w) => w.name === "video_preview");
            if (!existingWidget) {
                const widget = this.addDOMWidget(
                    "video_preview",
                    "div",
                    this._videoContainer,
                    {
                        serialize: false,
                        hideOnZoom: false,
                    }
                );
                widget.computeSize = function () {
                    return [this.parent?.size?.[0] || 300, 260];
                };
            }

            const desiredWidth = Math.max(this.size[0], 320);
            const desiredHeight = Math.max(300, 260 + (this.widgets?.length || 0) * 20 + 60);
            this.setSize([desiredWidth, desiredHeight]);
            this.setDirtyCanvas(true, true);
        };
    },
});
