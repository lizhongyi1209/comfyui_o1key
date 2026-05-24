import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "o1key.dotGrid",
    async setup() {
        function createBlackTile() {
            const size = 64;
            const c = document.createElement("canvas");
            c.width = size;
            c.height = size;
            const ctx = c.getContext("2d");
            ctx.fillStyle = "#1a1a1a";
            ctx.fillRect(0, 0, size, size);
            return c;
        }

        // Hook immediately so the first draw already uses our tile
        const orig = LGraphCanvas.prototype.drawBackCanvas;
        LGraphCanvas.prototype.drawBackCanvas = function () {
            if (!this._pattern || !this._pattern_img) {
                const ctx = this.bgcanvas?.getContext("2d");
                if (ctx) {
                    const t = createBlackTile();
                    this._pattern = ctx.createPattern(t, "repeat");
                    this._pattern_img = t;
                }
            }
            return orig.apply(this, arguments);
        };

        // Also apply to current canvas instance if already exists
        const canvas = app.canvas;
        if (canvas?.bgcanvas) {
            const bgCtx = canvas.bgcanvas.getContext("2d");
            const tile = createBlackTile();
            canvas._pattern = bgCtx.createPattern(tile, "repeat");
            canvas._pattern_img = tile;
            canvas.draw(true, true);
        }
    },
});
