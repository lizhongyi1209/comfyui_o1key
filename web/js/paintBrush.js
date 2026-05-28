import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Fabric.js local loader
let fabricLoaded = false;
function loadFabric() {
  if (fabricLoaded) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    // Load from local extension directory (allowed by CSP 'self')
    script.src = new URL("./lib/fabric.min.js", import.meta.url).href;
    script.onload = () => { fabricLoaded = true; resolve(); };
    script.onerror = () => reject(new Error("Failed to load Fabric.js"));
    document.head.appendChild(script);
  });
}

// --- Styles ---
const STYLES = `
.pb-overlay { position:fixed; inset:0; z-index:99999; background:rgba(0,0,0,0.85); display:flex; flex-direction:column; align-items:center; justify-content:center; }
.pb-toolbar { display:flex; gap:6px; padding:10px 16px; background:#1e1e1e; border-radius:8px; margin-bottom:10px; align-items:center; flex-wrap:wrap; }
.pb-toolbar button { background:#333; color:#eee; border:1px solid #555; border-radius:4px; padding:6px 12px; cursor:pointer; font-size:13px; transition:all .15s; }
.pb-toolbar button:hover { background:#444; }
.pb-toolbar button.active { background:#0066ff; border-color:#0066ff; color:#fff; }
.pb-toolbar .pb-sep { width:1px; height:24px; background:#555; margin:0 4px; }
.pb-toolbar input[type=color] { width:32px; height:28px; border:none; padding:0; cursor:pointer; border-radius:4px; }
.pb-toolbar input[type=range] { width:80px; accent-color:#0066ff; }
.pb-toolbar input.pb-mosaic-size { width:90px; }
.pb-toolbar .pb-label { color:#aaa; font-size:12px; }
.pb-canvas-wrap { border:2px solid #444; border-radius:4px; overflow:hidden; }
.pb-actions { display:flex; gap:10px; margin-top:10px; }
.pb-actions button { padding:8px 24px; border-radius:6px; font-size:14px; cursor:pointer; border:none; }
.pb-actions .pb-cancel { background:#555; color:#eee; }
.pb-actions .pb-confirm { background:#0066ff; color:#fff; }
`;

function injectStyles() {
  if (document.getElementById("pb-styles")) return;
  const el = document.createElement("style");
  el.id = "pb-styles";
  el.textContent = STYLES;
  document.head.appendChild(el);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

const PB_EXPORT_PROPS = ["pbTool"];

function clampPointer(canvas, pointer) {
  return {
    x: clamp(pointer.x, 0, canvas.width),
    y: clamp(pointer.y, 0, canvas.height),
  };
}

function makeCircleCursor(size) {
  const cursorSize = clamp(Math.round(size), 18, 80);
  const center = cursorSize / 2;
  const hotspot = Math.round(center);
  const radius = Math.max(3, center - 2);
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="${cursorSize}" height="${cursorSize}" viewBox="0 0 ${cursorSize} ${cursorSize}">
      <circle cx="${center}" cy="${center}" r="${radius}" fill="none" stroke="black" stroke-width="3"/>
      <circle cx="${center}" cy="${center}" r="${radius}" fill="none" stroke="white" stroke-width="1.5"/>
    </svg>
  `.trim();
  return `url("data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}") ${hotspot} ${hotspot}, crosshair`;
}

function applyMosaicCursor(canvas, state) {
  const cursor = makeCircleCursor(state.mosaicSize);
  canvas.defaultCursor = cursor;
  canvas.hoverCursor = cursor;
}

function configureMosaicObject(obj) {
  obj.set({
    selectable: false,
    hasControls: false,
    hasBorders: false,
    lockMovementX: true,
    lockMovementY: true,
    lockScalingX: true,
    lockScalingY: true,
    lockRotation: true,
    perPixelTargetFind: true,
    objectCaching: false,
  });
  obj.pbTool = "mosaic";
  return obj;
}

function normalizeMosaicObjects(canvas) {
  canvas.getObjects().forEach((obj) => {
    if (obj.pbTool === "mosaic" || obj.type === "image") {
      configureMosaicObject(obj);
    }
  });
}

// --- Get current image URL from node ---
function getImageUrl(node) {
  if (node.imgs && node.imgs.length > 0) {
    return node.imgs[node.imageIndex ?? 0].src;
  }
  const widget = node.widgets?.find(w => w.name === "image");
  if (!widget?.value) return null;
  const val = String(widget.value);
  const match = val.match(/^(.+?)(?:\s*\[(\w+)\])?$/);
  if (!match) return null;
  const filename = match[1];
  const type = match[2] || "input";
  const parts = filename.split("/");
  const name = parts.pop();
  const subfolder = parts.join("/");
  return `/view?filename=${encodeURIComponent(name)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`;
}

// --- History Manager ---
class HistoryManager {
  constructor(canvas) {
    this.canvas = canvas;
    this.stack = [];
    this.index = -1;
    this.locked = false;
  }
  save() {
    if (this.locked) return;
    this.index++;
    this.stack.length = this.index;
    this.stack.push(this.canvas.toJSON(PB_EXPORT_PROPS));
  }
  undo() {
    if (this.index <= 0) return;
    this.index--;
    this._restore();
  }
  redo() {
    if (this.index >= this.stack.length - 1) return;
    this.index++;
    this._restore();
  }
  _restore() {
    this.locked = true;
    this.canvas.loadFromJSON(this.stack[this.index], () => {
      normalizeMosaicObjects(this.canvas);
      this.canvas.renderAll();
      this.locked = false;
    });
  }
}

// --- Shape drawing handler ---
function setupShapeDrawing(canvas, state) {
  let startX, startY, shape;

  canvas.on("mouse:down", (opt) => {
    if (state.tool === "select" || state.tool === "brush" || state.tool === "eraser" || state.tool === "mosaic") return;
    const ptr = canvas.getPointer(opt.e);
    startX = ptr.x;
    startY = ptr.y;
    state.drawing = true;

    const opts = { left: startX, top: startY, fill: "transparent", stroke: state.color, strokeWidth: state.width, selectable: true };

    if (state.tool === "rect") {
      shape = new fabric.Rect({ ...opts, width: 0, height: 0 });
    } else if (state.tool === "circle") {
      shape = new fabric.Ellipse({ ...opts, rx: 0, ry: 0 });
    } else if (state.tool === "line") {
      shape = new fabric.Line([startX, startY, startX, startY], { stroke: state.color, strokeWidth: state.width, selectable: true });
    }
    if (shape) canvas.add(shape);
  });

  canvas.on("mouse:move", (opt) => {
    if (!state.drawing || !shape) return;
    const ptr = canvas.getPointer(opt.e);
    const dx = ptr.x - startX;
    const dy = ptr.y - startY;

    if (state.tool === "rect") {
      shape.set({ left: dx > 0 ? startX : ptr.x, top: dy > 0 ? startY : ptr.y, width: Math.abs(dx), height: Math.abs(dy) });
    } else if (state.tool === "circle") {
      shape.set({ left: dx > 0 ? startX : ptr.x, top: dy > 0 ? startY : ptr.y, rx: Math.abs(dx) / 2, ry: Math.abs(dy) / 2 });
    } else if (state.tool === "line") {
      shape.set({ x2: ptr.x, y2: ptr.y });
    }
    canvas.renderAll();
  });

  canvas.on("mouse:up", () => {
    if (!state.drawing || !shape) return;
    state.drawing = false;
    shape = null;
  });
}

// --- Mosaic brush handler ---
function createMosaicSource(sourceImg, canvas, blockSize) {
  const scaleX = canvas.width / sourceImg.width;
  const scaleY = canvas.height / sourceImg.height;
  const sourceBlockSize = Math.max(1, Math.round(blockSize / Math.min(scaleX, scaleY)));
  const smallW = Math.max(1, Math.ceil(sourceImg.width / sourceBlockSize));
  const smallH = Math.max(1, Math.ceil(sourceImg.height / sourceBlockSize));

  const smallCanvas = document.createElement("canvas");
  smallCanvas.width = smallW;
  smallCanvas.height = smallH;
  const smallCtx = smallCanvas.getContext("2d");
  smallCtx.imageSmoothingEnabled = true;
  smallCtx.drawImage(sourceImg, 0, 0, sourceImg.width, sourceImg.height, 0, 0, smallW, smallH);

  const pixelCanvas = document.createElement("canvas");
  pixelCanvas.width = sourceImg.width;
  pixelCanvas.height = sourceImg.height;
  const pixelCtx = pixelCanvas.getContext("2d");
  pixelCtx.imageSmoothingEnabled = false;
  pixelCtx.drawImage(smallCanvas, 0, 0, smallW, smallH, 0, 0, sourceImg.width, sourceImg.height);

  return pixelCanvas;
}

function getMosaicBrushBounds(sourceImg, canvas, centerX, centerY, size) {
  const scaleX = canvas.width / sourceImg.width;
  const scaleY = canvas.height / sourceImg.height;
  const sourceX = centerX / scaleX;
  const sourceY = centerY / scaleY;
  const radiusX = Math.max(1, (size / 2) / scaleX);
  const radiusY = Math.max(1, (size / 2) / scaleY);

  return {
    centerX: sourceX,
    centerY: sourceY,
    radius: Math.max(radiusX, radiusY),
    left: clamp(Math.floor(sourceX - radiusX), 0, sourceImg.width),
    top: clamp(Math.floor(sourceY - radiusY), 0, sourceImg.height),
    right: clamp(Math.ceil(sourceX + radiusX), 0, sourceImg.width),
    bottom: clamp(Math.ceil(sourceY + radiusY), 0, sourceImg.height),
  };
}

function expandMosaicBounds(bounds, brushBounds) {
  if (!bounds.left && bounds.left !== 0) {
    bounds.left = brushBounds.left;
    bounds.top = brushBounds.top;
    bounds.right = brushBounds.right;
    bounds.bottom = brushBounds.bottom;
    return;
  }
  bounds.left = Math.min(bounds.left, brushBounds.left);
  bounds.top = Math.min(bounds.top, brushBounds.top);
  bounds.right = Math.max(bounds.right, brushBounds.right);
  bounds.bottom = Math.max(bounds.bottom, brushBounds.bottom);
}

function paintMosaicStamp(sourceImg, canvas, mosaicSource, strokeCtx, bounds, point) {
  const brushBounds = getMosaicBrushBounds(sourceImg, canvas, point.x, point.y, point.size);
  if (brushBounds.right <= brushBounds.left || brushBounds.bottom <= brushBounds.top) return false;

  strokeCtx.save();
  strokeCtx.beginPath();
  strokeCtx.arc(brushBounds.centerX, brushBounds.centerY, brushBounds.radius, 0, Math.PI * 2);
  strokeCtx.clip();
  strokeCtx.drawImage(
    mosaicSource,
    brushBounds.left,
    brushBounds.top,
    brushBounds.right - brushBounds.left,
    brushBounds.bottom - brushBounds.top,
    brushBounds.left,
    brushBounds.top,
    brushBounds.right - brushBounds.left,
    brushBounds.bottom - brushBounds.top
  );
  strokeCtx.restore();

  expandMosaicBounds(bounds, brushBounds);
  return true;
}

function paintMosaicLine(sourceImg, canvas, mosaicSource, strokeCtx, bounds, from, to, size) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const distance = Math.hypot(dx, dy);
  const step = Math.max(2, size * 0.25);
  const steps = Math.max(1, Math.ceil(distance / step));
  let painted = false;

  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    painted = paintMosaicStamp(sourceImg, canvas, mosaicSource, strokeCtx, bounds, {
      x: from.x + dx * t,
      y: from.y + dy * t,
      size,
    }) || painted;
  }

  return painted;
}

function createMosaicStrokeObject(strokeCanvas, bounds, canvas, sourceImg) {
  const width = bounds.right - bounds.left;
  const height = bounds.bottom - bounds.top;
  if (width < 1 || height < 1) return Promise.resolve(null);

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = width;
  cropCanvas.height = height;
  cropCanvas.getContext("2d").drawImage(
    strokeCanvas,
    bounds.left,
    bounds.top,
    width,
    height,
    0,
    0,
    width,
    height
  );

  const scaleX = canvas.width / sourceImg.width;
  const scaleY = canvas.height / sourceImg.height;
  const dataUrl = cropCanvas.toDataURL("image/png");
  return new Promise(resolve => {
    fabric.Image.fromURL(dataUrl, (imgObj) => {
      configureMosaicObject(imgObj).set({
        left: bounds.left * scaleX,
        top: bounds.top * scaleY,
        scaleX,
        scaleY,
      });
      resolve(imgObj);
    });
  });
}

function setupMosaicDrawing(canvas, state, sourceImg, history) {
  let lastPoint, mosaicSource, strokeCanvas, strokeCtx, strokePreview, strokeBounds, strokePainted;

  const refreshPreview = () => {
    if (!strokePreview) return;
    strokePreview.dirty = true;
    canvas.requestRenderAll();
  };

  canvas.on("mouse:down", (opt) => {
    if (state.tool !== "mosaic") return;
    const ptr = clampPointer(canvas, canvas.getPointer(opt.e));
    lastPoint = ptr;
    state.drawing = true;
    strokePainted = false;
    strokeBounds = {};
    mosaicSource = createMosaicSource(sourceImg, canvas, state.mosaicSize);

    strokeCanvas = document.createElement("canvas");
    strokeCanvas.width = sourceImg.width;
    strokeCanvas.height = sourceImg.height;
    strokeCtx = strokeCanvas.getContext("2d");
    strokeCtx.imageSmoothingEnabled = false;

    strokePreview = new fabric.Image(strokeCanvas, {
      left: 0,
      top: 0,
      scaleX: canvas.width / sourceImg.width,
      scaleY: canvas.height / sourceImg.height,
      selectable: false,
      evented: false,
      excludeFromExport: true,
      objectCaching: false,
    });

    history.locked = true;
    canvas.add(strokePreview);
    history.locked = false;

    strokePainted = paintMosaicStamp(sourceImg, canvas, mosaicSource, strokeCtx, strokeBounds, {
      x: ptr.x,
      y: ptr.y,
      size: state.mosaicSize,
    }) || strokePainted;
    refreshPreview();
  });

  canvas.on("mouse:move", (opt) => {
    if (state.tool !== "mosaic" || !state.drawing || !strokeCtx || !lastPoint) return;
    const ptr = clampPointer(canvas, canvas.getPointer(opt.e));
    strokePainted = paintMosaicLine(
      sourceImg,
      canvas,
      mosaicSource,
      strokeCtx,
      strokeBounds,
      lastPoint,
      ptr,
      state.mosaicSize
    ) || strokePainted;
    lastPoint = ptr;
    refreshPreview();
  });

  canvas.on("mouse:up", async () => {
    if (!state.drawing || !strokePreview) return;

    history.locked = true;
    canvas.remove(strokePreview);
    history.locked = false;
    state.drawing = false;
    lastPoint = null;

    if (!strokePainted) {
      strokeCanvas = null;
      strokeCtx = null;
      strokePreview = null;
      mosaicSource = null;
      canvas.renderAll();
      return;
    }

    const mosaicObj = await createMosaicStrokeObject(strokeCanvas, strokeBounds, canvas, sourceImg);
    mosaicSource = null;
    strokeCanvas = null;
    strokeCtx = null;
    strokePreview = null;

    if (mosaicObj) {
      canvas.add(mosaicObj);
      canvas.discardActiveObject();
    }
    canvas.renderAll();
  });
}

// --- Open Paint Modal ---
async function openPaintModal(node) {
  injectStyles();
  await loadFabric();

  if (!node.properties) node.properties = {};

  // Detect if user switched to a different image — reset saved state
  const widget = node.widgets?.find(w => w.name === "image");
  const currentVal = widget?.value ? String(widget.value) : "";
  const originalImg = node.properties.paintBrushOriginal;
  if (originalImg && currentVal !== originalImg && currentVal !== "painted_" + originalImg) {
    // Image changed, clear old paint state
    delete node.properties.paintBrushCanvas;
    delete node.properties.paintBrushOriginal;
  }

  const savedState = node.properties.paintBrushCanvas;
  const storedOriginal = node.properties.paintBrushOriginal;

  // If re-editing, use the original image as background; otherwise use current
  let bgUrl;
  if (savedState && storedOriginal) {
    bgUrl = `/view?filename=${encodeURIComponent(storedOriginal)}&type=input&subfolder=`;
  } else {
    bgUrl = getImageUrl(node);
  }
  if (!bgUrl) { alert("请先加载一张图片"); return; }

  // Save original image name on first paint
  if (!storedOriginal) {
    node.properties.paintBrushOriginal = currentVal;
  }

  // Create overlay
  const overlay = document.createElement("div");
  overlay.className = "pb-overlay";

  const maxW = window.innerWidth * 0.8;
  const maxH = window.innerHeight * 0.75;

  // Load image to get dimensions
  const img = await new Promise((resolve, reject) => {
    const i = new Image();
    i.crossOrigin = "anonymous";
    i.onload = () => resolve(i);
    i.onerror = () => reject(new Error("图片加载失败"));
    i.src = bgUrl;
  });

  const scale = Math.min(maxW / img.width, maxH / img.height, 1);
  const cw = Math.round(img.width * scale);
  const ch = Math.round(img.height * scale);

  const state = { tool: "brush", color: "#ff0000", width: 4, mosaicSize: 14, drawing: false };

  const toolbar = buildToolbar(state);
  overlay.appendChild(toolbar);

  const wrap = document.createElement("div");
  wrap.className = "pb-canvas-wrap";
  const canvasEl = document.createElement("canvas");
  canvasEl.width = cw;
  canvasEl.height = ch;
  wrap.appendChild(canvasEl);
  overlay.appendChild(wrap);
  document.body.appendChild(overlay);

  // Init Fabric canvas
  const canvas = new fabric.Canvas(canvasEl, { width: cw, height: ch, isDrawingMode: true });
  canvas.freeDrawingBrush.color = state.color;
  canvas.freeDrawingBrush.width = state.width;

  // Set background image (always the original)
  await new Promise(resolve => {
    canvas.setBackgroundImage(bgUrl, () => { canvas.renderAll(); resolve(); }, {
      scaleX: cw / img.width, scaleY: ch / img.height, crossOrigin: "anonymous"
    });
  });

  // Restore previous drawing objects if re-editing
  if (savedState) {
    await new Promise(resolve => {
      canvas.loadFromJSON(savedState, () => {
        normalizeMosaicObjects(canvas);
        // Re-apply background since loadFromJSON may clear it
        canvas.setBackgroundImage(bgUrl, () => { canvas.renderAll(); resolve(); }, {
          scaleX: cw / img.width, scaleY: ch / img.height, crossOrigin: "anonymous"
        });
      });
    });
  }

  // History
  const history = new HistoryManager(canvas);
  setTimeout(() => history.save(), 300);
  canvas.on("object:added", () => history.save());
  canvas.on("object:modified", () => history.save());

  // Shape drawing
  setupShapeDrawing(canvas, state);
  setupMosaicDrawing(canvas, state, img, history);
  wireToolbar(toolbar, canvas, state, history);

  // Actions buttons
  const actions = document.createElement("div");
  actions.className = "pb-actions";
  actions.innerHTML = `<button class="pb-cancel">取消</button><button class="pb-confirm">确认</button>`;
  overlay.appendChild(actions);

  const close = () => { canvas.dispose(); overlay.remove(); };
  actions.querySelector(".pb-cancel").onclick = close;
  overlay.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
  overlay.tabIndex = 0;
  overlay.focus();

  // Keyboard shortcuts
  overlay.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "z" && !e.shiftKey) { e.preventDefault(); history.undo(); }
    if (e.ctrlKey && (e.key === "Z" || (e.key === "z" && e.shiftKey))) { e.preventDefault(); history.redo(); }
    if (e.key === "Delete" || e.key === "Backspace") {
      const active = canvas.getActiveObject();
      if (active) { canvas.remove(active); canvas.discardActiveObject(); history.save(); }
    }
  });

  // Confirm - save painted image and store canvas state for re-editing
  actions.querySelector(".pb-confirm").onclick = async () => {
    // Save canvas objects (without background) for future re-editing
    node.properties.paintBrushCanvas = canvas.toJSON(PB_EXPORT_PROPS);
    node.graph?.change?.();
    await savePaintedImage(canvas, node, img.width, img.height);
    close();
  };
}

// --- Save painted image ---
async function savePaintedImage(canvas, node, origW, origH) {
  // Export at original resolution
  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = origW;
  exportCanvas.height = origH;
  const ctx = exportCanvas.getContext("2d");

  const dataUrl = canvas.toDataURL({ format: "png", multiplier: origW / canvas.width });
  const exportImg = await new Promise((resolve) => {
    const i = new Image();
    i.onload = () => resolve(i);
    i.src = dataUrl;
  });
  ctx.drawImage(exportImg, 0, 0, origW, origH);

  // Get original filename for naming (use stored original, not current painted name)
  const origName = (node.properties.paintBrushOriginal || "").split("/").pop() || "image.png";
  const paintedName = "painted_" + origName;

  const blob = await new Promise(r => exportCanvas.toBlob(r, "image/png"));
  const formData = new FormData();
  formData.append("image", blob, paintedName);
  formData.append("type", "input");
  formData.append("overwrite", "true");

  const resp = await api.fetchApi("/upload/image", { method: "POST", body: formData });
  const data = await resp.json();

  // Add to combo options so the value persists across refresh
  const widget = node.widgets.find(w => w.name === "image");
  if (widget) {
    if (Array.isArray(widget.options?.values) && !widget.options.values.includes(data.name)) {
      widget.options.values.push(data.name);
    }
    widget.value = data.name;
    if (widget.callback) {
      widget.callback(data.name);
    }
  }
  // Mark graph as changed to trigger workflow auto-save
  node.graph?.change?.();
  app.graph.setDirtyCanvas(true, true);
}

// --- Build Toolbar ---
function buildToolbar(state) {
  const toolbar = document.createElement("div");
  toolbar.className = "pb-toolbar";
  toolbar.innerHTML = `
    <button data-tool="brush" class="active">画笔</button>
    <button data-tool="rect">矩形</button>
    <button data-tool="circle">圆形</button>
    <button data-tool="line">直线</button>
    <button data-tool="mosaic">马赛克</button>
    <button data-tool="eraser">橡皮擦</button>
    <span class="pb-sep"></span>
    <span class="pb-label">颜色</span>
    <input type="color" class="pb-color" value="${state.color}">
    <span class="pb-label">线宽</span>
    <input type="range" class="pb-width" min="1" max="40" value="${state.width}">
    <span class="pb-label">块大小</span>
    <input type="range" class="pb-mosaic-size" min="4" max="80" value="${state.mosaicSize}">
    <span class="pb-sep"></span>
    <button data-action="undo">撤回</button>
    <button data-action="clear">清空</button>
  `;
  return toolbar;
}

// --- Wire Toolbar ---
function wireToolbar(toolbar, canvas, state, history) {
  // Tool buttons
  toolbar.querySelectorAll("[data-tool]").forEach(btn => {
    btn.onclick = () => {
      toolbar.querySelectorAll("[data-tool]").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.tool = btn.dataset.tool;

      if (state.tool === "brush") {
        canvas.isDrawingMode = true;
        canvas.selection = false;
        canvas.defaultCursor = "default";
        canvas.hoverCursor = "move";
        canvas.freeDrawingBrush.color = state.color;
        canvas.freeDrawingBrush.width = state.width;
      } else if (state.tool === "eraser") {
        // Eraser: click on object to delete it
        canvas.isDrawingMode = false;
        canvas.selection = true;
        canvas.defaultCursor = "crosshair";
        canvas.hoverCursor = "pointer";
      } else if (state.tool === "mosaic") {
        canvas.isDrawingMode = false;
        canvas.selection = false;
        canvas.discardActiveObject();
        applyMosaicCursor(canvas, state);
      } else {
        canvas.isDrawingMode = false;
        canvas.selection = false;
        canvas.defaultCursor = "default";
        canvas.hoverCursor = "move";
      }
    };
  });

  // Eraser: remove object on click
  canvas.on("mouse:down", (opt) => {
    if (state.tool !== "eraser") return;
    const target = canvas.findTarget(opt.e);
    if (target) {
      canvas.remove(target);
      canvas.discardActiveObject();
      canvas.renderAll();
      history.save();
    }
  });

  // Color picker
  toolbar.querySelector(".pb-color").oninput = (e) => {
    state.color = e.target.value;
    if (canvas.isDrawingMode) {
      canvas.freeDrawingBrush.color = state.color;
    }
  };

  // Width slider
  toolbar.querySelector(".pb-width").oninput = (e) => {
    state.width = parseInt(e.target.value);
    if (canvas.isDrawingMode) {
      canvas.freeDrawingBrush.width = state.width;
    }
  };

  toolbar.querySelector(".pb-mosaic-size").oninput = (e) => {
    state.mosaicSize = parseInt(e.target.value);
    if (state.tool === "mosaic") {
      applyMosaicCursor(canvas, state);
    }
  };

  // Action buttons
  toolbar.querySelector("[data-action=undo]").onclick = () => history.undo();
  toolbar.querySelector("[data-action=clear]").onclick = () => {
    canvas.getObjects().forEach(obj => canvas.remove(obj));
    canvas.renderAll();
    history.save();
  };
}

// --- Extension Registration ---
app.registerExtension({
  name: "o1key.paintBrush",

  // Register command for toolbar button
  commands: [
    {
      id: "o1key.PaintBrush",
      icon: "pi pi-pencil",
      label: "画笔",
      tooltip: "画笔",
      function: () => {
        const selectedNodes = app.canvas.selected_nodes;
        const node = selectedNodes ? Object.values(selectedNodes)[0] : null;
        if (node) openPaintModal(node);
      }
    }
  ],

  // Add title attribute to our toolbar button for native tooltip
  // Hide "节点信息" button and reorder paint brush next to mask editor
  setup() {
    const observer = new MutationObserver(() => {
      const toolbox = document.querySelector('[class*="selection-toolbox"], [class*="SelectionToolbox"]');
      if (!toolbox) return;

      // Hide "节点信息" button (matches by aria-label or title)
      toolbox.querySelectorAll("button").forEach(btn => {
        const label = btn.title || btn.getAttribute("aria-label") || btn.textContent || "";
        if (label.includes("节点信息") || label.includes("Node Info") || label.includes("Info")) {
          btn.style.display = "none";
        }
      });

      // Find our paint brush button and move it next to mask editor
      const pencilIcon = toolbox.querySelector('[class*="pi-pencil"]');
      if (pencilIcon) {
        const paintBtn = pencilIcon.closest("button");
        if (paintBtn && !paintBtn.title) paintBtn.title = "画笔";

        // Find mask editor button (has mask/pen-tool icon)
        const allBtns = Array.from(toolbox.querySelectorAll("button"));
        const maskBtn = allBtns.find(b => {
          const cls = b.innerHTML || "";
          return cls.includes("mask") || cls.includes("pen-tool") || cls.includes("Mask");
        }) || allBtns[0];

        if (maskBtn && paintBtn && paintBtn.previousElementSibling !== maskBtn) {
          maskBtn.after(paintBtn);
        }
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
  },

  // Show button in toolbar when LoadImage node is selected
  getSelectionToolboxCommands(item) {
    if (item?.comfyClass === "LoadImage" || item?.type === "LoadImage") {
      return ["o1key.PaintBrush"];
    }
    return [];
  },

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LoadImage") return;

    const origMenu = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (canvasRef, options) {
      origMenu?.call(this, canvasRef, options);
      options.unshift({
        content: "画笔 (Paint)",
        callback: () => openPaintModal(this)
      });
    };
  }
});
