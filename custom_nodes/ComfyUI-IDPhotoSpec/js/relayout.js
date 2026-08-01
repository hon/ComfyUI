// 快捷键触发的画布重排扩展。
//
// 生成器（workflow_scripts/generate_workflows.py）用 NODE_SIZE 预估尺寸布局，
// 但部分节点（图片缩略图、多下拉 widget 等）渲染后的真实高度大于预估，
// 导致画布上节点重叠。本扩展在加载完成后用前端计算的真实尺寸
// （node.size）重新执行同一套布局算法，快捷键 Ctrl+Alt+L 触发。
import { app } from "../../scripts/app.js";

// 与 generate_workflows.py 中的布局常量保持一致。
const MARGIN = 40;
const GAP_X = 60;
const GAP_Y = 60;
const NOTE_SIZE = [400, 200];

// 输入框/可编辑区域聚焦时忽略快捷键，避免输入过程中误触。
function isEditableTarget() {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || el.isContentEditable;
}

// 计算每个节点的依赖深度（从源节点出发的最长路径），
// 与生成器的 DFS 逻辑一致：depth = 1 + max(depth[parent])。
function computeDepth(graph) {
  const byId = new Map(graph._nodes.map((n) => [n.id, n]));
  const links = graph.links || {};

  function parentsOf(node) {
    const parents = [];
    if (!node.inputs) return parents;
    for (const slot of node.inputs) {
      if (slot.link != null) {
        const l = links[slot.link];
        if (l && byId.has(l.origin_id)) parents.push(l.origin_id);
      }
    }
    return parents;
  }

  const depth = {};
  for (const id of graph._nodes.map((n) => n.id)) {
    if (id in depth) continue;
    const stack = [id];
    while (stack.length) {
      const cur = stack[stack.length - 1];
      const pending = parentsOf(byId.get(cur)).filter((p) => !(p in depth));
      if (pending.length) {
        stack.push(...pending);
        continue;
      }
      const pd = parentsOf(byId.get(cur)).map((p) => depth[p]);
      depth[cur] = 1 + (pd.length ? Math.max(...pd) : 0);
      stack.pop();
    }
  }
  return depth;
}

// 用真实渲染尺寸重排：按依赖深度分列，列内从上到下、列间从左到右。
function relayout(graph) {
  const flow = graph._nodes
    .filter((n) => n.type !== "Note")
    .sort((a, b) => a.id - b.id);
  const notes = graph._nodes.filter((n) => n.type === "Note");

  // 使用节点当前已渲染的尺寸（含图片预览等加载后的真实大小）。
  // 注意 node.size 可能是 Float64Array，不能用 Array.isArray 判断。
  // computeSize() 对含图片的节点会漏算图片预览高度，只能作为未初始化时的回退。
  for (const n of flow) {
    let s = n.size;
    if (!s || s.length < 2 || s[0] <= 0 || s[1] <= 0) {
      if (typeof n.computeSize === "function") {
        try {
          s = n.computeSize();
        } catch (e) {
          // 尺寸计算失败时保留原尺寸，布局仍按原值累加。
        }
      }
    }
    if (s && s.length >= 2 && s[0] > 0 && s[1] > 0) {
      n.size[0] = s[0];
      n.size[1] = s[1];
    }
  }

  const depth = computeDepth(graph);
  const cols = new Map();
  for (const n of flow) {
    const d = depth[n.id] ?? 0;
    if (!cols.has(d)) cols.set(d, []);
    cols.get(d).push(n);
  }

  // 有 Note 时主流程整体右移，为左上角的说明卡片留出空间。
  let x = MARGIN + (notes.length ? NOTE_SIZE[0] + GAP_X : 0);
  for (const d of [...cols.keys()].sort((a, b) => a - b)) {
    let y = MARGIN;
    let colWidth = 0;
    for (const n of cols.get(d)) {
      n.pos[0] = x;
      n.pos[1] = y;
      colWidth = Math.max(colWidth, n.size[0]);
      y += n.size[1] + GAP_Y;
    }
    x += colWidth + GAP_X;
  }

  // Note 节点固定放在左上角区域。
  notes.forEach((n, i) => {
    n.pos[0] = MARGIN;
    n.pos[1] = MARGIN + i * (NOTE_SIZE[1] + GAP_Y);
  });

  graph.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "AutoRelayout",
  setup() {
    // 在捕获阶段监听 keydown：ComfyUI 的键盘系统在冒泡阶段拦截按键，
    // 冒泡阶段的 hotkeys-js 收不到事件，捕获阶段则不受影响。
    window.addEventListener(
      "keydown",
      (e) => {
        if (isEditableTarget()) return;
        if ((e.ctrlKey || e.metaKey) && e.altKey && e.code === "KeyL") {
          e.preventDefault();
          relayout(app.graph);
        }
      },
      true,
    );
  },
});
