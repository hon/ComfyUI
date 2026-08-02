// 快捷键触发的画布重排扩展。
//
// 生成器（workflow_scripts/generate_workflows.py）用 NODE_SIZE 预估尺寸布局，
// 但部分节点（图片缩略图、多下拉 widget 等）渲染后的真实高度大于预估，
// 导致画布上节点重叠。本扩展在加载完成后用前端计算的真实尺寸
// （node.size）重新执行同一套布局算法，快捷键 Ctrl+Shift+M 触发（避开
// macOS 上 Option 会改写 e.key 的问题，与内置命令无冲突）。
//
// 快捷键经由命令+快捷键系统（keybindings）注册，因此命令会出现在「快捷键」
// 面板中，用户可在面板中录制/重绑定为任意组合。keybindingService 内置忽略
// 输入框聚焦时的按键，无需在扩展内自行处理。
import { app } from "../../scripts/app.js";

// 与 generate_workflows.py 中的布局常量保持一致。
const MARGIN = 40;
const GAP_X = 60;
const GAP_Y = 60;
const NOTE_SIZE = [400, 200];
const GROUP_PADDING = 20;
const GROUP_TITLE = 30; // LiteGraph.NODE_TITLE_HEIGHT：组标题栏高度
const GROUP_COLUMNS = 4; // 组内最大列数（横向网格），避免深度列铺得太宽
const GROUPS_PER_ROW = 3; // 整行能放多少组，超出换行

// 返回节点所属的组（其中心点落在哪个组框内），无归属则返回 null。
// 归属语义与前端 LGraphGroup.recomputeInsideNodes 一致：节点中心落在组框内。
function groupOfNode(graph, node) {
  let best = null;
  let bestArea = Infinity;
  for (const g of graph.groups) {
    const gx = g.pos[0];
    const gy = g.pos[1];
    const gw = g.size[0];
    const gh = g.size[1];
    const cx = node.pos[0] + (node.size[0] || 0) / 2;
    const cy = node.pos[1] + (node.size[1] || 0) / 2;
    if (cx >= gx && cx <= gx + gw && cy >= gy && cy <= gy + gh && gw * gh < bestArea) {
      best = g;
      bestArea = gw * gh;
    }
  }
  return best;
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

  // 归属以重排前位置判定、取最小面积组（应对嵌套），并固定到这里不再重新推导，
  // 否则组框重叠会把同节点判到不同组导致重叠。
  const membersByGroup = new Map();
  const grouped = new Set();
  for (const n of flow) {
    const g = groupOfNode(graph, n);
    if (g) {
      if (!membersByGroup.has(g)) membersByGroup.set(g, []);
      membersByGroup.get(g).push(n);
      grouped.add(n);
    }
  }

  // 把组当作单位排序：按照组内最浅成员的 depth 升序、同深度再按组原左边距 x 升序。
  // 在组内按 depth 做有限列数的网格（最多 GROUP_COLUMNS 列）横向铺开，避免深度跨度
  // 过大的组铺出几十列、宽度爆炸；组框则shrink-wrap到网格范围。
  const freeFlow = flow.filter((n) => !grouped.has(n));
  const plan = []; // {g, members, cols, colsUsed, w, h, minDepth}
  for (const [g, members] of membersByGroup) {
    const ordered = [...members].sort((a, b) => (depth[a.id] ?? 0) - (depth[b.id] ?? 0) || a.id - b.id);
    const colsUsed = Math.min(GROUP_COLUMNS, ordered.length);
    const cols = Array.from({ length: colsUsed }, () => []);
    ordered.forEach((n, i) => cols[i % colsUsed].push(n));
    const colWidths = cols.map((c) => Math.max(...c.map((n) => n.size[0] || 0)));
    // 各列的实际堆叠高度（节点各异高度 → 列高由该列所有节点之和决定，而非最大单高）。
    const colHeights = cols.map((c) => c.reduce((sum, n) => sum + (n.size[1] || 0), 0) + Math.max(0, c.length - 1) * GAP_Y);
    const rowH = Math.max(0, ...colHeights);
    const w = colWidths.reduce((a, b) => a + b, 0) + (colsUsed - 1) * GAP_X + 2 * GROUP_PADDING;
    const h = rowH + GROUP_TITLE + 2 * GROUP_PADDING;
    plan.push({
      g, members, cols, colsUsed, w, h,
      minDepth: Math.min(...members.map((n) => depth[n.id] ?? 0)),
      rowH,
    });
  }
  plan.sort((a, b) => a.minDepth - b.minDepth || a.g.pos[0] - b.g.pos[0]);

  // 把组整体放到画布上：左→右尽量并排，满 GROUPS_PER_ROW 换行到下一行。
  // 免费节点排在组区下方，以避开组框。
  let gx = MARGIN;
  let gy = MARGIN;
  let rowH = 0;
  let placedInRow = 0;
  let groupBottom = gy;
  for (const p of plan) {
    let cx = gx + GROUP_PADDING;
    const baseY = gy + GROUP_TITLE + GROUP_PADDING;
    p.cols.forEach((col) => {
      let cy = baseY;
      let colW = 0;
      for (const n of col) {
        n.pos = [cx, cy];
        colW = Math.max(colW, n.size[0] || 0);
        cy += n.size[1] + GAP_Y;
      }
      cx += colW + GAP_X;
    });
    p.g.pos = [gx, gy];
    p.g.size[0] = p.w;
    p.g.size[1] = p.h;
    groupBottom = Math.max(groupBottom, gy + p.h);
    rowH = Math.max(rowH, p.h);
    placedInRow++;
    if (placedInRow >= GROUPS_PER_ROW) {
      gx = MARGIN;
      gy += rowH + GAP_Y;
      rowH = 0;
      placedInRow = 0;
    } else {
      gx += p.w + GAP_X;
    }
  }

  // 无组的节点走原有全局列式布局，放到组所在区块下方以避开组框。
  const cols = new Map();
  for (const n of freeFlow) {
    const d = depth[n.id] ?? 0;
    if (!cols.has(d)) cols.set(d, []);
    cols.get(d).push(n);
  }

  // 有 Note 时主流程整体右移，为左上角的说明卡片留出空间。
  let x = MARGIN + (notes.length ? NOTE_SIZE[0] + GAP_X : 0);
  let y0 = groupBottom + GAP_Y;
  for (const d of [...cols.keys()].sort((a, b) => a - b)) {
    let y = y0;
    let colWidth = 0;
    for (const n of cols.get(d)) {
      // 必须整体赋值 pos 以触发 setter（其会同步 layoutStore），
      // 直接写 pos[0]/pos[1] 会绕过 setter，位置不生效。
      n.pos = [x, y];
      colWidth = Math.max(colWidth, n.size[0]);
      y += n.size[1] + GAP_Y;
    }
    x += colWidth + GAP_X;
  }

  // Note 节点固定放在左上角区域。
  notes.forEach((n, i) => {
    n.pos = [MARGIN, MARGIN + i * (NOTE_SIZE[1] + GAP_Y)];
  });

  graph.setDirtyCanvas(true, true);
}

// 经命令+快捷键系统注册（而非原生 keydown 监听）：命令会出现在「快捷键」
// 面板中可重绑定，输入框聚焦忽略由 keybindingService 内置处理。
app.registerExtension({
  name: "AutoRelayout",
  commands: [
    {
      id: "IDPhotoSpec.Relayout",
      label: "IDPhotoSpec: Relayout Workflow",
      function: () => {
        relayout(app.graph);
      },
    },
  ],
  keybindings: [
    {
      commandId: "IDPhotoSpec.Relayout",
      combo: { key: "m", ctrl: true, shift: true },
    },
  ],
});
