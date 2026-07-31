#!/usr/bin/env python3
"""生成中国护照证件照工作流（Web 画布 UI 格式）。

两个变体：
  - id_photo_generate.json：模式 1/2/3a
      模式 1/2：发型由提示词控制（默认预设或用户文本）
      模式 3a：保留参考照片的发型（img2img，denoise ~0.5）
    两者共享同一张图：参考图是 img2img 的起点，denoise 是模式开关
    （1.0 = 纯生成，~0.5 = 保留发型）。
  - id_photo_faceswap.json：模式 3b
      把参考人脸换到系统模板证件照上，发型与构图来自模板
      （img2img，denoise ~0.6）。

后处理链（BRIA 抠图 -> 白底 -> RealESRGAN -> 295x413）由两个变体共享。

用法：
  python3 workflow_scripts/generate_workflows.py [--template <file>] [--out <dir>]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS_PATH = ROOT / "workflow_scripts" / "assets.json"
DEFAULT_OUT = ROOT / "user" / "default" / "workflows"


def link(node_id, output=0):
    return [str(node_id), output]


def load_assets():
    with open(ASSETS_PATH) as f:
        return json.load(f)


# 此处用到的节点类型的前端 schema。每个条目把一个 class type 映射到
# 它的 widgets（按顺序存入 `widgets_values` 的值）、连接输入
# （name, type，按顺序）和输出（name, type，按顺序）。
NODE_SCHEMA = {
    "CheckpointLoaderSimple": {
        "widgets": ["ckpt_name"], "inputs": [],
        "outputs": [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")]},
    "LoadImage": {
        "widgets": ["image"], "inputs": [],
        "outputs": [("IMAGE", "IMAGE"), ("MASK", "MASK")]},
    "CLIPTextEncode": {
        "widgets": ["text"], "inputs": [("clip", "CLIP")],
        "outputs": [("CONDITIONING", "CONDITIONING")]},
    "InstantIDModelLoader": {
        "widgets": ["instantid_file"], "inputs": [],
        "outputs": [("INSTANTID", "INSTANTID")]},
    "InstantIDFaceAnalysis": {
        "widgets": ["provider"], "inputs": [],
        "outputs": [("FACEANALYSIS", "FACEANALYSIS")]},
    "ControlNetLoader": {
        "widgets": ["control_net_name"], "inputs": [],
        "outputs": [("CONTROL_NET", "CONTROL_NET")]},
    "ApplyInstantID": {
        "widgets": ["weight", "start_at", "end_at"],
        "inputs": [("instantid", "INSTANTID"), ("insightface", "FACEANALYSIS"),
                   ("control_net", "CONTROL_NET"), ("image", "IMAGE"),
                   ("model", "MODEL"), ("positive", "CONDITIONING"),
                   ("negative", "CONDITIONING"), ("image_kps", "IMAGE"),
                   ("mask", "MASK")],
        "outputs": [("MODEL", "MODEL"), ("positive", "CONDITIONING"),
                    ("negative", "CONDITIONING")]},
    "ImageResizeKJv2": {
        "widgets": ["width", "height", "upscale_method", "keep_proportion",
                    "pad_color", "crop_position", "divisible_by", "device"],
        "inputs": [("image", "IMAGE"), ("mask", "MASK")],
        "outputs": [("IMAGE", "IMAGE"), ("width", "INT"), ("height", "INT"),
                    ("mask", "MASK")]},
    "VAEEncode": {
        "widgets": [], "inputs": [("pixels", "IMAGE"), ("vae", "VAE")],
        "outputs": [("LATENT", "LATENT")]},
    "KSampler": {
        "widgets": ["seed", "control_after_generate", "steps", "cfg",
                    "sampler_name", "scheduler", "denoise"],
        "inputs": [("model", "MODEL"), ("positive", "CONDITIONING"),
                   ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
        "outputs": [("LATENT", "LATENT")]},
    "VAEDecode": {
        "widgets": [], "inputs": [("samples", "LATENT"), ("vae", "VAE")],
        "outputs": [("IMAGE", "IMAGE")]},
    "BRIA_RMBG_ModelLoader_Zho": {
        "widgets": [], "inputs": [],
        "outputs": [("rmbgmodel", "RMBGMODEL")]},
    "BRIA_RMBG_Zho": {
        "widgets": [], "inputs": [("rmbgmodel", "RMBGMODEL"), ("image", "IMAGE")],
        "outputs": [("image", "IMAGE"), ("mask", "MASK")]},
    "LayerUtility: ColorImage": {
        "widgets": ["width", "height", "color"], "inputs": [],
        "outputs": [("image", "IMAGE")]},
    "ImageCompositeMasked": {
        "widgets": ["x", "y", "resize_source"],
        "inputs": [("destination", "IMAGE"), ("source", "IMAGE"), ("mask", "MASK")],
        "outputs": [("IMAGE", "IMAGE")]},
    "UpscaleModelLoader": {
        "widgets": ["model_name"], "inputs": [],
        "outputs": [("UPSCALE_MODEL", "UPSCALE_MODEL")]},
    "ImageUpscaleWithModel": {
        "widgets": [], "inputs": [("upscale_model", "UPSCALE_MODEL"), ("image", "IMAGE")],
        "outputs": [("IMAGE", "IMAGE")]},
    "SaveImage": {
        "widgets": ["filename_prefix"], "inputs": [("images", "IMAGE")],
        "outputs": [("images", "IMAGE")]},
    "PreviewImage": {
        "widgets": [], "inputs": [("images", "IMAGE")],
        "outputs": [("images", "IMAGE")]},
}

# 画布节点尺寸，与前端为每种类型保存的尺寸保持一致。
# 图片相关节点（LoadImage/SaveImage/PreviewImage）在渲染缩略图/预览后
# 高度会大幅膨胀（官方模板中渲染后中位约 310x370 / 530x520），
# 这里直接按渲染后尺寸预留，避免布局时与相邻节点重叠。
NODE_SIZE = {
    "CheckpointLoaderSimple": [280, 128], "LoadImage": [280, 440],
    "CLIPTextEncode": [280, 84], "InstantIDModelLoader": [280, 84],
    "InstantIDFaceAnalysis": [280, 84], "ControlNetLoader": [280, 84],
    "ApplyInstantID": [280, 312], "ImageResizeKJv2": [280, 332],
    "VAEEncode": [280, 80], "KSampler": [280, 306], "VAEDecode": [280, 80],
    "BRIA_RMBG_ModelLoader_Zho": [280, 60], "BRIA_RMBG_Zho": [280, 80],
    "LayerUtility: ColorImage": [280, 136], "ImageCompositeMasked": [280, 180],
    "UpscaleModelLoader": [280, 84], "ImageUpscaleWithModel": [280, 80],
    "SaveImage": [280, 460], "PreviewImage": [280, 460],
}

LAYOUT_MARGIN = 40
NODE_GAP_X = 60
NODE_GAP_Y = 60
NOTE_SIZE = (400, 200)

# API 图省略的 widgets 的默认值（保存时由前端补齐）。
WIDGET_DEFAULTS = {"control_after_generate": "fixed", "device": "cpu"}

# 后处理链节点的语义化标题，按 postprocess_chain 内的相对顺序。
POST_CHAIN_TITLES = [
    "抠图模型", "抠图（移除背景）", "白色背景", "合成白底",
    "放大模型", "放大图像", "调整至 295×413", "保存证件照", "预览结果",
]


def to_ui_format(nodes, titles=None, note=None):
    """把 API 格式的工作流转换为前端 UI 格式，使其能在 Web 画布上打开。
    节点 id 保持不变；链接分配全新的 id。titles 为 {节点 id: 标题} 的可选映射。
    note 为可选的工作流功能说明文字，会作为 Note 节点放在画布右下区域。"""
    node_ids = sorted(nodes, key=int)
    links = []
    next_link = 1
    inputs_by_node = {}
    out_links_by_node = {nid: [] for nid in node_ids}

    # 第一遍：构建输入槽列表并创建链接。
    for node_id in node_ids:
        node = nodes[node_id]
        schema = NODE_SCHEMA[node["class_type"]]
        slots = []
        for name, typ in schema["inputs"]:
            val = node["inputs"].get(name)
            if isinstance(val, list) and len(val) == 2:
                src_id, src_slot = val
                src_type = NODE_SCHEMA[nodes[src_id]["class_type"]]["outputs"][src_slot][1]
                links.append([next_link, int(src_id), int(src_slot), int(node_id), len(slots), src_type])
                out_links_by_node[src_id].append((src_slot, next_link))
                slots.append((name, typ, next_link))
                next_link += 1
            else:
                slots.append((name, typ, None))
        inputs_by_node[node_id] = slots

    # 每个节点的依赖深度：决定从左到右布局中的列位置。
    # 深度是从源节点（无输入链接）出发的最长路径，因此流水线从左到右流动。
    depth = {}
    for node_id in node_ids:
        if node_id in depth:
            continue
        stack = [node_id]
        while stack:
            cur = stack[-1]
            parents = [str(links[l - 1][1]) for _, _, l in inputs_by_node[cur] if l is not None]
            pending = [p for p in parents if p not in depth]
            if pending:
                stack.extend(pending)
                continue
            depth[cur] = 1 + max((depth[p] for p in parents), default=0)
            stack.pop()

    # 布局：按依赖深度分列，列内节点从上到下依次排布。
    # 列内 y 坐标累加（上一节点底部 + 间距），列间 x 坐标累加
    # （上一列最大宽度 + 间距），因此任意高度的节点都不会重叠。
    # 有 note 时主流程整体右移，为左上角的说明卡片留出空间。
    cols = {}
    for node_id in node_ids:
        cols.setdefault(depth[node_id], []).append(node_id)
    pos = {}
    x = LAYOUT_MARGIN + (NOTE_SIZE[0] + NODE_GAP_X if note else 0)
    for d in sorted(cols):
        y = LAYOUT_MARGIN
        col_width = 0
        for node_id in cols[d]:
            pos[node_id] = (x, y)
            w, h = NODE_SIZE[nodes[node_id]["class_type"]]
            col_width = max(col_width, w)
            y += h + NODE_GAP_Y
        x += col_width + NODE_GAP_X

    ui_nodes = []
    for order, node_id in enumerate(node_ids):
        node = nodes[node_id]
        schema = NODE_SCHEMA[node["class_type"]]
        widgets = []
        for name in schema["widgets"]:
            val = node["inputs"].get(name)
            if val is None:
                val = WIDGET_DEFAULTS.get(name)
            widgets.append(val)
        inputs = [{"name": n, "type": t, "link": l} for n, t, l in inputs_by_node[node_id]]
        outputs = []
        for slot, (name, typ) in enumerate(schema["outputs"]):
            out_links = [l for s, l in out_links_by_node[node_id] if s == slot]
            outputs.append({"name": name, "type": typ, "links": out_links})
        ui_node = {
            "id": int(node_id),
            "type": node["class_type"],
            "pos": list(pos[node_id]),
            "size": NODE_SIZE[node["class_type"]],
            "flags": {},
            "order": order,
            "mode": 0,
            "inputs": inputs,
            "outputs": outputs,
            "properties": {"Node name for S&R": node["class_type"]},
            "widgets_values": widgets,
        }
        if titles and node_id in titles:
            ui_node["title"] = titles[node_id]
        ui_nodes.append(ui_node)

    if note:
        ui_nodes.append({
            "id": int(node_ids[-1]) + 1,
            "type": "Note",
            "pos": [LAYOUT_MARGIN, LAYOUT_MARGIN],
            "size": list(NOTE_SIZE),
            "flags": {},
            "order": len(ui_nodes),
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {},
            "widgets_values": [note],
        })

    return {
        "last_node_id": int(node_ids[-1]) + (1 if note else 0),
        "last_link_id": next_link - 1,
        "nodes": ui_nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def postprocess_chain(decoded_id):
    """BRIA 抠图 -> 白底 -> 放大 -> 295x413 -> 保存/预览。

    节点 id 从 VAEDecode 节点之后开始分配。
    返回从 VAEDecode 输出开始的 (nodes, save_id, preview_id)。
    """
    n = int(decoded_id) + 1
    nodes = {}
    nodes[str(n)] = {"class_type": "BRIA_RMBG_ModelLoader_Zho", "inputs": {}}
    nodes[str(n + 1)] = {"class_type": "BRIA_RMBG_Zho",
                         "inputs": {"rmbgmodel": link(str(n)), "image": link(decoded_id)}}
    nodes[str(n + 2)] = {"class_type": "LayerUtility: ColorImage",
                         "inputs": {"width": 640, "height": 640, "color": "#FFFFFF"}}
    nodes[str(n + 3)] = {"class_type": "ImageCompositeMasked",
                         "inputs": {"destination": link(str(n + 2)), "source": link(str(n + 1)),
                                    "x": 0, "y": 0, "resize_source": False,
                                    "mask": link(str(n + 1), 1)}}
    nodes[str(n + 4)] = {"class_type": "UpscaleModelLoader",
                         "inputs": {"model_name": "RealESRGAN_x2.pth"}}
    nodes[str(n + 5)] = {"class_type": "ImageUpscaleWithModel",
                         "inputs": {"upscale_model": link(str(n + 4)), "image": link(str(n + 3))}}
    nodes[str(n + 6)] = {"class_type": "ImageResizeKJv2",
                         "inputs": {"image": link(str(n + 5)), "width": 295, "height": 413,
                                    "upscale_method": "lanczos", "keep_proportion": "pad",
                                    "pad_color": "#FFFFFF", "crop_position": "center",
                                    "divisible_by": 1}}
    nodes[str(n + 7)] = {"class_type": "SaveImage",
                         "inputs": {"filename_prefix": "id_photo_", "images": link(str(n + 6))}}
    nodes[str(n + 8)] = {"class_type": "PreviewImage", "inputs": {"images": link(str(n + 6))}}
    return nodes, str(n + 7), str(n + 8)


def build_generate(assets):
    """模式 1/2/3a。参考图既是 InstantID 的身份来源，
    也是 img2img 的起点；denoise 决定具体模式。
    返回 (nodes, titles, note)。"""
    positive = assets["positive_prompt"] + ", " + assets["hairstyles"]["default"]
    nodes = {}
    nodes["1"] = {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": assets["checkpoint"]}}
    nodes["2"] = {"class_type": "LoadImage",
                  "inputs": {"image": assets["reference_image"]}}
    nodes["3"] = {"class_type": "CLIPTextEncode",
                  "inputs": {"text": positive, "clip": link("1", 1)}}
    nodes["4"] = {"class_type": "CLIPTextEncode",
                  "inputs": {"text": assets["negative_prompt"], "clip": link("1", 1)}}
    nodes["5"] = {"class_type": "InstantIDModelLoader",
                  "inputs": {"instantid_file": assets["instantid"]["ip_adapter"]}}
    nodes["6"] = {"class_type": "InstantIDFaceAnalysis",
                  "inputs": {"provider": assets["instantid"]["provider"]}}
    nodes["7"] = {"class_type": "ControlNetLoader",
                  "inputs": {"control_net_name": assets["instantid"]["controlnet"]}}
    nodes["8"] = {"class_type": "ApplyInstantID",
                  "inputs": {"instantid": link("5"), "insightface": link("6"),
                             "control_net": link("7"), "image": link("2"),
                             "model": link("1"), "positive": link("3"),
                             "negative": link("4"),
                             "weight": assets["instantid"]["weight"],
                             "start_at": assets["instantid"]["start_at"],
                             "end_at": assets["instantid"]["end_at"]}}
    nodes["9"] = {"class_type": "ImageResizeKJv2",
                  "inputs": {"image": link("2"), "width": 640, "height": 640,
                             "upscale_method": "lanczos", "keep_proportion": "pad",
                             "pad_color": "#FFFFFF", "crop_position": "center",
                             "divisible_by": 1}}
    nodes["10"] = {"class_type": "VAEEncode",
                   "inputs": {"pixels": link("9"), "vae": link("1", 2)}}
    nodes["11"] = {"class_type": "KSampler",
                   "inputs": {"model": link("8"), "positive": link("8", 1),
                              "negative": link("8", 2), "latent_image": link("10"),
                              "seed": assets["sampler"]["seed"],
                              "steps": assets["sampler"]["steps"],
                              "cfg": assets["sampler"]["cfg"],
                              "sampler_name": assets["sampler"]["sampler_name"],
                              "scheduler": assets["sampler"]["scheduler"],
                              "denoise": 1.0}}
    nodes["12"] = {"class_type": "VAEDecode",
                   "inputs": {"samples": link("11"), "vae": link("1", 2)}}
    post, _, _ = postprocess_chain("12")
    nodes.update(post)

    titles = {
        "1": "基础模型", "2": "参考照片（身份来源）",
        "3": "正向提示词", "4": "负向提示词",
        "5": "InstantID 模型", "6": "人脸分析", "7": "ControlNet 模型",
        "8": "应用 InstantID", "9": "缩放画布 640×640",
        "10": "编码到潜空间", "11": "采样器", "12": "解码图像",
    }
    for i, t in enumerate(POST_CHAIN_TITLES):
        titles[str(13 + i)] = t
    note = (
        "身份证证件照生成（模式 1/2/3a）\n"
        "- 参考照片（左上）既提供人脸身份，也作为画面起点\n"
        "- 正向/负向提示词与 KSampler 的 denoise 决定相似度："
        "denoise=1 全新照片，低值更接近参考\n"
        "- InstantID 强度由 weight 控制；输出经过抠图、白底合成与 295×413 裁剪"
    )
    return nodes, titles, note


def build_faceswap(assets, template_file):
    """模式 3b。模板提供 latent 起点（发型/构图），
    参考图通过 InstantID 提供身份。
    返回 (nodes, titles, note)。"""
    nodes = {}
    nodes["1"] = {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": assets["checkpoint"]}}
    nodes["2"] = {"class_type": "LoadImage",
                  "inputs": {"image": template_file}}
    nodes["3"] = {"class_type": "LoadImage",
                  "inputs": {"image": assets["reference_image"]}}
    nodes["4"] = {"class_type": "CLIPTextEncode",
                  "inputs": {"text": assets["positive_prompt"], "clip": link("1", 1)}}
    nodes["5"] = {"class_type": "CLIPTextEncode",
                  "inputs": {"text": assets["negative_prompt"], "clip": link("1", 1)}}
    nodes["6"] = {"class_type": "InstantIDModelLoader",
                  "inputs": {"instantid_file": assets["instantid"]["ip_adapter"]}}
    nodes["7"] = {"class_type": "InstantIDFaceAnalysis",
                  "inputs": {"provider": assets["instantid"]["provider"]}}
    nodes["8"] = {"class_type": "ControlNetLoader",
                  "inputs": {"control_net_name": assets["instantid"]["controlnet"]}}
    nodes["9"] = {"class_type": "ApplyInstantID",
                  "inputs": {"instantid": link("6"), "insightface": link("7"),
                             "control_net": link("8"), "image": link("3"),
                             "model": link("1"), "positive": link("4"),
                             "negative": link("5"),
                             "weight": assets["instantid"]["weight"],
                             "start_at": assets["instantid"]["start_at"],
                             "end_at": assets["instantid"]["end_at"]}}
    nodes["10"] = {"class_type": "ImageResizeKJv2",
                   "inputs": {"image": link("2"), "width": 640, "height": 640,
                              "upscale_method": "lanczos", "keep_proportion": "pad",
                              "pad_color": "#FFFFFF", "crop_position": "center",
                              "divisible_by": 1}}
    nodes["11"] = {"class_type": "VAEEncode",
                   "inputs": {"pixels": link("10"), "vae": link("1", 2)}}
    nodes["12"] = {"class_type": "KSampler",
                   "inputs": {"model": link("9"), "positive": link("9", 1),
                              "negative": link("9", 2), "latent_image": link("11"),
                              "seed": assets["sampler"]["seed"],
                              "steps": assets["sampler"]["steps"],
                              "cfg": assets["sampler"]["cfg"],
                              "sampler_name": assets["sampler"]["sampler_name"],
                              "scheduler": assets["sampler"]["scheduler"],
                              "denoise": 0.6}}
    nodes["13"] = {"class_type": "VAEDecode",
                   "inputs": {"samples": link("12"), "vae": link("1", 2)}}
    post, _, _ = postprocess_chain("13")
    nodes.update(post)

    titles = {
        "1": "基础模型", "2": "证件照模板（发型/构图）",
        "3": "参考照片（身份来源）",
        "4": "正向提示词", "5": "负向提示词",
        "6": "InstantID 模型", "7": "人脸分析", "8": "ControlNet 模型",
        "9": "应用 InstantID", "10": "缩放画布 640×640",
        "11": "编码到潜空间", "12": "采样器", "13": "解码图像",
    }
    for i, t in enumerate(POST_CHAIN_TITLES):
        titles[str(14 + i)] = t
    note = (
        "证件照换脸（模式 3b）\n"
        "- 证件照模板（左上）提供发型/构图，作为生成起点\n"
        "- 参考照片（左下）仅提供人脸身份（InstantID）\n"
        "- KSampler 的 denoise=0.6 固定：保留模板构图、换入参考人脸\n"
        "- 输出经过抠图、白底合成与 295×413 裁剪"
    )
    return nodes, titles, note


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", default=None,
                        help="template file for the faceswap workflow (default: first in assets.json)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    args = parser.parse_args()

    assets = load_assets()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    generate, generate_titles, generate_note = build_generate(assets)
    with open(out_dir / "id_photo_generate.json", "w") as f:
        json.dump(to_ui_format(generate, generate_titles, generate_note), f, indent=2)

    template = args.template or assets["templates"]["files"][0]
    template_dir = assets["templates"]["dir"]
    template_path = ROOT / "input" / template_dir / template
    if not template_path.exists():
        print(f"warning: template not found: {template_path}", file=sys.stderr)
        print(f"  faceswap workflow will not run until you place a template there.", file=sys.stderr)
    # LoadImage 相对于 input/ 解析文件；子目录文件使用标准的
    # "subdir/name.ext" 路径形式引用。
    faceswap, faceswap_titles, faceswap_note = build_faceswap(assets, f"{template_dir}/{template}" if template_dir else template)
    with open(out_dir / "id_photo_faceswap.json", "w") as f:
        json.dump(to_ui_format(faceswap, faceswap_titles, faceswap_note), f, indent=2)

    print(f"wrote {out_dir / 'id_photo_generate.json'} ({len(generate)} nodes)")
    print(f"wrote {out_dir / 'id_photo_faceswap.json'} ({len(faceswap)} nodes)")


if __name__ == "__main__":
    main()
