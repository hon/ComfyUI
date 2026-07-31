#!/usr/bin/env python3
"""Generate Chinese passport ID photo workflows (API format).

Two variants:
  - id_photo_generate.json: modes 1/2/3a
      mode 1/2: hairstyle controlled by the prompt (default preset or user text)
      mode 3a:  keep the reference photo's hairstyle (img2img denoise ~0.5)
    Both share one graph: the reference image is the img2img start point and
    denoise is the mode switch (1.0 = pure generation, ~0.5 = keep hairstyle).
  - id_photo_faceswap.json: mode 3b
      swap the reference face onto a system template ID photo, hairstyle and
      composition come from the template (img2img denoise ~0.6).

The post-process chain (BRIA matting -> white bg -> RealESRGAN -> 295x413)
is shared by both variants.

Usage:
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


def postprocess_chain(decoded_id):
    """BRIA matting -> white background -> upscale -> 295x413 -> save/preview.

    Node ids are allocated starting right after the VAEDecode node.
    Returns (nodes, save_id, preview_id) starting from the VAEDecode output.
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
    """Modes 1/2/3a. Reference image is both the InstantID identity source
    and the img2img start point; denoise selects the mode."""
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
    return nodes


def build_faceswap(assets, template_file):
    """Mode 3b. Template provides latent start (hairstyle/composition),
    reference image provides the identity via InstantID."""
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
    return nodes


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

    generate = build_generate(assets)
    with open(out_dir / "id_photo_generate.json", "w") as f:
        json.dump(generate, f, indent=2)

    template = args.template or assets["templates"]["files"][0]
    template_path = ROOT / "input" / assets["templates"]["dir"] / template
    if not template_path.exists():
        print(f"warning: template not found: {template_path}", file=sys.stderr)
        print(f"  faceswap workflow will not run until you place a template there.", file=sys.stderr)
    faceswap = build_faceswap(assets, template)
    with open(out_dir / "id_photo_faceswap.json", "w") as f:
        json.dump(faceswap, f, indent=2)

    print(f"wrote {out_dir / 'id_photo_generate.json'} ({len(generate)} nodes)")
    print(f"wrote {out_dir / 'id_photo_faceswap.json'} ({len(faceswap)} nodes)")


if __name__ == "__main__":
    main()
