"""证件照规格节点。

下拉选择国家/尺寸/背景色，输出目标 width/height/color，
驱动下游 ColorImage（背景色）与 ImageResizeKJv2（最终尺寸 + pad 填充色）。

规格数据与工作流生成器共享 workflow_scripts/id_photo_specs.json。
首版仅加载第一个国家（中国）的规格；后续多国支持可改为
联动下拉或按所选国家动态加载。
"""

import json
from pathlib import Path

SPECS_PATH = Path(__file__).resolve().parent.parent.parent / "workflow_scripts" / "id_photo_specs.json"


def _load_specs():
    with open(SPECS_PATH, encoding="utf-8") as f:
        return json.load(f)


_SPECS = _load_specs()
_COUNTRIES = list(_SPECS["countries"].keys())
_DEFAULT_COUNTRY = _COUNTRIES[0]
_SIZES = list(_SPECS["countries"][_DEFAULT_COUNTRY]["sizes"].keys())
_BACKGROUNDS = list(_SPECS["countries"][_DEFAULT_COUNTRY]["backgrounds"].keys())


class IDPhotoSpec:
    """证件照规格：国家/尺寸/背景色下拉，输出 width/height/color。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "country": (_COUNTRIES,),
                "size": (_SIZES,),
                "background": (_BACKGROUNDS,),
            },
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "color")
    FUNCTION = "get_spec"
    CATEGORY = "证件照"

    def get_spec(self, country, size, background):
        specs = _SPECS["countries"][country]
        width, height = specs["sizes"][size]["px"]
        color = specs["backgrounds"][background]["hex"]
        return (width, height, color)


NODE_CLASS_MAPPINGS = {"IDPhotoSpec": IDPhotoSpec}
NODE_DISPLAY_NAME_MAPPINGS = {"IDPhotoSpec": "证件照规格"}

# js/ 目录下的前端扩展会通过 /extensions/ComfyUI-IDPhotoSpec/ 加载。
WEB_DIRECTORY = "./js"
