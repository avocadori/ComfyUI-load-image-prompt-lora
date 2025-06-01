from .yaml_image_cycler import YAMLImageCycler
from .yaml_lora_extractor import YAMLLoRAExtractor
from .yaml_image_cycler_simple import YAMLImageCyclerSimple

NODE_CLASS_MAPPINGS = {
    "YAMLImageCycler": YAMLImageCycler,
    "YAMLLoRAExtractor": YAMLLoRAExtractor,
    "YAMLImageCyclerSimple": YAMLImageCyclerSimple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YAMLImageCycler": "YAML Image Cycler (Full)",
    "YAMLLoRAExtractor": "YAML LoRA Extractor",
    "YAMLImageCyclerSimple": "YAML Image Cycler (Simple)"
}
