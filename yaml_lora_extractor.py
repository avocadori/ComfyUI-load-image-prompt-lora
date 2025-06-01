import os
import yaml

class YAMLLoRAExtractor:
    """
    YAMLファイルからLoRA情報を抽出する専用ノード
    """

    # ComfyUI メタ情報 --------------------------
    CATEGORY      = "Loaders"
    FUNCTION      = "execute"
    RETURN_TYPES  = ("STRING", "STRING", "STRING", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES  = ("prompt", "lora1_name", "lora2_name", "lora3_name", "lora1_weight", "lora2_weight", "lora3_weight")

    def __init__(self):
        self._yaml_buf = {}    # キャッシュした YAML dict

    @classmethod
    def INPUT_TYPES(cls):
        """
        yaml_path と category を設定
        """
        default_yaml = "setting.yaml"
        try:
            if os.path.exists(default_yaml):
                keys = list(cls._peek_yaml_keys(default_yaml))
            else:
                keys = []
        except Exception:
            keys = []

        return {
            "required": {
                "yaml_path": ("STRING", {"default": default_yaml}),
                "category": (keys if keys else "STRING", )
            }
        }

    # ───────────────────────────────────────────
    # 内部ヘルパ
    # ───────────────────────────────────────────
    @staticmethod
    def _peek_yaml_keys(path):
        """YAMLファイルのキーを取得"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.keys() if data else []

    def _load_yaml(self, path):
        """YAMLファイルをキャッシュ付きで読み込み"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"YAMLファイルが見つかりません: {path}")
        
        if path not in self._yaml_buf:
            with open(path, "r", encoding="utf-8") as f:
                self._yaml_buf[path] = yaml.safe_load(f)
        return self._yaml_buf[path]

    def _parse_lora_string(self, lora_string):
        """LoRA文字列を解析して名前と重みを抽出
        例: '<lora:character1_v1:0.8>' -> ('character1_v1', 0.8)
        """
        if not lora_string or not isinstance(lora_string, str):
            return "", 1.0
        
        # <lora:name:weight> 形式の場合
        if lora_string.startswith("<lora:") and lora_string.endswith(">"):
            # <lora: と > を除去
            content = lora_string[6:-1]
            # : で分割
            parts = content.split(":")
            if len(parts) >= 2:
                name = parts[0].strip()
                try:
                    weight = float(parts[1].strip())
                except ValueError:
                    weight = 1.0
                return name, weight
            elif len(parts) == 1:
                return parts[0].strip(), 1.0
        
        # 通常の文字列の場合は名前のみ
        return str(lora_string).strip(), 1.0

    # ───────────────────────────────────────────
    # メイン処理
    # ───────────────────────────────────────────
    def execute(self, yaml_path: str, category: str):
        """メイン実行関数"""
        try:
            # YAML設定を読み込み
            cfg = self._load_yaml(yaml_path)

            if category not in cfg:
                available_categories = list(cfg.keys()) if cfg else []
                raise ValueError(
                    f"YAML にカテゴリ「{category}」が見つかりません。"
                    f"利用可能なカテゴリ: {available_categories}"
                )

            # カテゴリ設定を取得
            category_config = cfg[category]
            
            # プロンプトを取得
            prompt = str(category_config.get("prompt", ""))
            
            # LoRA情報を解析
            lora1_raw = category_config.get("lora1", "")
            lora2_raw = category_config.get("lora2", "")
            lora3_raw = category_config.get("lora3", "")
            
            lora1_name, lora1_weight = self._parse_lora_string(lora1_raw)
            lora2_name, lora2_weight = self._parse_lora_string(lora2_raw)
            lora3_name, lora3_weight = self._parse_lora_string(lora3_raw)

            print(f"[YAMLLoRAExtractor] カテゴリ: {category}")
            print(f"[YAMLLoRAExtractor] プロンプト: {prompt}")
            print(f"[YAMLLoRAExtractor] LoRA1: {lora1_name} (重み: {lora1_weight})")
            print(f"[YAMLLoRAExtractor] LoRA2: {lora2_name} (重み: {lora2_weight})")
            print(f"[YAMLLoRAExtractor] LoRA3: {lora3_name} (重み: {lora3_weight})")

            return (prompt, lora1_name, lora2_name, lora3_name, lora1_weight, lora2_weight, lora3_weight)

        except Exception as e:
            print(f"[YAMLLoRAExtractor] エラー: {e}")
            raise e
