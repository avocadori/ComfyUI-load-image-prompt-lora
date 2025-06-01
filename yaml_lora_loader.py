import os
import yaml
import folder_paths

class YAMLLoRALoader:
    """
    YAMLファイルからLoRA情報を読み込み、ComfyUI標準のLoRALoader形式で出力するノード
    WanVideo Lora Selectとの完全互換性を目指す
    """

    # ComfyUI メタ情報 --------------------------
    CATEGORY      = "Loaders"
    FUNCTION      = "execute"
    RETURN_TYPES  = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("prompt", "lora1", "lora2", "lora3")

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

        # 利用可能なLoRAファイルを取得
        try:
            lora_list = folder_paths.get_filename_list("loras")
            lora_names = [os.path.splitext(f)[0] for f in lora_list]
        except:
            lora_names = []

        return {
            "required": {
                "yaml_path": ("STRING", {"default": default_yaml}),
                "category": (keys if keys else "STRING", )
            },
            "optional": {
                "lora1_override": (["None"] + lora_names, {"default": "None"}),
                "lora2_override": (["None"] + lora_names, {"default": "None"}),
                "lora3_override": (["None"] + lora_names, {"default": "None"}),
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

    def _get_available_loras(self):
        """利用可能なLoRAファイルのリストを取得"""
        try:
            lora_files = folder_paths.get_filename_list("loras")
            lora_names = [os.path.splitext(f)[0] for f in lora_files]
            return lora_names
        except Exception as e:
            print(f"[YAMLLoRALoader] LoRAファイル取得エラー: {e}")
            return []

    def _parse_lora_string(self, lora_string):
        """LoRA文字列を解析して名前を抽出
        例: '<lora:character1_v1:0.8>' -> 'character1_v1'
        """
        if not lora_string or not isinstance(lora_string, str):
            return ""
        
        # <lora:name:weight> 形式の場合
        if lora_string.startswith("<lora:") and lora_string.endswith(">"):
            # <lora: と > を除去
            content = lora_string[6:-1]
            # : で分割して名前部分を取得
            parts = content.split(":")
            if len(parts) >= 1:
                return parts[0].strip()
        
        # 通常の文字列の場合はそのまま返す
        return str(lora_string).strip()

    def _validate_lora_name(self, lora_name, available_loras):
        """LoRA名が利用可能なLoRAリストに存在するかチェック"""
        if not lora_name:
            return "None"
        
        # 完全一致をチェック
        if lora_name in available_loras:
            return lora_name
        
        # 部分一致をチェック（大文字小文字を無視）
        lora_name_lower = lora_name.lower()
        for available in available_loras:
            if available.lower() == lora_name_lower:
                print(f"[YAMLLoRALoader] LoRA名を修正: {lora_name} -> {available}")
                return available
        
        # 見つからない場合は警告を出力して"None"を返す
        print(f"[YAMLLoRALoader] 警告: LoRA '{lora_name}' が見つかりません")
        print(f"[YAMLLoRALoader] 利用可能なLoRA: {available_loras[:10]}...")
        return "None"

    # ───────────────────────────────────────────
    # メイン処理
    # ───────────────────────────────────────────
    def execute(self, yaml_path: str, category: str, lora1_override="None", lora2_override="None", lora3_override="None"):
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

            # 利用可能なLoRAファイルを取得
            available_loras = self._get_available_loras()

            # カテゴリ設定を取得
            category_config = cfg[category]
            
            # プロンプトを取得
            prompt = str(category_config.get("prompt", ""))
            
            # LoRA情報を解析
            if lora1_override != "None":
                lora1_name = lora1_override
            else:
                lora1_raw = category_config.get("lora1", "")
                lora1_name = self._parse_lora_string(lora1_raw)
                lora1_name = self._validate_lora_name(lora1_name, available_loras)

            if lora2_override != "None":
                lora2_name = lora2_override
            else:
                lora2_raw = category_config.get("lora2", "")
                lora2_name = self._parse_lora_string(lora2_raw)
                lora2_name = self._validate_lora_name(lora2_name, available_loras)

            if lora3_override != "None":
                lora3_name = lora3_override
            else:
                lora3_raw = category_config.get("lora3", "")
                lora3_name = self._parse_lora_string(lora3_raw)
                lora3_name = self._validate_lora_name(lora3_name, available_loras)

            print(f"[YAMLLoRALoader] カテゴリ: {category}")
            print(f"[YAMLLoRALoader] プロンプト: {prompt}")
            print(f"[YAMLLoRALoader] LoRA1: {lora1_name}")
            print(f"[YAMLLoRALoader] LoRA2: {lora2_name}")
            print(f"[YAMLLoRALoader] LoRA3: {lora3_name}")

            return (prompt, lora1_name, lora2_name, lora3_name)

        except Exception as e:
            print(f"[YAMLLoRALoader] エラー: {e}")
            raise e
