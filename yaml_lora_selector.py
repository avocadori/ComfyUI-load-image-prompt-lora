import os
import yaml
import folder_paths

class YAMLLoRASelector:
    """
    YAMLファイルからLoRA情報を読み込み、WanVideo Lora Selectと完全互換の形式で出力するノード
    """

    # ComfyUI メタ情報 --------------------------
    CATEGORY      = "Loaders"
    FUNCTION      = "execute"
    RETURN_TYPES  = ("STRING", "FLOAT", "STRING", "FLOAT", "STRING", "FLOAT")
    RETURN_NAMES  = ("lora1", "strength1", "lora2", "strength2", "lora3", "strength3")

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

    def _get_available_loras(self):
        """利用可能なLoRAファイルのリストを取得"""
        try:
            lora_files = folder_paths.get_filename_list("loras")
            return lora_files
        except Exception as e:
            print(f"[YAMLLoRASelector] LoRAファイル取得エラー: {e}")
            return []

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

    def _find_matching_lora_file(self, lora_name, available_loras):
        """LoRA名に一致するファイルを検索"""
        if not lora_name:
            return ""
        
        # 完全一致をチェック（拡張子付き）
        for lora_file in available_loras:
            if lora_file == f"{lora_name}.safetensors":
                return lora_file
            elif lora_file == f"{lora_name}.ckpt":
                return lora_file
            elif lora_file == f"{lora_name}.pt":
                return lora_file
            elif lora_file == f"{lora_name}.pth":
                return lora_file
        
        # 拡張子なしでの完全一致
        for lora_file in available_loras:
            file_name_without_ext = os.path.splitext(lora_file)[0]
            if file_name_without_ext == lora_name:
                print(f"[YAMLLoRASelector] LoRAファイル発見: {lora_name} -> {lora_file}")
                return lora_file
        
        # 部分一致をチェック（大文字小文字を無視）
        lora_name_lower = lora_name.lower()
        for lora_file in available_loras:
            file_name_without_ext = os.path.splitext(lora_file)[0].lower()
            if file_name_without_ext == lora_name_lower:
                print(f"[YAMLLoRASelector] LoRAファイル発見（大文字小文字修正）: {lora_name} -> {lora_file}")
                return lora_file
        
        # 見つからない場合は警告を出力して空文字を返す
        print(f"[YAMLLoRASelector] 警告: LoRA '{lora_name}' に対応するファイルが見つかりません")
        print(f"[YAMLLoRASelector] 利用可能なLoRA: {available_loras[:10]}...")
        return ""

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

            # 利用可能なLoRAファイルを取得
            available_loras = self._get_available_loras()

            # カテゴリ設定を取得
            category_config = cfg[category]
            
            # LoRA情報を解析
            lora1_raw = category_config.get("lora1", "")
            lora2_raw = category_config.get("lora2", "")
            lora3_raw = category_config.get("lora3", "")
            
            lora1_name, lora1_weight = self._parse_lora_string(lora1_raw)
            lora2_name, lora2_weight = self._parse_lora_string(lora2_raw)
            lora3_name, lora3_weight = self._parse_lora_string(lora3_raw)

            # LoRAファイル名を検索
            lora1_file = self._find_matching_lora_file(lora1_name, available_loras)
            lora2_file = self._find_matching_lora_file(lora2_name, available_loras)
            lora3_file = self._find_matching_lora_file(lora3_name, available_loras)

            print(f"[YAMLLoRASelector] カテゴリ: {category}")
            print(f"[YAMLLoRASelector] LoRA1: {lora1_file} (重み: {lora1_weight})")
            print(f"[YAMLLoRASelector] LoRA2: {lora2_file} (重み: {lora2_weight})")
            print(f"[YAMLLoRASelector] LoRA3: {lora3_file} (重み: {lora3_weight})")

            return (lora1_file, lora1_weight, lora2_file, lora2_weight, lora3_file, lora3_weight)

        except Exception as e:
            print(f"[YAMLLoRASelector] エラー: {e}")
            raise e
