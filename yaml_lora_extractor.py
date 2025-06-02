import os
import yaml
import folder_paths

class YAMLLoRAExtractor:
    """
    YAMLファイルからLoRA情報を抽出する専用ノード
    WanVideo Lora Selectとの互換性を重視
    """

    # ComfyUI メタ情報 --------------------------
    CATEGORY      = "Loaders"
    FUNCTION      = "execute"
    RETURN_TYPES  = ("STRING", "*", "*", "*", "FLOAT", "FLOAT", "FLOAT")
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

    def _get_available_loras(self):
        """利用可能なLoRAファイルのリストを取得"""
        try:
            # ComfyUIのfolder_pathsを使用してLoRAファイルを取得
            lora_files = folder_paths.get_filename_list("loras")
            # 拡張子を除去してLoRA名のみを返す
            lora_names = [os.path.splitext(f)[0] for f in lora_files]
            return lora_names
        except Exception as e:
            print(f"[YAMLLoRAExtractor] LoRAファイル取得エラー: {e}")
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

    def _validate_lora_name(self, lora_name, available_loras):
        """LoRA名が利用可能なLoRAリストに存在するかチェック"""
        if not lora_name:
            return None

        # lora_name から拡張子を除去 (例: .safetensors, .pt など)
        name_without_ext, ext = os.path.splitext(lora_name)
        processed_lora_name = name_without_ext if ext.lower() in [".safetensors", ".pt", ".ckpt", ".bin"] else lora_name
        
        # 完全一致をチェック (拡張子なしの名前で比較)
        if processed_lora_name in available_loras:
            return processed_lora_name # 拡張子なしの正しい名前を返す
        
        # 部分一致をチェック（大文字小文字を無視、拡張子なしの名前で比較）
        processed_lora_name_lower = processed_lora_name.lower()
        for available in available_loras: # available_loras は既に拡張子なし
            if available.lower() == processed_lora_name_lower:
                print(f"[YAMLLoRAExtractor] LoRA名を修正: 元の名前 '{lora_name}' -> 利用可能な名前 '{available}'")
                return available # 拡張子なしの正しい名前を返す
        
        # 見つからない場合は警告を出力してそのまま返す (元の名前を返す)
        print(f"[YAMLLoRAExtractor] 警告: LoRA '{lora_name}' (処理後: '{processed_lora_name}') が見つかりません")
        print(f"[YAMLLoRAExtractor] 利用可能なLoRA (先頭10件): {available_loras[:10]}...")
        return lora_name # 元のlora_nameを返す (下流の処理でエラーになるかもしれないが、ここでは加工しない)

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
            
            # プロンプトを取得
            prompt = str(category_config.get("prompt", ""))
            
            # LoRA情報を解析
            lora1_raw = category_config.get("lora1", "")
            lora2_raw = category_config.get("lora2", "")
            lora3_raw = category_config.get("lora3", "")
            
            lora1_name, lora1_weight = self._parse_lora_string(lora1_raw)
            lora2_name, lora2_weight = self._parse_lora_string(lora2_raw)
            lora3_name, lora3_weight = self._parse_lora_string(lora3_raw)

            # LoRA名を検証・修正
            lora1_name = self._validate_lora_name(lora1_name, available_loras)
            lora2_name = self._validate_lora_name(lora2_name, available_loras)
            lora3_name = self._validate_lora_name(lora3_name, available_loras)

            print(f"[YAMLLoRAExtractor] カテゴリ: {category}")
            print(f"[YAMLLoRAExtractor] プロンプト: {prompt}")
            print(f"[YAMLLoRAExtractor] LoRA1: {lora1_name} (重み: {lora1_weight})")
            print(f"[YAMLLoRAExtractor] LoRA2: {lora2_name} (重み: {lora2_weight})")
            print(f"[YAMLLoRAExtractor] LoRA3: {lora3_name} (重み: {lora3_weight})")

            return (prompt, lora1_name, lora2_name, lora3_name, lora1_weight, lora2_weight, lora3_weight)

        except Exception as e:
            print(f"[YAMLLoRAExtractor] エラー: {e}")
            raise e
