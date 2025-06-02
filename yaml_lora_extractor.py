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
    RETURN_TYPES  = ("STRING", "STRING", "STRING", "STRING", "FLOAT", "FLOAT", "FLOAT") # loraX_name を STRING に変更
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
                "category": (keys if keys else "STRING", ),
                "raw_lora_names": ("BOOLEAN", {"default": False}), # 新しい入力パラメータ
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
        """利用可能なLoRAファイルのリストと名前のマッピングを取得"""
        try:
            lora_files_with_ext = folder_paths.get_filename_list("loras") # 例: ["subdir/loraA.safetensors", "loraB.pt"]
            
            # 拡張子なしの名前をキー、拡張子付きの元のファイル名を値とする辞書
            # 例: {"subdir/loraA": "subdir/loraA.safetensors", "loraB": "loraB.pt"}
            # 注意: 同じ拡張子なし名で異なる拡張子のファイルが存在する場合、後のもので上書きされる
            name_map = {os.path.splitext(f)[0]: f for f in lora_files_with_ext}
            
            # 拡張子なしの名前のリスト (従来通り)
            lora_names_no_ext = list(name_map.keys())
            
            return lora_names_no_ext, name_map, lora_files_with_ext
        except Exception as e:
            print(f"[YAMLLoRAExtractor] LoRAファイル取得エラー: {e}")
            return [], {}, []

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

    def _validate_lora_name(self, lora_name_from_yaml, available_loras_no_ext, lora_name_map, available_loras_with_ext, raw_mode):
        """LoRA名が利用可能なLoRAリストに存在するかチェックし、適切なファイル名を返す"""
        if not lora_name_from_yaml:
            return None

        if raw_mode:
            # RAWモードの場合: YAMLの記述をそのまま使い、拡張子付きリストと照合
            # <lora:NAME:WEIGHT> から抽出されたNAME部分が lora_name_from_yaml に入る
            
            # まず、lora_name_from_yaml が拡張子付きリストにそのまま存在するか確認
            if lora_name_from_yaml in available_loras_with_ext:
                return lora_name_from_yaml
            
            # 次に、lora_name_from_yaml の拡張子を除去したものが、拡張子なしマップのキーに存在するか確認し、
            # 存在すれば対応する拡張子付きファイル名を返す
            name_no_ext_yaml, _ = os.path.splitext(lora_name_from_yaml)
            if name_no_ext_yaml in lora_name_map: # lora_name_mapのキーは拡張子なし
                return lora_name_map[name_no_ext_yaml]

            # それでも見つからない場合は警告
            print(f"[YAMLLoRAExtractor] (RAW Mode) 警告: LoRA '{lora_name_from_yaml}' が利用可能なLoRAリストに見つかりません。")
            print(f"[YAMLLoRAExtractor] 利用可能なLoRA (全{len(available_loras_with_ext)}件): {available_loras_with_ext}")
            return lora_name_from_yaml # 元の記述を返す
        else:
            # 通常モード (Validated Mode)
            # YAMLの記述が拡張子を含んでいてもいなくても、まず拡張子なしの名前で処理
            name_no_ext_yaml, _ = os.path.splitext(lora_name_from_yaml)
            processed_name_no_ext = name_no_ext_yaml 

            # 拡張子なしの名前が lora_name_map のキーに存在するか (完全一致)
            if processed_name_no_ext in lora_name_map: # lora_name_mapのキーは拡張子なし
                return lora_name_map[processed_name_no_ext] # 対応する拡張子付きファイル名を返す
            
            # 大文字小文字を無視して比較
            processed_name_no_ext_lower = processed_name_no_ext.lower()
            # available_loras_no_ext は lora_name_map のキーセットと同じはず
            for name_no_ext_available in available_loras_no_ext: 
                if name_no_ext_available.lower() == processed_name_no_ext_lower:
                    print(f"[YAMLLoRAExtractor] LoRA名を修正: 元の名前 '{lora_name_from_yaml}' -> 利用可能な名前 '{lora_name_map[name_no_ext_available]}'")
                    return lora_name_map[name_no_ext_available] # 対応する拡張子付きファイル名を返す
            
            print(f"[YAMLLoRAExtractor] (Validated Mode) 警告: LoRA '{lora_name_from_yaml}' (処理後拡張子なし: '{processed_name_no_ext}') が見つかりません")
            print(f"[YAMLLoRAExtractor] 利用可能なLoRA (全{len(available_loras_with_ext)}件): {available_loras_with_ext}")
            return lora_name_from_yaml # 元の記述を返す

    # ───────────────────────────────────────────
    # メイン処理
    # ───────────────────────────────────────────
    def execute(self, yaml_path: str, category: str, raw_lora_names: bool): # 新しいパラメータを追加
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
            available_loras_no_ext, lora_name_map, available_loras_with_ext = self._get_available_loras()

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
            lora1_name = self._validate_lora_name(lora1_name, available_loras_no_ext, lora_name_map, available_loras_with_ext, raw_lora_names)
            lora2_name = self._validate_lora_name(lora2_name, available_loras_no_ext, lora_name_map, available_loras_with_ext, raw_lora_names)
            lora3_name = self._validate_lora_name(lora3_name, available_loras_no_ext, lora_name_map, available_loras_with_ext, raw_lora_names)

            print(f"[YAMLLoRAExtractor] カテゴリ: {category}")
            print(f"[YAMLLoRAExtractor] LoRA名処理モード: {'Raw' if raw_lora_names else 'Validated'}")
            print(f"[YAMLLoRAExtractor] プロンプト: {prompt}")
            print(f"[YAMLLoRAExtractor] LoRA1: {lora1_name} (重み: {lora1_weight})")
            print(f"[YAMLLoRAExtractor] LoRA2: {lora2_name} (重み: {lora2_weight})")
            print(f"[YAMLLoRAExtractor] LoRA3: {lora3_name} (重み: {lora3_weight})")

            return (prompt, lora1_name, lora2_name, lora3_name, lora1_weight, lora2_weight, lora3_weight)

        except Exception as e:
            print(f"[YAMLLoRAExtractor] エラー: {e}")
            raise e
