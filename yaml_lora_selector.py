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
    RETURN_TYPES  = ("WANVIDLORA", "STRING") # WanVideoLoraSelect の prev_lora と互換性のある型
    RETURN_NAMES  = ("wanvid_lora", "yaml_path")

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
                "category": (keys if keys else ["STRING"], )
            },
            "optional": {
                "yaml_path": ("STRING", {"default": default_yaml})
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
            return "None"  # LoRA名がない場合は "None" を返す
        
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
        
        # 見つからない場合
        if not available_loras:
            print(f"[YAMLLoRASelector] エラー: 利用可能なLoRAファイルがシステムに見つかりません。LoRA '{lora_name}' を読み込めませんでした。")
            return "None"
        else:
            # 利用可能なLoRAはあるが、指定されたlora_nameが見つからない場合
            print(f"[YAMLLoRASelector] 警告: LoRA '{lora_name}' に一致するファイルが見つかりませんでした。代わりに \"None\" を使用します。")
            return "None" # 指定されたLoRAが見つからない場合は "None" を返す

    # ───────────────────────────────────────────
    # メイン処理
    # ───────────────────────────────────────────
    def execute(self, category: str, yaml_path: str = "setting.yaml"):
        """メイン実行関数"""
        output_loras_list = []
        # エラー時のデフォルト返り値 (WANVIDLORA, STRING)
        default_wanvidlora_item = {
            "path": "None", 
            "strength": 1.0, 
            "name": "None", 
            "blocks": {}, 
            "layer_filter": "", 
            "low_mem_load": False
        }
        error_return_value = ([default_wanvidlora_item], yaml_path)


        try:
            # YAML設定を読み込み
            try:
                cfg = self._load_yaml(yaml_path)
            except FileNotFoundError as e:
                print(f"[YAMLLoRASelector] エラー: YAMLファイル '{yaml_path}' が見つかりません。詳細: {e}")
                return error_return_value

            if category not in cfg:
                available_categories = list(cfg.keys()) if cfg else []
                print(
                    f"[YAMLLoRASelector] エラー: YAML にカテゴリ「{category}」が見つかりません。"
                    f"利用可能なカテゴリ: {available_categories}"
                )
                return error_return_value

            # 利用可能なLoRAファイルを取得
            available_loras = self._get_available_loras()
            if not available_loras:
                print(f"[YAMLLoRASelector] 警告: システムに利用可能なLoRAファイルが見つかりませんでした。")
                # この場合でも、YAMLにLoRA名が指定されていれば、_find_matching_lora_file が "None" を返す

            # カテゴリ設定を取得
            category_config = cfg[category]
            
            lora_keys = ["lora1", "lora2", "lora3"]
            for key_base in lora_keys:
                lora_raw = category_config.get(key_base, "")
                if not lora_raw: # YAMLにloraXが定義されていないか空文字の場合はスキップ
                    print(f"[YAMLLoRASelector] {key_base} はYAMLで定義されていないか空のためスキップします。")
                    continue

                lora_name_from_yaml, lora_weight = self._parse_lora_string(lora_raw)
                
                # LoRAファイル名を検索
                # _find_matching_lora_file は見つからない場合 "None" を返す
                lora_file_path_found = self._find_matching_lora_file(lora_name_from_yaml, available_loras)

                # WanVideoLoraSelectのgetlorapathメソッドの戻り値の形式に合わせる
                # lora_file_path_found が "None" の場合でも、そのまま設定
                # folder_paths.get_full_path は "None" の場合、そのまま "None" を返すことを期待
                # ただし、実際にファイルパスとして存在しない "None" を渡すとエラーになる可能性があるため、
                # lora_file_path_found が "None" でない場合のみ get_full_path を呼ぶ
                actual_path = "None"
                if lora_file_path_found != "None":
                    try:
                        actual_path = folder_paths.get_full_path("loras", lora_file_path_found)
                        if actual_path is None: # get_full_path が見つからない場合 None を返すことがある
                           actual_path = "None"
                           print(f"[YAMLLoRASelector] 警告: folder_paths.get_full_path が {lora_file_path_found} のパスを見つけられませんでした。")
                    except Exception as path_e:
                        print(f"[YAMLLoRASelector] 警告: folder_paths.get_full_path でエラー: {path_e}。ファイル: {lora_file_path_found}")
                        actual_path = "None" # エラー時もNoneにフォールバック
                
                lora_info = {
                    "path": actual_path,
                    "strength": lora_weight,
                    "name": lora_name_from_yaml if lora_name_from_yaml else "None", # YAMLでの名前
                    "blocks": {},  # YAMLLoRASelectorではblocksの指定はサポートしない
                    "layer_filter": "", # 同上
                    "low_mem_load": False # 同上
                }
                output_loras_list.append(lora_info)
                print(f"[YAMLLoRASelector] {key_base}: {lora_info['path']} (強度: {lora_info['strength']})")

            if not output_loras_list: # 処理できるLoRAが一つもなかった場合
                print(f"[YAMLLoRASelector] カテゴリ '{category}' に有効なLoRA設定が見つかりませんでした。")
                output_loras_list.append(default_wanvidlora_item) # デフォルトの "None" LoRA情報を追加

            return (output_loras_list, yaml_path)

        except Exception as e:
            print(f"[YAMLLoRASelector] 予期せぬエラーが発生しました: {e}")
            return error_return_value
