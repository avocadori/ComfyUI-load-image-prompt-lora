import os
import yaml
from PIL import Image
import numpy as np
import sys

class YAMLImageCycler:
    """
    指定カテゴリの画像を 1 枚ずつ巡回させながら、
    YAML に書かれた prompt / lora1-3 とマスクを一緒に返すノード。
    """

    # ComfyUI メタ情報 --------------------------
    CATEGORY      = "Loaders"
    FUNCTION      = "execute"
    RETURN_TYPES  = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("image", "mask", "prompt", "lora1", "lora2", "lora3")

    def __init__(self):
        # インスタンス変数として状態を管理
        self._cursor = {}      # {category: index}
        self._yaml_buf = {}    # キャッシュした YAML dict

    @classmethod
    def INPUT_TYPES(cls):
        """
        yaml_path と parent_dir は毎回 UI で設定。
        category は文字列入力でも良いが、
        UI 側でゆらぎを防ぐためドロップダウンにしやすいよう選択肢も返す。
        （YAML を開けない場合は空リストにして手打ちで指定可能）
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
                "parent_dir": ("STRING", {"default": "./data"}),
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

    def _get_image_files(self, folder_path):
        """フォルダから画像ファイル一覧を取得"""
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"フォルダがありません: {folder_path}")

        supported_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tga")
        images = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(supported_extensions)
        )
        
        if not images:
            raise RuntimeError(f"フォルダ {folder_path} に画像が見つかりません。")
        
        return images

    def _load_image_as_tensor(self, image_path):
        """画像をComfyUI形式のテンソルとして読み込み"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
        
        try:
            pil_image = Image.open(image_path).convert("RGB")
            # ComfyUI の IMAGE フォーマット (B, H, W, C, fp32 0-1)
            np_img = np.array(pil_image).astype(np.float32) / 255.0
            np_img = np_img[None,]  # Batch 次元を付加
            return np_img
        except Exception as e:
            raise RuntimeError(f"画像の読み込みに失敗しました: {image_path}, エラー: {e}")

    def _load_mask_as_tensor(self, mask_path):
        """マスク画像をComfyUI形式のテンソルとして読み込み"""
        if not os.path.exists(mask_path):
            return None
        
        try:
            pil_mask = Image.open(mask_path).convert("L")  # グレースケールに変換
            # ComfyUI の MASK フォーマット (B, H, W, fp32 0-1)
            np_mask = np.array(pil_mask).astype(np.float32) / 255.0
            np_mask = np_mask[None,]  # Batch 次元を付加
            return np_mask
        except Exception as e:
            print(f"[YAMLImageCycler] マスクの読み込みに失敗: {mask_path}, エラー: {e}")
            return None

    def _create_empty_mask(self, height, width):
        """空のマスクを作成"""
        return np.zeros((1, height, width), dtype=np.float32)

    def _find_mask_file(self, image_path, mask_folder=None):
        """画像に対応するマスクファイルを検索"""
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_dir = os.path.dirname(image_path)
        
        # マスクフォルダが指定されている場合はそこを検索
        if mask_folder and os.path.isdir(mask_folder):
            search_dirs = [mask_folder]
        else:
            # デフォルトでは画像と同じフォルダ内のmasksサブフォルダを検索
            search_dirs = [
                os.path.join(image_dir, "masks"),
                image_dir  # 画像と同じフォルダも検索
            ]
        
        mask_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
        
        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue
                
            for ext in mask_extensions:
                mask_path = os.path.join(search_dir, f"{image_name}{ext}")
                if os.path.exists(mask_path):
                    return mask_path
        
        return None

    def _get_comfyui_lora_paths(self):
        """ComfyUIのLoRAフォルダパスを取得"""
        possible_paths = []
        
        # 現在のディレクトリから上位を検索
        current_dir = os.getcwd()
        for i in range(5):  # 最大5階層上まで検索
            test_path = os.path.join(current_dir, "models", "loras")
            if os.path.isdir(test_path):
                possible_paths.append(test_path)
            current_dir = os.path.dirname(current_dir)
        
        # 相対パス
        relative_paths = [
            "models/loras",
            "../models/loras",
            "../../models/loras",
            "../../../models/loras"
        ]
        
        for rel_path in relative_paths:
            if os.path.isdir(rel_path):
                possible_paths.append(os.path.abspath(rel_path))
        
        # 環境変数から取得
        if os.environ.get("COMFYUI_PATH"):
            env_path = os.path.join(os.environ.get("COMFYUI_PATH"), "models", "loras")
            if os.path.isdir(env_path):
                possible_paths.append(env_path)
        
        return list(set(possible_paths))  # 重複を除去

    def _check_lora_exists(self, lora_name):
        """LoRAファイルの存在確認"""
        if not lora_name:
            return False
            
        lora_paths = self._get_comfyui_lora_paths()
        lora_extensions = [".safetensors", ".ckpt", ".pt", ".pth"]
        
        for base_path in lora_paths:
            for ext in lora_extensions:
                lora_file = os.path.join(base_path, f"{lora_name}{ext}")
                if os.path.exists(lora_file):
                    print(f"[YAMLImageCycler] LoRAファイル確認: {lora_file}")
                    return True
        
        # 見つからない場合は検索パスを表示
        print(f"[YAMLImageCycler] LoRAファイル '{lora_name}' が見つかりません")
        print(f"[YAMLImageCycler] 検索パス: {lora_paths}")
        return False

    def _extract_lora_name(self, lora_string):
        """LoRA文字列から名前部分を抽出
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
                lora_name = parts[0].strip()
                # LoRAファイルの存在確認
                self._check_lora_exists(lora_name)
                return lora_name
        
        # 通常の文字列の場合はそのまま返す
        lora_name = str(lora_string).strip()
        if lora_name:
            self._check_lora_exists(lora_name)
        return lora_name

    # ───────────────────────────────────────────
    # メイン処理
    # ───────────────────────────────────────────
    def execute(self, yaml_path: str, parent_dir: str, category: str):
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

            # 画像フォルダのパスを構築
            cat_folder = os.path.join(parent_dir, category)
            images = self._get_image_files(cat_folder)

            # 巡回インデックスを取得・更新
            current_idx = self._cursor.get(category, 0) % len(images)
            self._cursor[category] = (current_idx + 1) % len(images)  # 次回用に +1（オーバーフロー対策）

            # 画像を読み込み
            img_path = os.path.join(cat_folder, images[current_idx])
            image_tensor = self._load_image_as_tensor(img_path)

            # マスクを検索・読み込み
            category_config = cfg[category]
            mask_folder = category_config.get("mask_folder")  # YAMLでマスクフォルダを指定可能
            if mask_folder:
                mask_folder = os.path.join(parent_dir, mask_folder)
            
            mask_path = self._find_mask_file(img_path, mask_folder)
            if mask_path:
                mask_tensor = self._load_mask_as_tensor(mask_path)
                print(f"[YAMLImageCycler] マスク読み込み: {os.path.basename(mask_path)}")
            else:
                # マスクが見つからない場合は空のマスクを作成
                height, width = image_tensor.shape[1], image_tensor.shape[2]
                mask_tensor = self._create_empty_mask(height, width)
                print(f"[YAMLImageCycler] マスクが見つからないため空のマスクを作成")

            # YAML から設定値を抽出
            prompt = str(category_config.get("prompt", ""))
            lora1_raw = category_config.get("lora1", "")
            lora2_raw = category_config.get("lora2", "")
            lora3_raw = category_config.get("lora3", "")
            
            # LoRA名前を抽出（<lora:name:weight>形式から名前部分のみ取得）
            lora1 = self._extract_lora_name(lora1_raw)
            lora2 = self._extract_lora_name(lora2_raw)
            lora3 = self._extract_lora_name(lora3_raw)

            print(f"[YAMLImageCycler] カテゴリ: {category}, 画像: {images[current_idx]} ({current_idx + 1}/{len(images)})")
            print(f"[YAMLImageCycler] LoRA: {lora1}, {lora2}, {lora3}")

            return (image_tensor, mask_tensor, prompt, lora1, lora2, lora3)

        except Exception as e:
            print(f"[YAMLImageCycler] エラー: {e}")
            raise e
