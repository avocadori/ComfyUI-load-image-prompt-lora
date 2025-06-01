import os
import yaml
from PIL import Image
import numpy as np

class YAMLImageCycler:
    """
    指定カテゴリの画像を 1 枚ずつ巡回させながら、
    YAML に書かれた prompt / lora1-3 を一緒に返すノード。
    """

    # ComfyUI メタ情報 --------------------------
    CATEGORY      = "Loaders"
    FUNCTION      = "execute"
    RETURN_TYPES  = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("image", "prompt", "lora1", "lora2", "lora3")

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

            # YAML から設定値を抽出（None → 空文字）
            category_config = cfg[category]
            prompt = str(category_config.get("prompt", ""))
            lora1 = str(category_config.get("lora1", ""))
            lora2 = str(category_config.get("lora2", ""))
            lora3 = str(category_config.get("lora3", ""))

            print(f"[YAMLImageCycler] カテゴリ: {category}, 画像: {images[current_idx]} ({current_idx + 1}/{len(images)})")

            return (image_tensor, prompt, lora1, lora2, lora3)

        except Exception as e:
            print(f"[YAMLImageCycler] エラー: {e}")
            raise e
