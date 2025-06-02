import os
import yaml
from PIL import Image
import numpy as np

class YAMLImageCyclerSimple:
    """
    指定カテゴリの画像を 1 枚ずつ巡回させながら、
    対応するマスクを一緒に返すシンプルなノード。
    """

    # ComfyUI メタ情報 --------------------------
    CATEGORY      = "Loaders"
    FUNCTION      = "execute"
    RETURN_TYPES  = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES  = ("image", "mask", "category", "yaml_path")

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
        abs_path = os.path.abspath(path) # 絶対パスに変換してみる
        if not os.path.exists(abs_path): # 絶対パスで存在確認
            cwd = os.getcwd()
            error_message = (
                f"YAMLファイルが見つかりません。\n"
                f"  指定されたパス: {path}\n"
                f"  絶対パスとして解釈: {abs_path}\n"
                f"  現在の作業ディレクトリ: {cwd}\n"
                f"  ファイルが存在するか、アクセス権があるか確認してください。"
            )
            raise FileNotFoundError(error_message)
        
        # pathの代わりにabs_pathを使うことで、キャッシュキーの一貫性を保つ
        if abs_path not in self._yaml_buf:
            with open(abs_path, "r", encoding="utf-8") as f:
                self._yaml_buf[abs_path] = yaml.safe_load(f)
        return self._yaml_buf[abs_path]

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
            print(f"[YAMLImageCyclerSimple] マスクの読み込みに失敗: {mask_path}, エラー: {e}")
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

    # ───────────────────────────────────────────
    # メイン処理
    # ───────────────────────────────────────────
    def execute(self, yaml_path: str, parent_dir: str, category: str):
        """メイン実行関数"""
        try:
            # YAMLパスのチェックと正規化
            abs_yaml_path = os.path.abspath(yaml_path)
            if not os.path.exists(abs_yaml_path):
                cwd = os.getcwd()
                error_message = (
                    f"executeメソッド: YAMLファイルが見つかりません。\n"
                    f"  指定されたyaml_path: {yaml_path}\n"
                    f"  絶対パスとして解釈: {abs_yaml_path}\n"
                    f"  現在の作業ディレクトリ: {cwd}\n"
                    f"  ComfyUIのUIで指定したパスが正しいか確認してください。"
                )
                raise FileNotFoundError(error_message)

            # YAML設定を読み込み（マスクフォルダ設定のため）
            cfg = self._load_yaml(abs_yaml_path) # 正規化されたパスを使用

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
                print(f"[YAMLImageCyclerSimple] マスク読み込み: {os.path.basename(mask_path)}")
            else:
                # マスクが見つからない場合は空のマスクを作成
                height, width = image_tensor.shape[1], image_tensor.shape[2]
                mask_tensor = self._create_empty_mask(height, width)
                print(f"[YAMLImageCyclerSimple] マスクが見つからないため空のマスクを作成")

            print(f"[YAMLImageCyclerSimple] カテゴリ: {category}, 画像: {images[current_idx]} ({current_idx + 1}/{len(images)})")

            return (image_tensor, mask_tensor, category, yaml_path)

        except Exception as e:
            print(f"[YAMLImageCyclerSimple] エラー: {e}")
            raise e
