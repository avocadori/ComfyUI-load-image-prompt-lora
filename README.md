# ComfyUI YAML Image Cycler ノード

指定したカテゴリの画像を順番に巡回しながら、YAMLファイルに定義されたプロンプトとLoRA設定を一緒に出力するComfyUIカスタムノードです。

## 機能

- 指定フォルダ内の画像を順番に巡回表示
- YAMLファイルからプロンプトとLoRA設定を自動読み込み
- カテゴリ別の設定管理
- 複数のLoRA（最大3つ）に対応

## インストール方法

1. ComfyUIの`custom_nodes`フォルダ内に本フォルダをコピー
2. ComfyUIを再起動

## ファイル構成

```
yaml_image_cycler/
├── __init__.py              # ノード登録ファイル
├── yaml_image_cycler.py     # メインノードクラス
├── setting.yaml             # 設定ファイル（サンプル）
└── README.md               # このファイル
```

## 使用方法

### 1. フォルダ構成の準備

```
data/
├── character1/
│   ├── img001.png
│   ├── img002.jpg
│   └── img003.webp
├── character2/
│   ├── portrait01.png
│   └── portrait02.jpg
└── background/
    ├── landscape01.png
    └── landscape02.png
```

### 2. YAML設定ファイルの編集

`setting.yaml`を編集して、各カテゴリの設定を定義：

```yaml
character1:
  prompt: "1girl, beautiful, anime style, detailed face"
  lora1: "<lora:character1_v1:0.8>"
  lora2: "<lora:anime_style:0.6>"
  lora3: ""

character2:
  prompt: "1boy, handsome, realistic, portrait"
  lora1: "<lora:character2_v2:0.9>"
  lora2: "<lora:realistic_style:0.7>"
  lora3: "<lora:portrait_enhance:0.5>"
```

### 3. ComfyUIでの使用

1. ノードメニューから「Loaders」→「YAML Image Cycler」を選択
2. パラメータを設定：
   - `yaml_path`: 設定ファイルのパス（デフォルト: setting.yaml）
   - `parent_dir`: 画像フォルダの親ディレクトリ（デフォルト: ./data）
   - `category`: 使用するカテゴリ名（YAMLで定義したもの）

### 4. 出力

ノードは以下の5つの出力を提供：
- `image`: 巡回中の画像（ComfyUI IMAGE形式）
- `prompt`: YAMLで定義されたプロンプト
- `lora1`: 1番目のLoRA設定
- `lora2`: 2番目のLoRA設定
- `lora3`: 3番目のLoRA設定

## 対応画像形式

- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- WebP (.webp)
- TIFF (.tiff)
- TGA (.tga)

## 注意事項

- 画像は各カテゴリフォルダ内でアルファベット順にソートされます
- 巡回状態は各カテゴリごとに独立して管理されます
- YAMLファイルが存在しない場合、カテゴリは手動入力になります
- 画像フォルダが存在しないか空の場合はエラーが発生します

## トラブルシューティング

### よくあるエラー

1. **「YAMLファイルが見つかりません」**
   - `yaml_path`で指定したファイルが存在するか確認
   - ファイルパスが正しいか確認

2. **「フォルダがありません」**
   - `parent_dir`で指定したフォルダが存在するか確認
   - カテゴリ名に対応するサブフォルダが存在するか確認

3. **「画像が見つかりません」**
   - 指定フォルダ内に対応形式の画像ファイルがあるか確認
   - ファイル拡張子が対応形式か確認

### デバッグ情報

ノード実行時にコンソールに以下の情報が出力されます：
```
[YAMLImageCycler] カテゴリ: character1, 画像: img001.png (1/3)
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。
