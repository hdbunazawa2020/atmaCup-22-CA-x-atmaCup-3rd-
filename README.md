# 🏀 バスケットボール選手判別チャレンジ

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-green.svg)](https://hydra.cc/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

バスケットボールの試合動画から切り出された静止画に対し、**特定のバウンディングボックス（bbox）内の選手を識別**するコンペティションです。

---

## 📋 目次

- [コンペティション概要](#コンペティション概要)
- [タスク詳細](#タスク詳細)
- [データ説明](#データ説明)
- [ディレクトリ構成](#ディレクトリ構成)
- [セットアップ](#セットアップ)
- [実行方法](#実行方法)
- [開発ガイド](#開発ガイド)

---

## 🎯 コンペティション概要

### タスク

バスケットボールの試合動画から切り出された静止画と選手の位置情報（bounding box）が与えられるので、**その位置にいる選手のIDを予測**します。

### 評価指標

**Macro F1 スコア**で評価されます。

### 主な特徴

- 📹 **複数画角**: 上（フカン）と横からの2つの視点
- 🔄 **選手交代**: 試合途中で選手の入れ替えが発生
- ❓ **未知選手**: 学習データに存在しない選手は `-1` (unknown) として予測

---

## 📊 タスク詳細

### 入力データ

- **画像**: 試合動画から切り出された静止画
- **bbox情報**: 選手の位置（x, y, w, h）
- **画角**: side（横）/ top（フカン）

### 出力

- **label_id**: bbox内の選手ID
  - 学習データに存在する選手: 該当するID（整数）
  - 学習データに存在しない選手: `-1`

### データの特徴

1. **学習データ**: 全フレームの全選手の位置とIDが提供される
2. **テストデータ**: 
   - bbox情報のみ提供（IDは予測対象）
   - 一部フレームは両画角あり、残りは横（side）のみ
   - セッション単位で提供（時間的に離れたシーン）

---

## 📁 データ説明

### 画像データ

**ダウンロード**: [Google Drive](https://drive.google.com/file/d/1YXbi2O6-PIaQ3amm3-tkuWJhk1OObjtf/view?usp=drive_link)

#### ファイル命名規則

```
{quarter}__{angle}__{session}__{frame}.jpg
```

| 要素 | 説明 | 例 |
|------|------|-----|
| `quarter` | クオーター番号（試合の経過） | 1, 2, 3, 4 |
| `angle` | 画角 | `side` / `top` |
| `session` | シーンセッション番号 | 0, 1, 2, ... |
| `frame` | フレーム番号 | 0, 1, 2, ... |

**例**: `1__side__0__42.jpg`
- クオーター: 1
- 画角: 横（side）
- セッション: 0
- フレーム: 42

### メタデータ

#### 1. `train_meta.csv` (学習データ)

全画像の全選手の位置とIDを記録

| カラム | 説明 | 値の例 |
|--------|------|--------|
| `quarter` | クオーター番号 | 1, 2, 3, 4 |
| `angle` | 画角 | `side`, `top` |
| `session` | セッション番号 | 常に `0` |
| `frame` | フレーム番号 | 0, 1, 2, ... |
| `x`, `y`, `w`, `h` | bbox座標（ピクセル単位） | 100, 200, 50, 80 |
| `label_id` | 選手ID | 0, 1, 2, ..., 9 |

#### 2. `test_meta.csv` (予測対象)

予測対象の画像とbbox位置

- `label_id` カラムは**存在しない**（予測対象）
- `session` が **0以外の値**を取りうる
- `angle` は常に **`side`**

#### 3. `test_top_meta.csv` (補助データ)

テストデータの一部セッションの上（top）画角メタデータ

- 全セッションではなく**一部のみ**
- `test_meta.csv` の補助として使用

#### 4. `sample_submission.csv` (提出フォーマット)

提出ファイルの形式例

```csv
label_id
2
5
-1
0
...
```

- `label_id` 列のみ（カンマ区切りなし）
- `test_meta.csv` と同じ順序

---

## 📁 ディレクトリ構成

```
basketball-player-detection/
├── README.md                      # プロジェクト説明
│
├── input/                         # 入出力データ
│   ├── images/                    # 画像データ
│   │   ├── 1__side__0__0.jpg
│   │   ├── 1__side__0__1.jpg
│   │   └── ...
│   ├── train_meta.csv             # 学習用メタデータ
│   ├── test_meta.csv              # テスト用メタデータ
│   ├── test_top_meta.csv          # 追加の上視点データ
│   └── sample_submission.csv      # 提出フォーマット
│
├── notebook/                      # Jupyter Notebook
│   ├── 000_[EDA]basketball_player_detection.ipynb
│   └── 900_[ENS]ensemble.ipynb
│
├── script/                        # 実行スクリプト
│   ├── conf/                      # Hydra設定ファイル
│   │   ├── config.yaml            # メイン設定
│   │   ├── 000_data_preprocess/
│   │   ├── 200_train_model/
│   │   └── 300_inference/
│   │
│   ├── 000_data_preprocess/       # データ前処理
│   │   └── 000_data_preprocess.py
│   │
│   ├── 200_train_model/           # モデル学習
│   │   └── 200_train_resnet.py
│   │
│   ├── 300_inference/             # 推論
│   │   └── 300_inference.py
│   │
│   └── generate_template.py       # テンプレート生成
│
├── experiment/                    # 実行結果
│   ├── 200_train_resnet_exp001/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── config.yaml
│
├── src/                           # ソースコード
│   ├── models/                    # モデル定義
│   ├── datasets/                  # データローダー
│   ├── utils/                     # ユーティリティ
│   └── training/                  # 学習ロジック
│
├── requirements.txt               # 依存パッケージ
└── .gitignore
```

### ディレクトリ詳細

#### 1. `input/` - 入出力データ

入力データと提出ファイルを格納

#### 2. `notebook/` - Jupyter Notebook

各種検討、可視化、後処理に活用

**連番ルール:**
- `0xx`: データの前処理・EDA
- `1xx`: 機械学習（決定木など）
- `2xx`: Deep Learning（CNN系）
- `3xx`: Deep Learning（その他）
- `9xx`: アンサンブル

#### 3. `script/` - 実行スクリプト

本番実行用のPythonスクリプト

**連番ルール:**
- `0xx`: データの前処理
- `1xx`: 機械学習（決定木など）
- `2xx`: Deep Learning（CNN系）
- `3xx`: Deep Learning（その他）
- `9xx`: アンサンブル

**サブディレクトリ:**
- `conf/`: Hydra設定ファイル

#### 4. `output/` - 実行結果

各実験の出力を保存

**命名規則**: `{連番}_{スクリプト名}_exp{実験番号}/`

**例**: `200_train_resnet_exp001/`

---

## 🔧 セットアップ

### 前提条件

- Python 3.9 以上
- CUDA 11.8 以上（GPU使用時）

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/your-org/basketball-player-detection.git
cd basketball-player-detection

# 仮想環境の作成
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### データのダウンロード

```bash
# 画像データをダウンロード（Google Drive）
# input/images/ に配置

# メタデータは input/ 直下に配置
```

---

## 🚀 実行方法

### 基本的な実行方法

**重要**: ワーキングディレクトリは `script/` ディレクトリとする

```bash
# scriptディレクトリに移動
cd script

# スクリプト実行（引数なし）
python 000_data_preprocess/000_data_preprocess.py
```

### データ前処理

```bash
cd script
python 000_data_preprocess/000_data_preprocess.py
```

### モデル学習

```bash
cd script
python 200_train_model/200_train_resnet.py
```

### 推論・提出ファイル作成

```bash
cd script
python 300_inference/300_inference.py
```

---

## ⚙️ 設定ファイル（Hydra）

### 基本構成

[Hydra](https://hydra.cc/)を使用して設定を管理

- **メイン設定**: `script/conf/config.yaml`
- **スクリプト別設定**: `script/conf/{スクリプト名}/{設定名}.yaml`

### 設定例

#### `config.yaml` (メイン)

```yaml
defaults:
  - 000_data_preprocess: 000_data_preprocess_default
  - 200_train_model: 200_train_resnet_default
  - 300_inference: 300_inference_default

# グローバル設定
data_dir: ../input
output_dir: ../output
seed: 42
```

#### `conf/200_train_model/200_train_resnet_default.yaml`

```yaml
model:
  backbone: resnet50
  num_classes: 11  # 10選手 + unknown(-1)

training:
  batch_size: 32
  epochs: 50
  lr: 1e-4
  optimizer: AdamW

data:
  image_size: 224
  augmentation: true
```

### コマンドラインでの設定変更

```bash
# 設定ファイルを指定
python 200_train_resnet.py 200_train_model=200_train_resnet_custom

# パラメータを直接オーバーライド
python 200_train_resnet.py \
    training.batch_size=64 \
    training.lr=5e-5 \
    model.backbone=resnet101
```

詳細は [Hydraドキュメント](https://hydra.cc/docs/intro/) を参照

---

## 🛠️ 開発ガイド

### 新規スクリプトの作成

`generate_template.py` を使用してテンプレート生成

```bash
cd script
python generate_template.py --name 250_train_efficientnet
```

生成されるファイル:
- `script/250_train_efficientnet/250_train_efficientnet.py`
- `script/conf/250_train_efficientnet/250_train_efficientnet_default.yaml`

### コーディング規約

#### 1. Docstring

[Google Style](https://google.github.io/styleguide/pyguide.html) に準拠

```python
def detect_player(image: np.ndarray, bbox: tuple) -> int:
    """
    画像とbboxから選手IDを予測
    
    Args:
        image (np.ndarray): 入力画像
        bbox (tuple): (x, y, w, h) のバウンディングボックス
        
    Returns:
        int: 予測された選手ID（-1はunknown）
        
    Raises:
        ValueError: bboxが画像範囲外の場合
    """
    pass
```

#### 2. Type Hints

Python 3.9以降対応のため、`from __future__ import annotations` を使用

```python
from __future__ import annotations

def process_data(data: list[dict]) -> dict[str, list[int]]:
    """型ヒントの例"""
    pass
```

#### 3. パスの扱い

`pathlib` を使用（OS間の互換性のため）

```python
from pathlib import Path

# Good
data_dir = Path("../input")
image_path = data_dir / "images" / "1__side__0__0.jpg"

# Bad
data_dir = "../input"
image_path = data_dir + "/images/1__side__0__0.jpg"
```

### コードフォーマット

[Black](https://black.readthedocs.io/en/stable/) でフォーマットを統一

#### インストール

```bash
pip install black
```

#### 使用方法

```bash
# 単一ファイル
black script/000_data_preprocess/000_data_preprocess.py

# ディレクトリ全体
black script/

# チェックのみ（変更なし）
black --check script/
```

#### VSCode拡張機能

[Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) をインストール

設定（`.vscode/settings.json`）:
```json
{
    "python.formatting.provider": "black",
    "editor.formatOnSave": true
}
```

---

## 📈 実験管理

### 実験の命名規則

```
{連番}_{スクリプト名}_exp{実験番号}
```

**例**: 
- `200_train_resnet_exp001`
- `200_train_resnet_exp002`
- `300_inference_exp001`

### 実験ディレクトリ構成

```
output/200_train_resnet_exp001/
├── config.yaml              # 使用した設定
├── checkpoints/             # モデルチェックポイント
│   ├── best_model.pth
│   └── last_model.pth
├── logs/                    # ログファイル
│   ├── train.log
│   └── tensorboard/
└── metrics.json             # 評価指標
```

---

## 🎯 ベースライン手法

### 1. シンプルなCNN

- ResNet50ベースの画像分類
- bbox領域をクロップして識別

### 2. 時系列情報の活用

- 前後フレームの情報を利用
- LSTM/Transformerで時間的一貫性を考慮

### 3. マルチビュー学習

- side と top の両画角を統合
- Attention機構で画角間の対応付け

### 4. 未知選手の検出

- Out-of-Distribution検出
- 信頼度スコアで -1 を判定

---

## 🐛 トラブルシューティング

### よくある問題

#### 1. 画像が見つからない

```python
# パスの確認
from pathlib import Path

image_dir = Path("../input/images")
print(f"画像ディレクトリ存在: {image_dir.exists()}")
print(f"画像数: {len(list(image_dir.glob('*.jpg')))}")
```

#### 2. メモリ不足

```bash
# バッチサイズを減らす
python 200_train_resnet.py training.batch_size=16
```

#### 3. 設定ファイルが読み込めない

```bash
# 実行ディレクトリを確認
pwd  # scriptディレクトリにいることを確認

# 設定ファイルの存在確認
ls conf/config.yaml
```

---

## 📚 参考資料

- [Hydra Documentation](https://hydra.cc/)
- [PyTorch Object Detection Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [Black Code Style](https://black.readthedocs.io/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

## 🤝 コントリビューション

プルリクエストを歓迎します！

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

---

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

---

## 📞 お問い合わせ

- **Issues**: [GitHub Issues](https://github.com/your-org/basketball-player-detection/issues)
- **Email**: your.email@example.com

---

**⭐ このプロジェクトが役立った場合は、Starをつけていただけると嬉しいです！**