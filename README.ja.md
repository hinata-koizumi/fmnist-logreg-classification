# 東大 深層学習講座 コンペティション

[English version](README.md)

## コンペティション結果
- **最終順位**: **15位**/1593人中
- **LBスコア**: **0.905**

## 概要
MNISTのファッション版 (Fashion MNIST，クラス数10) をソフトマックス回帰によって分類．

Fashion MNISTの詳細については以下のリンクを参考にしてください．
Fashion MNIST: https://github.com/zalandoresearch/fashion-mnist

## ルール
- 訓練データはx_train， y_train，テストデータはx_testで与えられます．
- 予測ラベルは one_hot表現ではなく0~9のクラスラベル で表してください．
- 下のセルで指定されているx_train、y_train以外の学習データは使わないでください．
- ソフトマックス回帰のアルゴリズム部分の実装はnumpyのみで行ってください (sklearnやtensorflowなどは使用しないでください)．
- データの前処理部分でsklearnの関数を使う (例えば sklearn.model_selection.train_test_split) のは問題ありません．

## アプローチ

- データ前処理/分割

  - `x_train.npy`, `y_train.npy`, `x_test.npy` を読み込み、28×28→784にフラット化し0–1へ正規化（`float64`）。前処理は `model.preprocess`
  - 訓練:検証=90:10 の層化分割（`train_test_split`、既定 `random_state=0`）。学習ミニバッチのみシャッフル、検証/テストは順序固定
  - 特徴量構築後に標準化（`StandardScaler`）とPCA（白色化付き）を適用。累積寄与率が0.65未満なら安全装置で停止
  - フルデータ再学習時も同じスキーマでスケーラ/PCAを学習し直す（検証後に全データで再学習してテストへ）

- 画像拡張（困難クラス強化: Pullover=2, Shirt=6）

  - 平行移動: 最大±1px（上下左右、`np.roll`、境界0埋め）
  - 水平反転: 確率0.3
  - ガウスノイズ: σ=0.015（メイン設定では0.01）
  - サンプリング: 各難クラスから割合 `sample_ratio` を抽出し、各サンプルに `augment_factor` 個生成（既定2）
  - 適用箇所: ホールドアウト学習時は学習分割（既定有効）、最終モデルでは全訓練データにも適用（既定有効）
  - 既定（メインスクリプト）: train側 `sample_ratio=0.5`、full側 `0.8`

- 特徴量

  - 基本: 生画素＋エッジ/コントラスト差分（水平・垂直・斜め）を連結（完全ベクトル化）
  - HOG（任意）: cell=`4`、bins=`8`。`hog_extra_cell_sizes` で別スケールを追加。薄い空間ピラミッド（1×1＋2×2）も任意で付加
  - LBP（任意）: cell=`4`、bins=`16`。`lbp_extra_cell_sizes` 対応。薄い空間ピラミッド対応
  - 統計特徴（任意）: 行/列の順列エントロピー（order=3）、隣接行/列の相関統計
  - 代替パイプライン（任意）: PCA→PolynomialFeatures（2次）→StandardScaler。これを有効化すると上記手作り特徴は無効化

- モデル（NumPy実装ソフトマックス回帰）

  - 最適化: ミニバッチGD（既定 lr=0.2, batch=128, epochs=80）
  - 正則化/安定化: L2正則化（バイアス除外）、勾配クリップ（1.0）、ラベルスムージング対応、クラス重み対応
  - 学習率減衰: 検証損失が `patience=5` 連続で改善しないと `lr_decay=0.95` で減衰
  - ベストエポック復元＋EMA: 60エポック以降 `ema_decay=0.9` でEMA追跡。検証精度で「best vs EMA」をA/Bし良い方を採用
  - ログ: エポック損失/検証損失/クリップ率を記録し、`training_logs_model_1.csv` として保存

- 検証・アンサンブル・キャリブレーション

  - アンサンブル: 事前プリセット（例: `wide7`）をそれぞれ学習し、logit空間で加重融合。重みは「均等」vs「検証精度比例」をA/Bし、差が±3e-4未満なら均等、それ以外は高精度側を採用
  - バイナリ校正: 指定クラス（既定 2,6）ごとに二値分類の小アンサンブルを学習し、クラス確率をブレンド（α∈{0.1..0.5}）。中庸域（0.15≤p≤0.85）のみ適用し、検証精度が改善した場合に採用。二値側は「重み平均 vs 中央値集約」をA/Bして良い方を使用
  - 分布バイアス補正γ: 検証における真のクラス頻度と予測頻度からlogit補正を計算（γ=0.30）。最大確率 `p_max` に応じて0.70〜0.85帯でソフト適用し、検証精度が +1e-4 以上改善する場合のみ有効化。テスト時も同様に適用（`--bias-scale` で倍率調整）
  - 付随出力: 混同行列・信頼性（ECE）・γ適用影響・重み差分などの分析ダッシュボードをコンソールに出力

- スペシャリスト（二値モデル、任意）

  - 対象ペア（既定 2↔6）専用の特徴パイプライン＋ロジスティック回帰を学習
  - 適用は「予測が対象ペア」かつ「確信差が閾値以下」のサンプルへ限定し、`blend_alpha` で確率を上書きブレンド
  - 検証でベース vs スペシャリストの精度を報告し、必要に応じて採用（メイン設定では無効）

- TTA（テスト時、限定的）

  - 条件: top-2 が {2,6} かつ |p6−p2| ≤ 0.06 のサンプルのみ
  - 水平反転を生成し、logit を平均化（ベースと反転の2通り）

- クロスバリデーション

  - `--mode cv` で層化K分割（既定5）。各foldの単体精度とアンサンブル精度を集計し、平均±標準偏差を出力。スペシャリスト指標も任意で併記

- 推論/保存・再学習

  - ホールドアウト検証後、全訓練（＋必要なら拡張）で再学習してテストへ推論
  - 予測は `data/output/predictions.csv` に `label` ヘッダで保存

- 実行・再現性

  - 乱数固定・BLASスレッド数固定で実行（`main.py` 参照）
  - 安全装置: `--pca-components` のCLI上書きは明示的に禁止（事故防止）。既定は `384`

## 使用技術

- Python 3.9+

- NumPy（ソフトマックス回帰のコア実装、数値計算全般）

- Pandas（DataFrame操作、CSV保存）

- scikit-learn
  - `sklearn.model_selection`（`train_test_split`, `StratifiedKFold`）
  - `sklearn.preprocessing`（`StandardScaler`, `PolynomialFeatures`, `FunctionTransformer`）
  - `sklearn.decomposition`（`PCA`）
  - `sklearn.pipeline`（`Pipeline`）
  - `sklearn.metrics`（`confusion_matrix`）

- 標準ライブラリ
  - `argparse`（コマンドライン引数解析）
  - `pathlib`（パス操作）
  - `copy`（`deepcopy`）
  - `math`（数学関数）

