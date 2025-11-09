# University of Tokyo Deep Learning Course Competition

[日本語版はこちら](README.ja.md)

## Competition Results
- **Final Rank**: **15th** out of 1,593 participants
- **LB Score**: **0.905**

## Overview
Classification of Fashion MNIST (10 classes) using softmax regression.

For more details about Fashion MNIST, please refer to:
Fashion MNIST: https://github.com/zalandoresearch/fashion-mnist

## Rules
- Training data is provided as `x_train`, `y_train`, and test data as `x_test`.
- Prediction labels should be class labels 0~9, not one-hot representation.
- Do not use training data other than `x_train` and `y_train` specified in the cells below.
- The softmax regression algorithm implementation must be done using only numpy (do not use sklearn, tensorflow, etc.).
- It is acceptable to use sklearn functions for data preprocessing (e.g., `sklearn.model_selection.train_test_split`).

## Approach

- Data Preprocessing/Splitting

  - Load `x_train.npy`, `y_train.npy`, `x_test.npy`, flatten 28×28→784 and normalize to 0–1 (`float64`). Preprocessing via `model.preprocess`
  - Stratified split: train:validation = 90:10 (`train_test_split`, default `random_state=0`). Only training minibatches are shuffled; validation/test are in fixed order
  - After feature construction, apply standardization (`StandardScaler`) and PCA (with whitening). Safety check: stop if cumulative explained variance < 0.65
  - Retrain on full data using the same schema (retrain scaler/PCA after validation, then retrain on full data for test)

- Image Augmentation (Difficult Class Enhancement: Pullover=2, Shirt=6)

  - Translation: max ±1px (up/down/left/right, `np.roll`, zero-padding at boundaries)
  - Horizontal flip: probability 0.3
  - Gaussian noise: σ=0.015 (0.01 in main config)
  - Sampling: extract `sample_ratio` from each difficult class, generate `augment_factor` variants per sample (default 2)
  - Application: enabled by default for training split during hold-out training; also applied to full training data for final model (enabled by default)
  - Default (main script): `sample_ratio=0.5` for train side, `0.8` for full side

- Features

  - Base: raw pixels + edge/contrast differences (horizontal, vertical, diagonal) concatenated (fully vectorized)
  - HOG (optional): cell=`4`, bins=`8`. Add different scales via `hog_extra_cell_sizes`. Optional thin spatial pyramid (1×1 + 2×2)
  - LBP (optional): cell=`4`, bins=`16`. Supports `lbp_extra_cell_sizes`. Supports thin spatial pyramid
  - Statistical features (optional): row/column permutation entropy (order=3), adjacent row/column correlation statistics
  - Alternative pipeline (optional): PCA→PolynomialFeatures (degree 2)→StandardScaler. When enabled, the above handcrafted features are disabled

- Model (NumPy-implemented Softmax Regression)

  - Optimization: mini-batch GD (default lr=0.2, batch=128, epochs=80)
  - Regularization/Stabilization: L2 regularization (bias excluded), gradient clipping (1.0), label smoothing support, class weights support
  - Learning rate decay: if validation loss does not improve for `patience=5` consecutive epochs, decay by `lr_decay=0.95`
  - Best epoch restoration + EMA: track EMA with `ema_decay=0.9` after epoch 60. A/B test "best vs EMA" on validation accuracy and adopt the better one
  - Logging: record epoch loss/validation loss/clip rate and save as `training_logs_model_1.csv`

- Validation, Ensemble, and Calibration

  - Ensemble: train each preset (e.g., `wide7`) separately, weighted fusion in logit space. A/B test "equal" vs "validation accuracy proportional" weights; if difference < ±3e-4, use equal; otherwise use higher accuracy side
  - Binary calibration: for each specified class (default 2,6), train a small ensemble of binary classifiers and blend class probabilities (α∈{0.1..0.5}). Apply only to ambiguous region (0.15≤p≤0.85), adopt if validation accuracy improves. A/B test "weighted average vs median aggregation" for binary side and use the better one
  - Distribution bias correction γ: compute logit correction from true class frequency and predicted frequency on validation (γ=0.30). Soft application based on max probability `p_max` in 0.70–0.85 range; enable only if validation accuracy improves by ≥+1e-4. Also apply at test time (adjust multiplier via `--bias-scale`)
  - Additional output: print analysis dashboard to console including confusion matrix, reliability (ECE), γ application impact, weight differences, etc.

- Specialist (Binary Model, Optional)

  - Train dedicated feature pipeline + logistic regression for target pair (default 2↔6)
  - Apply only to samples where "prediction is target pair" AND "confidence gap ≤ threshold", overwrite-blend probabilities with `blend_alpha`
  - Report base vs specialist accuracy on validation, adopt if needed (disabled in main config)

- TTA (Test Time, Limited)

  - Condition: only samples where top-2 is {2,6} AND |p6−p2| ≤ 0.06
  - Generate horizontal flip and average logits (base and flipped, 2 variants)

- Cross-Validation

  - `--mode cv` for stratified K-fold (default 5). Aggregate individual and ensemble accuracies per fold, output mean±std. Optionally include specialist metrics

- Inference/Save and Retraining

  - After hold-out validation, retrain on full training data (+ augmentation if needed) and infer on test
  - Save predictions to `data/output/predictions.csv` with `label` header

- Execution and Reproducibility

  - Execute with fixed random seeds and fixed BLAS thread counts (see `main.py`)
  - Safety mechanism: CLI override of `--pca-components` is explicitly disabled (to prevent accidents). Default is `384`

## Technologies Used

- Python 3.9+

- NumPy (core implementation of softmax regression, general numerical computation)

- Pandas (DataFrame operations, CSV saving)

- scikit-learn
  - `sklearn.model_selection` (`train_test_split`, `StratifiedKFold`)
  - `sklearn.preprocessing` (`StandardScaler`, `PolynomialFeatures`, `FunctionTransformer`)
  - `sklearn.decomposition` (`PCA`)
  - `sklearn.pipeline` (`Pipeline`)
  - `sklearn.metrics` (`confusion_matrix`)

- Standard Library
  - `argparse` (command-line argument parsing)
  - `pathlib` (path operations)
  - `copy` (`deepcopy`)
  - `math` (mathematical functions)
