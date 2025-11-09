#main.py============================================
#!/usr/bin/env python3


from pathlib import Path
import argparse
import sys
import os
import random

# 再現性のための環境固定
os.environ['PYTHONHASHSEED'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
random.seed(0)
import numpy as np
np.random.seed(0)

# Allow importing the implementation utilities.
IMPLEMENTATION_DIR = Path(__file__).resolve().parent
if IMPLEMENTATION_DIR.exists():
    sys.path.append(str(IMPLEMENTATION_DIR))

from model import (  # type: ignore
    train_pipeline,
    run_cross_validation_mode,
    DEFAULT_DIFFICULT_CLASSES,
)

CALIBRATION_BEST_CONFIG = {
        "ensemble": "wide7",
    "pca_components": 384,  
    "augment_factor": 2,
    "augment_sample_ratio_train": 0.5,  # 0.8 → 0.5 に下げてラベルノイズ減衰
    "augment_sample_ratio_full": 0.8,
    "augment_max_shift": 1,
    "augment_noise_std": 0.01,  # 0.015 → 0.01 に下げてノイズ減衰
    "use_edges": True,
    "use_hog_features": True,
    "hog_cell_size": 4,
    "hog_bins": 8,
    "hog_extra_cell_sizes": (7,),
    "use_lbp_features": True,
    "lbp_cell_size": 4,
    "lbp_bins": 16,
    "lbp_extra_cell_sizes": (7,),
    "use_entropy_features": False,
    "entropy_order": 3,
    "use_corr_features": False,
    "use_spatial_pyramid": False,
    "reg_scale": 1.5,  # 1.0 → 1.5 に強化
        "label_smoothing": 0.0,
    "use_specialist": False,
        "calibrate_classes": (2, 6),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fashion MNIST PB0.9024達成モデル - メインスクリプト (wide7 + HOG/LBP + calibration on classes 2 and 6)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/input"),
        help="Directory with x_train.npy, y_train.npy, x_test.npy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/predictions.csv"),
        help="Where to save predictions when running in train mode.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "cv"),
        default="train",
        help="train: fit on train/val split and write predictions; cv: run cross-validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for the train/validation split and augmentation.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds when running cross-validation.",
    )
    parser.add_argument(
        "--cv-random-state",
        type=int,
        default=42,
        help="Random seed used for cross-validation splits.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="Number of PCA components (overrides config default).",
    )
    parser.add_argument(
        "--min-explained",
        type=float,
        default=0.70,
        help="Minimum cumulative explained variance required (safety floor).",
    )
    parser.add_argument(
        "--bias-scale",
        type=float,
        default=1.0,
        help="Scale factor for logit bias application (default: 1.0).",
    )
    args = parser.parse_args()

    if args.mode == "train":
        print("=== Fashion MNIST PB0.9024達成モデル - メインスクリプト ===")
        print("完全な再現性を確保した実行を開始します...")
        
        # Override pca_components if specified
        config = CALIBRATION_BEST_CONFIG.copy()
        if args.pca_components is not None:
            # 事故防止: CLIからの上書きを検知したら明示的に停止
            print(f"PCA components overridden to: {args.pca_components}")
            print("ERROR: PCA components override via CLI is disabled to prevent accidental regressions.")
            print("       Remove --pca-components to use the safe default/config value (384) or implement auto-selection.")
            raise SystemExit(1)
        
        val_acc = train_pipeline(
            args.data_dir,
            args.output,
            difficult_classes=DEFAULT_DIFFICULT_CLASSES,
            random_state=args.random_state,
            bias_scale=args.bias_scale,
            **config,
        )
        print(f"Hold-out validation accuracy: {val_acc:.4f}")
        print(f"予測ファイルを保存しました: {args.output}")
        print("=== 実行完了 ===")
    else:
        run_cross_validation_mode(
            args.data_dir,
            ensemble=CALIBRATION_BEST_CONFIG["ensemble"],
            pca_components=CALIBRATION_BEST_CONFIG["pca_components"],
            augment_factor=CALIBRATION_BEST_CONFIG["augment_factor"],
            augment_sample_ratio_train=CALIBRATION_BEST_CONFIG["augment_sample_ratio_train"],
            augment_max_shift=CALIBRATION_BEST_CONFIG["augment_max_shift"],
            augment_noise_std=CALIBRATION_BEST_CONFIG["augment_noise_std"],
            use_edges=CALIBRATION_BEST_CONFIG["use_edges"],
            use_hog_features=CALIBRATION_BEST_CONFIG["use_hog_features"],
            hog_cell_size=CALIBRATION_BEST_CONFIG["hog_cell_size"],
            hog_bins=CALIBRATION_BEST_CONFIG["hog_bins"],
            hog_extra_cell_sizes=CALIBRATION_BEST_CONFIG["hog_extra_cell_sizes"],
            use_lbp_features=CALIBRATION_BEST_CONFIG["use_lbp_features"],
            lbp_cell_size=CALIBRATION_BEST_CONFIG["lbp_cell_size"],
            lbp_bins=CALIBRATION_BEST_CONFIG["lbp_bins"],
            lbp_extra_cell_sizes=CALIBRATION_BEST_CONFIG["lbp_extra_cell_sizes"],
            use_entropy_features=CALIBRATION_BEST_CONFIG["use_entropy_features"],
            entropy_order=CALIBRATION_BEST_CONFIG["entropy_order"],
            use_corr_features=CALIBRATION_BEST_CONFIG["use_corr_features"],
            n_splits=args.cv_folds,
            random_state=args.cv_random_state,
            cv_epochs=None,
            augment_enabled=True,
            difficult_classes=DEFAULT_DIFFICULT_CLASSES,
            label_smoothing=CALIBRATION_BEST_CONFIG["label_smoothing"],
            use_poly_features=False,
            poly_pca_components=100,
            poly_degree=2,
            reg_scale=CALIBRATION_BEST_CONFIG["reg_scale"],
            use_specialist=CALIBRATION_BEST_CONFIG["use_specialist"],
        )


if __name__ == "__main__":
    main()

