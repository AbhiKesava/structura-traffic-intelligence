"""
End-to-end model training pipeline: Baselines, XGBoost, and PyTorch ST-GCN.
"""

import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from src.utils.config import (
    MODELS_DIR,
    HISTORY_STEPS,
    PREDICTION_HORIZONS_STEPS,
    PREDICTION_HORIZONS_MIN,
    RANDOM_SEED
)
from src.data.loader import load_metr_la
from src.data.preprocessor import temporal_split, TrafficStandardScaler, create_spatiotemporal_windows
from src.features.build_features import build_multi_sensor_tabular_dataset, build_sensor_tabular_features
from src.models.baseline import PersistenceBaseline, HistoricalAverageBaseline
from src.models.xgboost_model import TrafficXGBoostModel
from src.models.pytorch_stgcn import SpatioTemporalGCN, STGCNTrainer, calculate_normalized_laplacian
from src.models.evaluate import evaluate_multi_horizon


def train_all_models(fast_train: bool = False, epochs_pytorch: int = 8):
    """
    Train and benchmark all models on the METR-LA dataset.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Starting Traffic Congestion Prediction Model Training Pipeline")
    print("=" * 60)

    # 1. Load Data
    t0 = time.time()
    speed_df, adj_mx, sensor_df = load_metr_la()
    print(f"[1/6] Data loaded: {speed_df.shape[0]} timestamps x {speed_df.shape[1]} sensors ({time.time() - t0:.2f}s)")

    # 2. Sequential Splitting & Scaling
    train_df, val_df, test_df = temporal_split(speed_df)
    scaler = TrafficStandardScaler().fit(train_df.values)
    print(f"[2/6] Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"      Train speed stats: Mean={scaler.mean:.2f} mph, Std={scaler.std:.2f} mph")

    # 3. Baselines Evaluation on Test Set
    print("\n[3/6] Evaluating Baseline Models...")
    # Slicing spatiotemporal windows for evaluation
    X_test_seq, Y_test_seq = create_spatiotemporal_windows(test_df.values)
    
    # Baseline 1: Persistence
    pers_model = PersistenceBaseline()
    y_pred_pers = pers_model.predict_sequence(X_test_seq)
    metrics_persistence = evaluate_multi_horizon(Y_test_seq, y_pred_pers)
    print(f"  -> Persistence Baseline Overall MAE: {metrics_persistence['Overall']['MAE']} mph, F1: {metrics_persistence['Overall']['Macro_F1']}%")

    # Baseline 2: Historical Average
    ha_model = HistoricalAverageBaseline().fit(train_df)
    test_sample_indices = range(len(X_test_seq))
    # Predict HA for test timestamps
    ha_preds = []
    max_h = max(PREDICTION_HORIZONS_STEPS)
    test_timestamps = test_df.index[HISTORY_STEPS - 1 : len(test_df) - max_h]
    for ts in test_timestamps:
        pred_ts = ha_model.predict_multi_horizon(ts)  # (3, N)
        ha_preds.append(pred_ts)
    ha_preds = np.array(ha_preds)  # (B, 3, N)
    metrics_ha = evaluate_multi_horizon(Y_test_seq[:len(ha_preds)], ha_preds)
    print(f"  -> Historical Average Overall MAE: {metrics_ha['Overall']['MAE']} mph, F1: {metrics_ha['Overall']['Macro_F1']}%")

    # 4. Train XGBoost Model
    print("\n[4/6] Training Multi-Horizon XGBoost Model...")
    sample_ratio = 0.08 if fast_train else 0.18
    X_train_xgb, Y_train_xgb, feat_names = build_multi_sensor_tabular_dataset(
        train_df, adj_mx=adj_mx, sensor_sample_ratio=sample_ratio, seed=RANDOM_SEED
    )
    X_val_xgb, Y_val_xgb, _ = build_multi_sensor_tabular_dataset(
        val_df, adj_mx=adj_mx, sensor_sample_ratio=sample_ratio, seed=RANDOM_SEED
    )
    print(f"  -> Tabular Training Samples: {X_train_xgb.shape[0]}, Features: {X_train_xgb.shape[1]}")

    n_estimators = 80 if fast_train else 120
    xgb_model = TrafficXGBoostModel(n_estimators=n_estimators, max_depth=6, learning_rate=0.08)
    xgb_model.fit(X_train_xgb, Y_train_xgb, X_val_xgb, Y_val_xgb, feature_names=feat_names)
    xgb_save_path = MODELS_DIR / "xgboost_model.joblib"
    xgb_model.save(xgb_save_path)
    print(f"  -> XGBoost saved to {xgb_save_path}")

    # Evaluate XGBoost across representative test sensors
    X_test_xgb, Y_test_xgb, _ = build_multi_sensor_tabular_dataset(
        test_df, adj_mx=adj_mx, sensor_sample_ratio=sample_ratio, seed=RANDOM_SEED
    )
    y_pred_xgb = xgb_model.predict(X_test_xgb)
    metrics_xgb = evaluate_multi_horizon(Y_test_xgb, y_pred_xgb)
    print(f"  -> XGBoost Overall MAE: {metrics_xgb['Overall']['MAE']} mph, F1: {metrics_xgb['Overall']['Macro_F1']}%")

    # 5. Train PyTorch Spatio-Temporal Graph Model (ST-GCN)
    print(f"\n[5/6] Training PyTorch Spatio-Temporal GCN ({epochs_pytorch} epochs)...")
    norm_adj = calculate_normalized_laplacian(adj_mx)
    X_train_seq, Y_train_seq = create_spatiotemporal_windows(train_df.values)
    X_val_seq, Y_val_seq = create_spatiotemporal_windows(val_df.values)

    train_dataset = TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(Y_train_seq))
    val_dataset = TensorDataset(torch.from_numpy(X_val_seq), torch.from_numpy(Y_val_seq))
    test_dataset = TensorDataset(torch.from_numpy(X_test_seq), torch.from_numpy(Y_test_seq))

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    stgcn_model = SpatioTemporalGCN(
        num_sensors=speed_df.shape[1],
        history_steps=HISTORY_STEPS,
        num_horizons=len(PREDICTION_HORIZONS_MIN),
        spatial_dim=32,
        hidden_dim=64
    )
    trainer = STGCNTrainer(
        model=stgcn_model,
        norm_adj=norm_adj,
        scaler_params=scaler.to_dict(),
        lr=0.003
    )

    best_val_mae = float("inf")
    for epoch in range(1, epochs_pytorch + 1):
        t_ep = time.time()
        train_loss = trainer.train_epoch(train_loader)
        val_mae, _, _ = trainer.evaluate(val_loader)
        print(f"  Epoch {epoch:02d}/{epochs_pytorch:02d} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.2f} mph | Time: {time.time() - t_ep:.1f}s")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            trainer.save(MODELS_DIR / "pytorch_stgcn.pt")

    # Evaluate PyTorch Model on Test Set
    best_stgcn = STGCNTrainer.load(MODELS_DIR / "pytorch_stgcn.pt", adj_mx)
    _, y_pred_stgcn, y_true_stgcn = best_stgcn.evaluate(test_loader)
    metrics_stgcn = evaluate_multi_horizon(y_true_stgcn, y_pred_stgcn)
    print(f"  -> PyTorch ST-GCN Overall MAE: {metrics_stgcn['Overall']['MAE']} mph, F1: {metrics_stgcn['Overall']['Macro_F1']}%")

    # 6. Save Consolidated Metrics
    print("\n[6/6] Consolidating and Saving Benchmarks...")
    all_metrics = {
        "Persistence": metrics_persistence,
        "Historical_Average": metrics_ha,
        "XGBoost": metrics_xgb,
        "PyTorch_STGCN": metrics_stgcn,
        "scaler": scaler.to_dict(),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Metrics saved to {metrics_path}")
    print("\n" + "=" * 60)
    print("Benchmark Comparison Summary (Overall):")
    print(f"{'Model':<22} | {'MAE (mph)':<10} | {'RMSE (mph)':<10} | {'Acc (%)':<9} | {'F1 (%)':<8}")
    print("-" * 65)
    for model_name, m_dict in [("Persistence", metrics_persistence),
                               ("Historical Average", metrics_ha),
                               ("XGBoost", metrics_xgb),
                               ("PyTorch ST-GCN", metrics_stgcn)]:
        m = m_dict["Overall"]
        print(f"{model_name:<22} | {m['MAE']:<10.2f} | {m['RMSE']:<10.2f} | {m['Accuracy']:<9.1f} | {m['Macro_F1']:<8.1f}")
    print("=" * 60)

    return all_metrics


if __name__ == "__main__":
    train_all_models(fast_train=False, epochs_pytorch=6)
