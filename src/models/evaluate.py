"""
Evaluation metrics for speed regression and congestion classification.
"""

import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.models.classifier import speeds_to_class_indices
from src.utils.config import CONGESTION_CLASSES, PREDICTION_HORIZONS_MIN


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE, RMSE, and MAPE metrics for regression predictions.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
    rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
    
    # MAPE calculation with epsilon protection against div by zero
    non_zero_mask = y_true_flat > 1.0
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs((y_true_flat[non_zero_mask] - y_pred_flat[non_zero_mask]) / y_true_flat[non_zero_mask])) * 100.0)
    else:
        mape = 0.0

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "MAPE": round(mape, 2)
    }


def compute_classification_metrics(y_true_speeds: np.ndarray, y_pred_speeds: np.ndarray) -> Dict[str, Any]:
    """
    Compute classification metrics after mapping speeds to congestion categories.
    """
    y_true_cls = speeds_to_class_indices(y_true_speeds).flatten()
    y_pred_cls = speeds_to_class_indices(y_pred_speeds).flatten()

    acc = float(accuracy_score(y_true_cls, y_pred_cls))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, average="macro", zero_division=0
    )

    labels = [0, 1, 2, 3]
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels).tolist()

    return {
        "Accuracy": round(acc * 100.0, 2),
        "Macro_Precision": round(float(p) * 100.0, 2),
        "Macro_Recall": round(float(r) * 100.0, 2),
        "Macro_F1": round(float(f1) * 100.0, 2),
        "Confusion_Matrix": cm,
        "Class_Labels": CONGESTION_CLASSES
    }


def evaluate_multi_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons_min: List[int] = PREDICTION_HORIZONS_MIN
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate speed predictions across all horizons (15m, 30m, 60m) as well as overall.
    y_true, y_pred: shape (N_samples, len(horizons)) or (N_samples, len(horizons), N_sensors)
    """
    results = {}
    
    # If 3D, reshape to (N_samples * N_sensors, len(horizons))
    if y_true.ndim == 3:
        # permute and reshape so horizon is dim 1
        # e.g., (B, H, N) -> (B * N, H)
        B, H, N = y_true.shape
        y_true = np.swapaxes(y_true, 1, 2).reshape(-1, H)
        y_pred = np.swapaxes(y_pred, 1, 2).reshape(-1, H)

    for h_idx, h_min in enumerate(horizons_min):
        true_h = y_true[:, h_idx]
        pred_h = y_pred[:, h_idx]

        reg_metrics = compute_regression_metrics(true_h, pred_h)
        cls_metrics = compute_classification_metrics(true_h, pred_h)

        results[f"{h_min}min"] = {**reg_metrics, **cls_metrics}

    # Overall aggregated metrics
    reg_overall = compute_regression_metrics(y_true, y_pred)
    cls_overall = compute_classification_metrics(y_true, y_pred)
    results["Overall"] = {**reg_overall, **cls_overall}

    return results
