"""
Unit tests for models, evaluation metrics, and congestion classification.
"""

import numpy as np
import torch
import pytest
from src.utils.config import speed_to_congestion
from src.models.classifier import (
    speed_to_class_label,
    speeds_to_class_indices,
    speeds_to_class_names,
    CLASS_TO_INT
)
from src.models.evaluate import (
    compute_regression_metrics,
    compute_classification_metrics,
    evaluate_multi_horizon
)
from src.models.baseline import PersistenceBaseline
from src.models.pytorch_stgcn import SpatioTemporalGCN, calculate_normalized_laplacian


def test_speed_to_congestion_mapping():
    assert speed_to_congestion(65.0) == "Free"
    assert speed_to_congestion(50.0) == "Free"
    assert speed_to_congestion(45.0) == "Moderate"
    assert speed_to_congestion(35.0) == "Moderate"
    assert speed_to_congestion(28.0) == "Heavy"
    assert speed_to_congestion(20.0) == "Heavy"
    assert speed_to_congestion(15.0) == "Severe"
    assert speed_to_congestion(0.0) == "Severe"

    # Vectorized check
    speeds = np.array([55.0, 40.0, 25.0, 10.0])
    indices = speeds_to_class_indices(speeds)
    assert np.array_equal(indices, [0, 1, 2, 3])

    names = speeds_to_class_names(speeds)
    assert np.array_equal(names, ["Free", "Moderate", "Heavy", "Severe"])


def test_compute_metrics():
    y_true = np.array([[60.0, 50.0, 40.0], [55.0, 45.0, 35.0]], dtype=np.float32)
    y_pred = np.array([[58.0, 48.0, 38.0], [53.0, 43.0, 33.0]], dtype=np.float32)

    reg_metrics = compute_regression_metrics(y_true, y_pred)
    assert np.isclose(reg_metrics["MAE"], 2.0, atol=1e-2)
    assert np.isclose(reg_metrics["RMSE"], 2.0, atol=1e-2)
    assert reg_metrics["MAPE"] > 0.0

    cls_metrics = compute_classification_metrics(y_true, y_pred)
    assert cls_metrics["Accuracy"] >= 0.0
    assert "Confusion_Matrix" in cls_metrics


def test_persistence_baseline():
    baseline = PersistenceBaseline()
    X = np.random.uniform(30.0, 60.0, size=(10, 12, 5))
    preds = baseline.predict_sequence(X, num_horizons=3)

    assert preds.shape == (10, 3, 5)
    for h in range(3):
        assert np.allclose(preds[:, h, :], X[:, -1, :])


def test_pytorch_stgcn_forward():
    num_sensors = 10
    history = 12
    batch_size = 4
    
    adj = np.eye(num_sensors, dtype=np.float32)
    norm_adj = calculate_normalized_laplacian(adj)

    model = SpatioTemporalGCN(
        num_sensors=num_sensors,
        history_steps=history,
        num_horizons=3,
        spatial_dim=16,
        hidden_dim=32
    )
    
    x = torch.randn(batch_size, history, num_sensors)
    out = model(x, norm_adj)

    assert out.shape == (batch_size, 3, num_sensors)
