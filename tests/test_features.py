"""
Unit tests for feature engineering modules.
"""

import numpy as np
import pandas as pd
import pytest
from src.features.build_features import extract_cyclical_time_features, build_sensor_tabular_features


def test_cyclical_time_features():
    timestamps = pd.date_range("2024-03-01 00:00:00", periods=288, freq="5min")
    feats = extract_cyclical_time_features(timestamps)

    assert feats.shape == (288, 5)
    # Sin and cos must be bounded between -1 and 1
    assert np.all(feats[:, :4] >= -1.0) and np.all(feats[:, :4] <= 1.0)
    # Is_weekend column must be either 0.0 or 1.0
    assert set(np.unique(feats[:, 4])).issubset({0.0, 1.0})


def test_sensor_tabular_features():
    # 200 timesteps x 10 sensors
    dates = pd.date_range("2024-03-01 00:00:00", periods=200, freq="5min")
    speeds = np.random.uniform(25.0, 65.0, size=(200, 10)).astype(np.float32)
    df = pd.DataFrame(speeds, index=dates, columns=[f"s_{i}" for i in range(10)])

    adj_mx = np.eye(10, dtype=np.float32)
    X, Y, feat_names = build_sensor_tabular_features(df, sensor_idx=0, adj_mx=adj_mx)

    assert len(X) == 200 - 12 - 12 + 1
    assert X.shape[1] == len(feat_names)
    assert Y.shape[1] == 3  # 3 horizons (15, 30, 60 mins)
    assert not np.isnan(X).any()
    assert not np.isnan(Y).any()
