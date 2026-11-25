"""
Unit tests for data loading, preprocessing, and windowing.
"""

import numpy as np
import pandas as pd
import pytest
from src.data.loader import load_metr_la, generate_sensor_metadata, compute_adjacency_matrix
from src.data.preprocessor import temporal_split, TrafficStandardScaler, create_spatiotemporal_windows
from src.utils.config import HISTORY_STEPS, PREDICTION_HORIZONS_STEPS


def test_load_metr_la():
    speed_df, adj_mx, sensor_df = load_metr_la()
    assert isinstance(speed_df, pd.DataFrame)
    assert len(speed_df) > 1000
    assert speed_df.shape[1] == 207
    assert adj_mx.shape == (207, 207)
    assert len(sensor_df) == 207
    assert not speed_df.isnull().any().any(), "Speeds should not contain NaN values"


def test_sensor_metadata_and_adjacency():
    sensor_df = generate_sensor_metadata(num_sensors=20)
    assert len(sensor_df) == 20
    assert "latitude" in sensor_df.columns
    assert "longitude" in sensor_df.columns
    assert "highway" in sensor_df.columns

    adj_mx = compute_adjacency_matrix(sensor_df)
    assert adj_mx.shape == (20, 20)
    assert np.allclose(np.diag(adj_mx), 1.0), "Self-loops in adjacency must be 1.0"
    assert (adj_mx >= 0.0).all() and (adj_mx <= 1.0).all()


def test_temporal_split_no_leakage():
    speed_df, _, _ = load_metr_la()
    train_df, val_df, test_df = temporal_split(speed_df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

    assert len(train_df) + len(val_df) + len(test_df) == len(speed_df)
    assert train_df.index.max() < val_df.index.min(), "Data leakage: Train overlaps with Validation"
    assert val_df.index.max() < test_df.index.min(), "Data leakage: Validation overlaps with Test"


def test_traffic_standard_scaler():
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    scaler = TrafficStandardScaler()
    scaler.fit(data)

    assert np.isclose(scaler.mean, 30.0)
    scaled = scaler.transform(data)
    assert np.isclose(np.mean(scaled), 0.0, atol=1e-5)

    reconstructed = scaler.inverse_transform(scaled)
    assert np.allclose(reconstructed, data)


def test_create_spatiotemporal_windows():
    # Synthetic array with 100 timesteps and 5 sensors
    dummy_data = np.random.uniform(20.0, 60.0, size=(100, 5)).astype(np.float32)
    X, Y = create_spatiotemporal_windows(dummy_data, history_steps=12, horizon_steps=[3, 6, 12])

    expected_samples = 100 - 12 - 12 + 1
    assert X.shape == (expected_samples, 12, 5)
    assert Y.shape == (expected_samples, 3, 5)
    # Check that horizon 3 (15 min) in Y corresponds to timestep 12 + 3 - 1 = 14
    assert np.allclose(Y[0, 0], dummy_data[14])
    # Check horizon 12 (60 min) corresponds to timestep 12 + 12 - 1 = 23
    assert np.allclose(Y[0, 2], dummy_data[23])
