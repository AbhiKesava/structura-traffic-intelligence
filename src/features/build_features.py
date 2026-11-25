"""
Feature extraction and tabular dataset construction for machine learning models (XGBoost).
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from src.utils.config import (
    HISTORY_STEPS,
    PREDICTION_HORIZONS_STEPS,
    PREDICTION_HORIZONS_MIN
)


def extract_cyclical_time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Extract cyclical sin/cos representations of time of day and day of week.
    Returns: array of shape (len(timestamps), 5):
      [sin_time, cos_time, sin_day, cos_day, is_weekend]
    """
    minute_of_day = timestamps.hour * 60 + timestamps.minute
    total_minutes = 24 * 60

    sin_time = np.sin(2 * np.pi * minute_of_day / total_minutes)
    cos_time = np.cos(2 * np.pi * minute_of_day / total_minutes)

    day_of_week = timestamps.dayofweek
    sin_day = np.sin(2 * np.pi * day_of_week / 7.0)
    cos_day = np.cos(2 * np.pi * day_of_week / 7.0)
    is_weekend = (day_of_week >= 5).astype(float)

    return np.column_stack([sin_time, cos_time, sin_day, cos_day, is_weekend]).astype(np.float32)


def build_sensor_tabular_features(
    speed_df: pd.DataFrame,
    sensor_idx: int,
    adj_mx: Optional[np.ndarray] = None,
    history_steps: int = HISTORY_STEPS,
    horizon_steps: List[int] = PREDICTION_HORIZONS_STEPS,
    top_k_neighbors: int = 5
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build rich tabular feature matrix for a specific sensor.
    
    Features engineered:
      - Raw lags: t, t-1, t-2, t-3 (15m), t-5 (25m), t-11 (55m)
      - Rolling stats: 15-min mean/std, 60-min mean/std
      - Speed acceleration/deltas: (t - t-2), (t - t-5)
      - Neighboring sensors average speed at t (if adj_mx provided)
      - Cyclical temporal encodings (sin/cos of hour and day of week)
      
    Targets:
      - Speed at horizons: t+3 (15m), t+6 (30m), t+12 (60m)
    """
    speeds = speed_df.values  # (T, N)
    timestamps = speed_df.index
    T, N = speeds.shape
    max_h = max(horizon_steps)
    num_samples = T - history_steps - max_h + 1

    s_speed = speeds[:, sensor_idx]
    time_feats = extract_cyclical_time_features(timestamps)

    # Precompute neighbor indices if adj_mx given
    neighbor_indices = []
    if adj_mx is not None:
        row_adj = adj_mx[sensor_idx].copy()
        row_adj[sensor_idx] = 0.0  # exclude self
        top_k = min(top_k_neighbors, N - 1)
        neighbor_indices = np.argsort(row_adj)[-top_k:]

    feature_rows = []
    target_rows = []
    
    feature_names = [
        "speed_t", "speed_t_minus_1", "speed_t_minus_2", "speed_t_minus_3",
        "speed_t_minus_5", "speed_t_minus_11",
        "roll_mean_15m", "roll_std_15m",
        "roll_mean_60m", "roll_std_60m",
        "delta_15m", "delta_30m",
        "neighbor_mean_speed",
        "sin_time", "cos_time", "sin_day", "cos_day", "is_weekend"
    ]

    for i in range(num_samples):
        # Current index in raw series is t = i + history_steps - 1
        curr_t = i + history_steps - 1
        history_window = s_speed[curr_t - history_steps + 1 : curr_t + 1]

        # Lags
        v_t = history_window[-1]
        v_t1 = history_window[-2]
        v_t2 = history_window[-3]
        v_t3 = history_window[-4]
        v_t5 = history_window[-6]
        v_t11 = history_window[0]

        # Rolling statistics
        roll_15 = history_window[-3:]
        roll_60 = history_window
        mean_15 = np.mean(roll_15)
        std_15 = np.std(roll_15)
        mean_60 = np.mean(roll_60)
        std_60 = np.std(roll_60)

        # Deltas
        delta_15 = v_t - v_t3
        delta_30 = v_t - history_window[-7]

        # Neighbor speed
        if len(neighbor_indices) > 0:
            neigh_speed = np.mean(speeds[curr_t, neighbor_indices])
        else:
            neigh_speed = v_t

        # Time features at curr_t
        t_feats = time_feats[curr_t]

        row = [
            v_t, v_t1, v_t2, v_t3, v_t5, v_t11,
            mean_15, std_15, mean_60, std_60,
            delta_15, delta_30,
            neigh_speed,
            t_feats[0], t_feats[1], t_feats[2], t_feats[3], t_feats[4]
        ]
        feature_rows.append(row)

        # Targets
        targets = [s_speed[curr_t + h] for h in horizon_steps]
        target_rows.append(targets)

    X = np.array(feature_rows, dtype=np.float32)
    Y = np.array(target_rows, dtype=np.float32)

    return X, Y, feature_names


def build_multi_sensor_tabular_dataset(
    speed_df: pd.DataFrame,
    adj_mx: Optional[np.ndarray] = None,
    sensor_sample_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build pooled tabular dataset across a representative sample of sensors for fast training.
    """
    np.random.seed(seed)
    N = speed_df.shape[1]
    num_selected = max(5, int(N * sensor_sample_ratio))
    selected_sensors = np.random.choice(N, size=num_selected, replace=False)

    all_X = []
    all_Y = []
    feature_names = None

    for s_idx in selected_sensors:
        X_s, Y_s, feat_names = build_sensor_tabular_features(
            speed_df, s_idx, adj_mx=adj_mx
        )
        all_X.append(X_s)
        all_Y.append(Y_s)
        feature_names = feat_names

    X_pooled = np.concatenate(all_X, axis=0)
    Y_pooled = np.concatenate(all_Y, axis=0)
    return X_pooled, Y_pooled, feature_names
