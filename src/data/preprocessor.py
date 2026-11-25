"""
Dataset splitting, normalization, and sequence windowing.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from src.utils.config import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    HISTORY_STEPS,
    PREDICTION_HORIZONS_STEPS,
    PREDICTION_HORIZONS_MIN
)


class TrafficStandardScaler:
    """Standard Z-score scaler fitted on training speed tensors."""
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def fit(self, data: np.ndarray):
        self.mean = float(np.nanmean(data))
        self.std = float(np.nanstd(data))
        if self.std < 1e-6:
            self.std = 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return (data * self.std) + self.mean

    def to_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: Dict[str, float]):
        return cls(mean=d["mean"], std=d["std"])


def temporal_split(
    speed_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series sequentially into train, validation, and test sets.
    Preserves strict temporal causality (no forward look-ahead).
    """
    n = len(speed_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = speed_df.iloc[:train_end]
    val_df = speed_df.iloc[train_end:val_end]
    test_df = speed_df.iloc[val_end:]

    return train_df, val_df, test_df


def create_spatiotemporal_windows(
    data: np.ndarray,
    history_steps: int = HISTORY_STEPS,
    horizon_steps: List[int] = PREDICTION_HORIZONS_STEPS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for spatio-temporal deep learning models (PyTorch).
    
    Args:
        data: shape (T, N) where T is timesteps and N is number of sensors.
        history_steps: number of historical steps to condition on (e.g. 12 = 60 mins).
        horizon_steps: list of step offsets to predict (e.g. [3, 6, 12] for 15, 30, 60 mins).
        
    Returns:
        X: shape (num_samples, history_steps, N)
        Y: shape (num_samples, len(horizon_steps), N)
    """
    T, N = data.shape
    max_horizon = max(horizon_steps)
    num_samples = T - history_steps - max_horizon + 1

    if num_samples <= 0:
        raise ValueError(f"Data length {T} is too short for history {history_steps} and max horizon {max_horizon}")

    X = np.zeros((num_samples, history_steps, N), dtype=np.float32)
    Y = np.zeros((num_samples, len(horizon_steps), N), dtype=np.float32)

    for i in range(num_samples):
        # History: [i, i + history_steps)
        X[i] = data[i : i + history_steps]
        # Targets at specified horizons
        # horizon 3 corresponds to index: i + history_steps + 3 - 1
        for h_idx, h_step in enumerate(horizon_steps):
            target_idx = i + history_steps + h_step - 1
            Y[i, h_idx] = data[target_idx]

    return X, Y
