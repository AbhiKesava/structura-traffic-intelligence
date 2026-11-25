"""
Baseline models: Historical Average (HA) and Persistence (Last Observed Value).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.utils.config import PREDICTION_HORIZONS_STEPS, PREDICTION_HORIZONS_MIN


class PersistenceBaseline:
    """
    Persistence (Naive) baseline: predicts the last observed speed for all future horizons.
    """
    def __init__(self):
        pass

    def fit(self, speed_df: pd.DataFrame):
        return self

    def predict_from_lags(self, last_speed: np.ndarray, num_horizons: int = 3) -> np.ndarray:
        """
        last_speed: shape (N,) or (B, N) or (B,)
        Returns: shape (..., num_horizons)
        """
        last_speed = np.asarray(last_speed)
        return np.repeat(last_speed[..., np.newaxis], num_horizons, axis=-1)

    def predict_sequence(
        self,
        X_seq: np.ndarray,
        num_horizons: int = 3
    ) -> np.ndarray:
        """
        X_seq: shape (B, history_steps, N) or (B, history_steps)
        Returns: shape (B, num_horizons, N) or (B, num_horizons)
        """
        last_val = X_seq[:, -1]  # shape (B, N) or (B,)
        if last_val.ndim == 2:
            # (B, N) -> (B, num_horizons, N)
            return np.repeat(last_val[:, np.newaxis, :], num_horizons, axis=1)
        else:
            # (B,) -> (B, num_horizons)
            return np.repeat(last_val[:, np.newaxis], num_horizons, axis=1)


class HistoricalAverageBaseline:
    """
    Historical Average baseline: computes historical mean per (day_of_week, 5-minute interval).
    """
    def __init__(self):
        self.profile: Optional[pd.DataFrame] = None

    def fit(self, speed_df: pd.DataFrame):
        """
        Fits speed profile grouped by day-of-week and time-of-day interval (0 to 287).
        """
        dow = speed_df.index.dayofweek
        slot = speed_df.index.hour * 12 + speed_df.index.minute // 5

        df_copy = speed_df.copy()
        df_copy["_dow"] = dow
        df_copy["_slot"] = slot

        # Group by dow and slot and compute mean across time
        self.profile = df_copy.groupby(["_dow", "_slot"]).mean()
        return self

    def predict_for_timestamp(
        self,
        target_timestamp: pd.Timestamp,
        sensor_ids: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Predict historical average speed at a specific future timestamp.
        """
        dow = target_timestamp.dayofweek
        slot = target_timestamp.hour * 12 + target_timestamp.minute // 5
        
        try:
            row = self.profile.loc[(dow, slot)]
            if sensor_ids is not None:
                return row[sensor_ids].values.astype(np.float32)
            return row.values.astype(np.float32)
        except KeyError:
            # Fallback to global mean
            return self.profile.mean(axis=0).values.astype(np.float32)

    def predict_multi_horizon(
        self,
        current_timestamp: pd.Timestamp,
        horizon_minutes: List[int] = PREDICTION_HORIZONS_MIN,
        sensor_ids: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Predict for multi-horizons [15m, 30m, 60m].
        Returns: shape (len(horizon_minutes), num_sensors)
        """
        preds = []
        for h in horizon_minutes:
            future_ts = current_timestamp + pd.Timedelta(minutes=h)
            p = self.predict_for_timestamp(future_ts, sensor_ids=sensor_ids)
            preds.append(p)
        return np.array(preds, dtype=np.float32)
