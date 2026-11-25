"""
XGBoost multi-horizon traffic speed prediction model.
"""

import os
import joblib
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.utils.config import MODELS_DIR, PREDICTION_HORIZONS_MIN


class TrafficXGBoostModel:
    """
    XGBoost multi-horizon regressor for traffic speed prediction.
    Maintains dedicated optimized gradient boosting trees for 15, 30, and 60-minute horizons.
    """
    def __init__(
        self,
        n_estimators: int = 120,
        max_depth: int = 6,
        learning_rate: float = 0.08,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "objective": "reg:squarederror",
            "eval_metric": "mae"
        }
        self.models: Dict[int, xgb.XGBRegressor] = {}
        self.feature_names: Optional[List[str]] = None
        self.horizons_min = PREDICTION_HORIZONS_MIN

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Train separate gradient boosted trees for each prediction horizon.
        X_train: shape (N_samples, N_features)
        Y_train: shape (N_samples, len(horizons_min))
        """
        self.feature_names = feature_names
        for h_idx, h_min in enumerate(self.horizons_min):
            y_train_h = Y_train[:, h_idx]
            model = xgb.XGBRegressor(**self.params)
            
            if X_val is not None and Y_val is not None:
                y_val_h = Y_val[:, h_idx]
                eval_set = [(X_val, y_val_h)]
                model.fit(X_train, y_train_h, eval_set=eval_set, verbose=False)
            else:
                model.fit(X_train, y_train_h, verbose=False)
                
            self.models[h_min] = model

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict speed across all horizons.
        Returns: shape (N_samples, len(horizons_min))
        """
        preds = []
        for h_min in self.horizons_min:
            model = self.models.get(h_min)
            if model is None:
                raise RuntimeError(f"Model for horizon {h_min}m not fitted or loaded.")
            p = model.predict(X)
            preds.append(p)
            
        return np.column_stack(preds).astype(np.float32)

    def get_feature_importances(self) -> Dict[int, Dict[str, float]]:
        """Return feature importance scores per horizon."""
        importances = {}
        for h_min, model in self.models.items():
            scores = model.feature_importances_
            if self.feature_names and len(self.feature_names) == len(scores):
                importances[h_min] = {name: round(float(score), 4) for name, score in zip(self.feature_names, scores)}
            else:
                importances[h_min] = {f"feat_{i}": round(float(score), 4) for i, score in enumerate(scores)}
        return importances

    def save(self, filepath: Optional[Path] = None):
        """Save trained models to disk."""
        if filepath is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = MODELS_DIR / "xgboost_model.joblib"
            
        data = {
            "models": self.models,
            "params": self.params,
            "feature_names": self.feature_names,
            "horizons_min": self.horizons_min
        }
        joblib.dump(data, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Optional[Path] = None):
        """Load trained models from disk."""
        if filepath is None:
            filepath = MODELS_DIR / "xgboost_model.joblib"
        data = joblib.load(filepath)
        instance = cls()
        instance.models = data["models"]
        instance.params = data["params"]
        instance.feature_names = data.get("feature_names")
        instance.horizons_min = data.get("horizons_min", PREDICTION_HORIZONS_MIN)
        return instance
