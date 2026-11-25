"""
Speed-to-Congestion mapping and classification utilities.
"""

import numpy as np
from typing import Union, List
from src.utils.config import CONGESTION_THRESHOLDS, CONGESTION_CLASSES


# Integer encoding: 0: Free, 1: Moderate, 2: Heavy, 3: Severe
CLASS_TO_INT = {
    "Free": 0,
    "Moderate": 1,
    "Heavy": 2,
    "Severe": 3
}

INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}


def speed_to_class_label(speed: Union[float, int]) -> int:
    """Map single continuous speed to integer class."""
    if speed >= CONGESTION_THRESHOLDS["FREE_FLOW"]:
        return CLASS_TO_INT["Free"]
    elif speed >= CONGESTION_THRESHOLDS["MODERATE"]:
        return CLASS_TO_INT["Moderate"]
    elif speed >= CONGESTION_THRESHOLDS["HEAVY"]:
        return CLASS_TO_INT["Heavy"]
    else:
        return CLASS_TO_INT["Severe"]


def speeds_to_class_indices(speeds: np.ndarray) -> np.ndarray:
    """Vectorized mapping of speeds array to integer class indices [0, 1, 2, 3]."""
    classes = np.zeros(speeds.shape, dtype=np.int32)
    classes[speeds >= CONGESTION_THRESHOLDS["FREE_FLOW"]] = CLASS_TO_INT["Free"]
    classes[(speeds >= CONGESTION_THRESHOLDS["MODERATE"]) & (speeds < CONGESTION_THRESHOLDS["FREE_FLOW"])] = CLASS_TO_INT["Moderate"]
    classes[(speeds >= CONGESTION_THRESHOLDS["HEAVY"]) & (speeds < CONGESTION_THRESHOLDS["MODERATE"])] = CLASS_TO_INT["Heavy"]
    classes[speeds < CONGESTION_THRESHOLDS["HEAVY"]] = CLASS_TO_INT["Severe"]
    return classes


def speeds_to_class_names(speeds: np.ndarray) -> Union[str, List[str], np.ndarray]:
    """Map speeds to string class names."""
    indices = speeds_to_class_indices(np.asarray(speeds))
    if indices.ndim == 0:
        return INT_TO_CLASS[int(indices)]
    
    flat = indices.flatten()
    names = [INT_TO_CLASS[int(idx)] for idx in flat]
    return np.array(names).reshape(indices.shape)
