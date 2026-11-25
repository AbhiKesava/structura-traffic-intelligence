"""
Centralized Configuration for Traffic Congestion Prediction System.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models_saved"

# Dataset parameters
SAMPLING_INTERVAL_MINUTES = 5
NUM_SENSORS_METR_LA = 207

# Prediction horizons: 15, 30, 60 minutes
# Since data is at 5-minute intervals, these correspond to 3, 6, 12 steps
PREDICTION_HORIZONS_MIN = [15, 30, 60]
PREDICTION_HORIZONS_STEPS = [h // SAMPLING_INTERVAL_MINUTES for h in PREDICTION_HORIZONS_MIN]
HORIZON_MAP = {15: 3, 30: 6, 60: 12}

# History input window: e.g. 12 steps (60 minutes of history)
HISTORY_STEPS = 12

# Train / Val / Test split ratios (sequential temporal split)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Congestion Classification Thresholds (in mph)
# Free Flow: >= 50 mph
# Moderate: 35 <= speed < 50 mph
# Heavy: 20 <= speed < 35 mph
# Severe: < 20 mph
CONGESTION_THRESHOLDS = {
    "FREE_FLOW": 50.0,
    "MODERATE": 35.0,
    "HEAVY": 20.0
}

CONGESTION_CLASSES = ["Free", "Moderate", "Heavy", "Severe"]
CONGESTION_COLOR_MAP = {
    "Free": "#22c55e",       # Green
    "Moderate": "#eab308",   # Yellow
    "Heavy": "#f97316",      # Orange
    "Severe": "#ef4444"      # Red
}

def speed_to_congestion(speed: float) -> str:
    """Map a continuous speed value (mph) to a congestion category."""
    if speed >= CONGESTION_THRESHOLDS["FREE_FLOW"]:
        return "Free"
    elif speed >= CONGESTION_THRESHOLDS["MODERATE"]:
        return "Moderate"
    elif speed >= CONGESTION_THRESHOLDS["HEAVY"]:
        return "Heavy"
    else:
        return "Severe"

RANDOM_SEED = 42
