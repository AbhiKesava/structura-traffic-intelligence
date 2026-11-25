"""
Data loader and ingestion for the METR-LA traffic dataset.
Handles loading existing H5/NPZ/CSV datasets as well as realistic synthetic data generation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from src.utils.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    NUM_SENSORS_METR_LA,
    SAMPLING_INTERVAL_MINUTES,
    RANDOM_SEED
)


# Representative coordinates for Los Angeles Highway System sensors (I-5, I-10, US-101, I-405, I-110, I-210)
LA_HIGHWAYS = [
    {"name": "I-405", "start_lat": 33.850, "start_lon": -118.300, "end_lat": 34.180, "end_lon": -118.480, "weight": 0.25},
    {"name": "I-10",  "start_lat": 34.020, "start_lon": -118.480, "end_lat": 34.070, "end_lon": -118.150, "weight": 0.25},
    {"name": "US-101","start_lat": 34.050, "start_lon": -118.240, "end_lat": 34.160, "end_lon": -118.520, "weight": 0.20},
    {"name": "I-110", "start_lat": 33.740, "start_lon": -118.280, "end_lat": 34.070, "end_lon": -118.240, "weight": 0.15},
    {"name": "I-5",   "start_lat": 33.900, "start_lon": -118.100, "end_lat": 34.200, "end_lon": -118.400, "weight": 0.15},
]


def generate_sensor_metadata(num_sensors: int = NUM_SENSORS_METR_LA, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate realistic sensor metadata with LA coordinates along major corridors."""
    np.random.seed(seed)
    sensors = []
    
    sensors_per_hwy = []
    total_assigned = 0
    for h in LA_HIGHWAYS[:-1]:
        cnt = int(num_sensors * h["weight"])
        sensors_per_hwy.append((h, cnt))
        total_assigned += cnt
    sensors_per_hwy.append((LA_HIGHWAYS[-1], num_sensors - total_assigned))

    sensor_idx = 0
    for hwy, cnt in sensors_per_hwy:
        fractions = np.linspace(0.05, 0.95, cnt) + np.random.uniform(-0.02, 0.02, cnt)
        fractions = np.clip(fractions, 0, 1)
        fractions.sort()
        
        for frac in fractions:
            lat = hwy["start_lat"] + frac * (hwy["end_lat"] - hwy["start_lat"]) + np.random.normal(0, 0.003)
            lon = hwy["start_lon"] + frac * (hwy["end_lon"] - hwy["start_lon"]) + np.random.normal(0, 0.003)
            sensor_id = f"7{sensor_idx+100000:06d}"
            direction = np.random.choice(["N", "S"] if "0" in hwy["name"] or "5" in hwy["name"] else ["E", "W"])
            
            sensors.append({
                "sensor_id": sensor_id,
                "highway": hwy["name"],
                "direction": direction,
                "latitude": round(float(lat), 6),
                "longitude": round(float(lon), 6),
                "speed_limit": 65.0
            })
            sensor_idx += 1

    return pd.DataFrame(sensors)


def compute_adjacency_matrix(
    sensor_df: pd.DataFrame,
    sigma: float = 0.1,
    epsilon: float = 0.1
) -> np.ndarray:
    """
    Compute thresholded Gaussian distance kernel adjacency matrix for the sensors:
    W_ij = exp(- (dist(i, j) / sigma)^2) if dist(i, j) <= threshold, else 0
    """
    coords = sensor_df[["latitude", "longitude"]].values
    num_sensors = len(coords)
    
    # Pairwise Euclidean distance in coordinate space (scaled approx to km)
    # 1 deg lat ~ 111 km, 1 deg lon at lat 34 ~ 92 km
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    diff[..., 0] *= 111.0
    diff[..., 1] *= 92.0
    distances_km = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Thresholded Gaussian kernel
    std_dist = np.std(distances_km[distances_km > 0])
    if std_dist == 0:
        std_dist = 1.0
    adj_mx = np.exp(- (distances_km / (std_dist * sigma)) ** 2)
    adj_mx[adj_mx < epsilon] = 0.0
    np.fill_diagonal(adj_mx, 1.0)
    return adj_mx


def generate_synthetic_metr_la(
    num_days: int = 14,
    num_sensors: int = NUM_SENSORS_METR_LA,
    output_dir: Path = RAW_DATA_DIR,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Generate realistic multi-day 5-minute traffic speed time series data mimicking METR-LA.
    Includes diurnal cycles (morning/evening rush hours), weekend relaxation, spatial correlation,
    and temporary bottlenecks.
    """
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sensor metadata and adjacency
    sensor_df = generate_sensor_metadata(num_sensors=num_sensors, seed=seed)
    adj_mx = compute_adjacency_matrix(sensor_df)

    # Date range at 5-minute intervals
    start_date = pd.Timestamp("2024-03-01 00:00:00")
    total_steps = num_days * 24 * (60 // SAMPLING_INTERVAL_MINUTES)
    timestamps = pd.date_range(start=start_date, periods=total_steps, freq=f"{SAMPLING_INTERVAL_MINUTES}min")

    minute_of_day_arr = np.array(timestamps.hour * 60 + timestamps.minute, dtype=np.float32)
    day_of_week_arr = np.array(timestamps.dayofweek, dtype=np.float32)
    is_weekend = (day_of_week_arr >= 5).astype(np.float32)

    # Base speed profiles
    # Normal free flow: 60 - 68 mph
    base_speed = 62.0

    # Diurnal commuter curve:
    # Morning rush peak at 8:00 AM (480 min), evening rush peak at 5:30 PM (1050 min)
    morning_peak = 8 * 60
    evening_peak = 17.5 * 60

    # Gaussian dips in speed during rush hours (weekday effect)
    morning_dip = 24.0 * np.exp(-((minute_of_day_arr - morning_peak) / 75.0) ** 2)
    evening_dip = 28.0 * np.exp(-((minute_of_day_arr - evening_peak) / 90.0) ** 2)
    weekday_congestion = (morning_dip + evening_dip) * (1.0 - 0.65 * is_weekend)

    # Sensor speed matrix: time x sensors
    speeds = np.zeros((total_steps, num_sensors), dtype=np.float32)

    # Highway-specific vulnerability factor
    hwy_factors = {"I-405": 1.25, "I-10": 1.20, "US-101": 1.15, "I-110": 1.05, "I-5": 1.10}
    sensor_vuln = np.array([hwy_factors.get(row["highway"], 1.0) for _, row in sensor_df.iterrows()])

    # Generate correlated background noise using adjacency smoothing
    for t in range(total_steps):
        time_drop = weekday_congestion[t] * sensor_vuln
        # Sensor specific variation
        noise = np.random.normal(0, 3.5, size=num_sensors)
        # Spatial smoothing across connected neighbors
        spatial_noise = np.dot(adj_mx, noise) / np.maximum(adj_mx.sum(axis=1), 1.0)
        
        speed_t = base_speed - time_drop + spatial_noise
        
        # Occasional localized severe incident bottlenecks
        if np.random.rand() < 0.005:  # 0.5% chance per timestep of an incident
            incident_sensor = np.random.randint(0, num_sensors)
            affected = np.where(adj_mx[incident_sensor] > 0.4)[0]
            speed_t[affected] -= np.random.uniform(15.0, 30.0)

        speeds[t, :] = np.clip(speed_t, 5.0, 75.0)

    # Create speed DataFrame
    speed_df = pd.DataFrame(speeds, index=timestamps, columns=sensor_df["sensor_id"].values)

    # Save to disk
    speed_csv_path = output_dir / "metr_la_speeds.csv"
    sensor_csv_path = output_dir / "sensor_metadata.csv"
    adj_pkl_path = output_dir / "adj_mx.pkl"

    speed_df.to_csv(speed_csv_path)
    sensor_df.to_csv(sensor_csv_path, index=False)
    with open(adj_pkl_path, "wb") as f:
        pickle.dump({"sensor_ids": list(sensor_df["sensor_id"].values), "adj_mx": adj_mx}, f)

    return speed_df, adj_mx, sensor_df


def load_metr_la(
    data_dir: Path = RAW_DATA_DIR,
    auto_generate_if_missing: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Load METR-LA dataset. If not found on disk, auto-generates realistic dataset.
    
    Returns:
        speed_df: DataFrame of sensor speeds (index: datetime, columns: sensor_ids)
        adj_mx: Adjacency matrix of sensor network (shape: N x N)
        sensor_df: DataFrame with sensor metadata (sensor_id, highway, lat, lon)
    """
    speed_csv = data_dir / "metr_la_speeds.csv"
    sensor_csv = data_dir / "sensor_metadata.csv"
    adj_pkl = data_dir / "adj_mx.pkl"

    if speed_csv.exists() and sensor_csv.exists() and adj_pkl.exists():
        speed_df = pd.read_csv(speed_csv, index_col=0, parse_dates=True)
        speed_df.columns = [str(c) for c in speed_df.columns]
        sensor_df = pd.read_csv(sensor_csv, dtype={"sensor_id": str})
        sensor_df["sensor_id"] = sensor_df["sensor_id"].astype(str)
        with open(adj_pkl, "rb") as f:
            adj_data = pickle.load(f)
            adj_mx = adj_data["adj_mx"]
        return speed_df, adj_mx, sensor_df

    if auto_generate_if_missing:
        return generate_synthetic_metr_la(num_days=14, output_dir=data_dir)
    else:
        raise FileNotFoundError(f"METR-LA data files not found in {data_dir}")
