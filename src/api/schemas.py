"""
Pydantic schemas for FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SensorMetadata(BaseModel):
    sensor_id: str
    highway: str
    direction: str
    latitude: float
    longitude: float
    speed_limit: float


class HorizonPrediction(BaseModel):
    horizon_minutes: int
    predicted_speed_mph: float
    congestion_class: str
    color_code: str


class SensorPredictionResponse(BaseModel):
    sensor_id: str
    model_name: str
    last_speed_mph: float
    current_congestion: str
    predictions: List[HorizonPrediction]


class NetworkPredictionItem(BaseModel):
    sensor_id: str
    latitude: float
    longitude: float
    current_speed: float
    current_congestion: str
    speed_15m: float
    congestion_15m: str
    speed_30m: float
    congestion_30m: str
    speed_60m: float
    congestion_60m: str


class NetworkPredictionResponse(BaseModel):
    model_name: str
    total_sensors: int
    predictions: List[NetworkPredictionItem]


class SensorPredictRequest(BaseModel):
    sensor_id: str = Field(..., description="Sensor identifier, e.g. '7100000'")
    history_speeds: Optional[List[float]] = Field(
        None,
        description="List of past 12 speed readings at 5-minute intervals (60 mins history). If omitted, latest known readings are used."
    )
    model_name: str = Field(
        "historical_average",
        description="Model to use: 'historical_average', 'xgboost', 'pytorch_stgcn', or 'persistence'"
    )


class NetworkPredictRequest(BaseModel):
    model_name: str = Field(
        "historical_average",
        description="Model to use: 'historical_average', 'xgboost', 'pytorch_stgcn', or 'persistence'"
    )
    timestamp_idx: Optional[int] = Field(
        -1,
        description="Index of test sample/timestamp to simulate. -1 represents latest available."
    )


class HealthResponse(BaseModel):
    status: str
    models_available: List[str]
    num_sensors: int
    horizons_minutes: List[int]
    congestion_thresholds: Dict[str, float]


class CorridorSummary(BaseModel):
    name: str
    full_name: str
    num_sensors: int
    avg_speed_mph: float
    avg_speed_kmh: float
    congestion_class: str
    color_code: str
    representative_sensor: str
    length_miles: float
    estimated_travel_time_min: float
    delay_min: float


class SensorTrajectoryResponse(BaseModel):
    sensor_id: str
    highway: str
    direction: str
    latitude: float
    longitude: float
    speed_limit: float
    segment_length_miles: float
    timestamps_history: List[str]
    speeds_history: List[float]
    timestamps_forecast: List[str]
    speeds_forecast: List[float]
    ground_truth: Optional[List[float]] = None
    current_speed_mph: float
    current_speed_kmh: float
    current_congestion: str
    current_travel_time_min: float
    free_flow_travel_time_min: float
    current_delay_min: float
    forecast_15m: float
    forecast_30m: float
    forecast_60m: float
    forecast_congestion_15m: str
    forecast_congestion_30m: str
    forecast_congestion_60m: str
    confidence_score: float
    data_sufficiency_status: str
    method_explanation: str


class DepartureTimeSlotResult(BaseModel):
    offset_minutes: int
    departure_label: str
    clock_time: str
    predicted_speed_mph: float
    predicted_speed_kmh: float
    estimated_travel_time_min: float
    free_flow_time_min: float
    delay_min: float
    congestion_class: str
    color_code: str


class DepartureTimeCompareRequest(BaseModel):
    sensor_id: str = "7100000"
    model_name: str = "historical_average"
    departure_offsets: List[int] = [0, 15, 30, 45, 60]


class DepartureTimeCompareResponse(BaseModel):
    sensor_id: str
    highway: str
    direction: str
    segment_length_miles: float
    slots: List[DepartureTimeSlotResult]
    best_departure_label: str
    time_saved_min: float
    advice: str


class RoadCompareRequest(BaseModel):
    highway_names: List[str] = ["I-405", "I-10", "US-101", "I-110", "I-5"]
    model_name: str = "historical_average"
    departure_offset_min: int = 0


class RoadCompareItem(BaseModel):
    highway: str
    full_name: str
    distance_miles: float
    avg_speed_mph: float
    avg_speed_kmh: float
    estimated_travel_time_min: float
    free_flow_time_min: float
    delay_min: float
    congestion_class: str
    color_code: str


class RoadCompareResponse(BaseModel):
    roads: List[RoadCompareItem]
    recommended_highway: str
    recommendation_reason: str


class CsvValidationResponse(BaseModel):
    is_valid: bool
    total_rows: int
    detected_columns: List[str]
    missing_columns: List[str]
    invalid_columns: List[str]
    summary_message: str
    warnings: List[str]
    sample_preview: List[Dict[str, Any]]
    overview_metrics: Optional[Dict[str, Any]] = None


class MLPCongestionRequest(BaseModel):
    road_segment: str = "I-405 North · Sepulveda Pass"
    traffic_volume: float = 2450.0
    occupancy: float = 42.5
    temperature: float = 24.0
    rainfall: float = 0.0
    time_of_day: str = "08:30"
    day_of_week: str = "Monday"
    save_history: bool = True
    user_email: Optional[str] = None


class MLPSpeedRequest(BaseModel):
    road_segment: str = "I-405 North · Sepulveda Pass"
    traffic_volume: float = 2450.0
    occupancy: float = 42.5
    temperature: float = 24.0
    rainfall: float = 0.0
    time_of_day: str = "08:30"
    day_of_week: str = "Monday"
    save_history: bool = True
    user_email: Optional[str] = None


class PeakForecastRequest(BaseModel):
    road_segment: str = "I-405 North · Sepulveda Pass"
    current_volume: float = 2450.0
    current_occupancy: float = 42.5
    current_speed: float = 47.0
    temperature: float = 24.0
    rainfall: float = 0.0
    hour: int = 8
    day_of_week: int = 0
    save_history: bool = True
    user_email: Optional[str] = None


class DatasetPreprocessRequest(BaseModel):
    missing_strategy: str = "mean"
    normalize: bool = True
    encode_cats: bool = True
    test_ratio: float = 0.15
    val_ratio: float = 0.15


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str = "Traffic analyst"


class LoginRequest(BaseModel):
    email: str
    password: str


class QuickLoginRequest(BaseModel):
    email: str
    name: str
    role: str = "Traffic analyst"


class UserStateUpdateRequest(BaseModel):
    state: Dict[str, Any]


