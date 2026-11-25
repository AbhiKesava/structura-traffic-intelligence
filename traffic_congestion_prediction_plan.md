# Implementation Plan: Traffic Congestion Prediction System

## 1. Project Overview & Objectives
- **Goal:** Predict highway traffic speed at 15, 30, and 60 minutes into the future across sensor locations, and map predicted speeds into standardized congestion severity categories.
- **Dataset:** **METR-LA** (Los Angeles County highway traffic dataset: 207 loop detector sensors, collected at 5-minute sampling intervals; contains speed readings and sensor adjacency graph).
- **Target Horizons:** 
  - $t + 15$ min (Horizon 1: 3 time steps ahead)
  - $t + 30$ min (Horizon 2: 6 time steps ahead)
  - $t + 60$ min (Horizon 3: 12 time steps ahead)
- **Congestion Classification Thresholds:**
  - **Free Flow:** Speed $\ge 50\text{ mph}$ (Green)
  - **Moderate:** $35\text{ mph} \le \text{Speed} < 50\text{ mph}$ (Yellow)
  - **Heavy:** $20\text{ mph} \le \text{Speed} < 35\text{ mph}$ (Orange)
  - **Severe:** $\text{Speed} < 20\text{ mph}$ (Red)
- **Technology Stack:**
  - Core: Python 3.10+, NumPy, Pandas, Scikit-learn
  - Tree Modeling: XGBoost
  - Deep Learning: PyTorch (Spatio-Temporal GCN / GRU)
  - Backend API: FastAPI, Pydantic, Uvicorn
  - Interactive UI: Streamlit, Plotly, PyDeck / Folium
  - Testing: Pytest, HTTPX

---

## 2. System Architecture & Directory Structure

```
mlp@12/
├── data/
│   ├── raw/                  # METR-LA raw data (h5 / npz / csv) & sensor graph (adj_mx.pkl)
│   ├── processed/            # Scaled tensors, lag feature matrices, train/val/test splits
│   └── download_data.py      # Automated script to fetch METR-LA dataset or synthetic fallback
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py         # Data loading, cleaning, missing value imputation
│   │   └── preprocessor.py   # Z-score normalization, train/val/test sequential splitter
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py # Cyclical time features (hour, day), lag features, rolling windows
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py       # Historical Average (HA) & Last-known speed baselines
│   │   ├── xgboost_model.py  # Multi-output / Direct horizon XGBoost regressors
│   │   ├── pytorch_stgcn.py  # Spatial-Temporal Neural Network (GCN + Temporal Conv/GRU)
│   │   ├── classifier.py     # Speed-to-congestion mapping and class metrics
│   │   └── evaluate.py       # MAE, RMSE, MAPE, Accuracy, Macro F1, Confusion Matrix
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py            # FastAPI service definitions
│   │   └── schemas.py        # Pydantic request/response models
│   └── utils/
│       ├── __init__.py
│       └── config.py         # Global hyperparams, paths, and speed thresholds
├── app/
│   └── streamlit_app.py      # Interactive dashboard with map, forecasts, and live sensor monitors
├── tests/
│   ├── test_data.py          # Data ingestion and tensor shape checks
│   ├── test_features.py      # Lag creation & cyclical time features tests
│   ├── test_models.py        # Model forward passes, training steps, and mapping tests
│   └── test_api.py           # FastAPI endpoint smoke and integration tests
├── models_saved/             # Serialized model weights & scaler artifacts
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation and run instructions
```

---

## 3. Detailed Component Plan

### A. Data Pipeline
1. **Acquisition & Fallback:**
   - Script to download standard METR-LA HDF5/NPZ (`metr-la.h5` / `sensor_graph/adj_mx.pkl`).
   - Built-in deterministic synthetic generator mimicking METR-LA (207 sensors, 5-min intervals, diurnal morning/evening commute speed dips, spatial correlation) so the pipeline runs out-of-the-box in offline/restricted network environments.
2. **Quality & Cleaning:**
   - Detect and interpolate zero/missing readings (loop detector dropout).
   - Compute descriptive statistics per sensor (min, max, mean, variance).
3. **Temporal Train / Validation / Test Split:**
   - Respect strict time ordering (no future leakage):
     - Train: 70%
     - Validation: 10%
     - Test: 20%
   - Fit StandardScaler on Train split only; apply to Val/Test.

### B. Feature Engineering
1. **Temporal Features:**
   - Time of day: Normalized $[0, 1)$ or sinusoidal cyclical encodings $\sin(2\pi \cdot t/288), \cos(2\pi \cdot t/288)$.
   - Day of week: One-hot or cyclical $\sin(2\pi \cdot d/7), \cos(2\pi \cdot d/7)$.
   - Weekend indicator.
2. **Sequential Lag Features:**
   - Recent historical window (e.g., past 12 steps = 60 minutes): $[t-55, t-50, \dots, t-5, t]$.
   - Rolling statistics: Mean and standard deviation over 15-min and 60-min past windows.
   - Trend differential: $\Delta v = v_t - v_{t-15}$.
3. **Spatial Adjacency:**
   - Normalized graph Laplacian / adjacency matrix for sensor network graph integration.

### C. Modeling Strategy & Target Mapping
1. **Targets:**
   - Regression for 3 horizons: $y_{15} = v_{t+3}$, $y_{30} = v_{t+6}$, $y_{60} = v_{t+12}$.
2. **Candidate Models:**
   - **Baseline 1:** Historical Average (HA) conditioned on day-of-week and time-of-day.
   - **Baseline 2:** Last Observed Value (Persistence).
   - **Tabular / Gradient Boosting:** Multi-horizon XGBoost Regressors with optimized tree depth and learning rate.
   - **Deep Learning (PyTorch):** Spatial-Temporal Network:
     - Spatial Graph Convolution (GCN) layer capturing road network topology.
     - Temporal Gated Recurrent Unit (GRU) or 1D-CNN capturing dynamic speed propagation.
     - Multi-horizon prediction head outputting $(N_{\text{sensors}}, 3)$.
3. **Speed to Congestion Mapping:**
   - Continuous speed prediction $\hat{v}_h \to \text{Congestion Class}_h$ via calibrated thresholds:
     - Free Flow ($\ge 50\text{ mph}$)
     - Moderate ($[35, 50)\text{ mph}$)
     - Heavy ($[20, 35)\text{ mph}$)
     - Severe ($< 20\text{ mph}$)
4. **Evaluation Metrics:**
   - Speed Regression: MAE, RMSE, MAPE per horizon ($15\text{m}, 30\text{m}, 60\text{m}$).
   - Congestion Classification: Accuracy, Macro Precision, Recall, Macro F1-score, Confusion Matrix.

### D. FastAPI Backend
1. **Endpoints:**
   - `GET /health`: Service status and loaded model metadata.
   - `GET /sensors`: Sensor IDs, spatial coordinates (lat/lng), and status.
   - `POST /predict/sensor`: Given sensor ID and recent speed history, returns 15m, 30m, 60m speed forecasts + mapped congestion categories.
   - `POST /predict/network`: Batch prediction for all sensors at the current time step.
   - `GET /metrics`: Validation and test set benchmark metrics for each model.
2. **Validation:** Pydantic schemas enforcing history window length and value bounds.

### E. Interactive Streamlit Dashboard
1. **Network Map & Sensor Selection:**
   - Interactive spatial map displaying 207 Los Angeles traffic sensor nodes color-coded by current congestion level.
2. **Forecast Curves & Horizons:**
   - Sensor-level time series plot showing ground truth history and forecasted trajectory at 15m, 30m, 60m.
   - Side-by-side comparison of models (Baseline vs XGBoost vs PyTorch ST-GCN).
3. **Congestion Severity Indicator:**
   - Visual status badges with color coding (Green / Yellow / Orange / Red).
4. **Performance Benchmark Matrix:**
   - Interactive metric tables (MAE, RMSE, MAPE, F1) across horizons and models with confusion matrix heatmaps.
5. **Real-time / Simulation Slider:**
   - Time slider allowing user to scrub through historical test intervals to see predictions unfold dynamically.

### F. Testing & Validation Strategy
1. **Unit Tests:**
   - Data pipeline: Verification of window slicing $(X, Y)$ and temporal non-leakage.
   - Features: Verify cyclical encodings and lag alignment.
   - Classification mapper: Test edge cases and boundary speeds.
2. **Model Tests:**
   - Check input/output tensor shapes for PyTorch and XGBoost inference.
   - Ensure predictions are within physical speed limits (e.g., $0 \le v \le 100\text{ mph}$).
3. **API Tests:**
   - Test FastAPI endpoints using `TestClient` (200 responses, schema compliance, error handling for invalid sensor IDs).

---

## 4. Phase-by-Phase Task List

| Task ID | Phase | Description | Deliverable |
| :--- | :--- | :--- | :--- |
| **TASK-1** | Setup & Dependencies | Setup virtual environment, directory layout, `requirements.txt`, global config | Clean modular repo layout |
| **TASK-2** | Data Ingestion | METR-LA downloader, synthetic fallback generator, sensor graph loader | `loader.py`, `data/raw/` |
| **TASK-3** | Preprocessing & Features | Temporal splitting (70/10/20), lag generation ($t-12\dots t$), cyclical time encodings, normalization | `preprocessor.py`, `build_features.py` |
| **TASK-4** | Baselines & XGBoost | Implement HA, Persistence, and Multi-horizon XGBoost models; training & evaluation pipeline | `baseline.py`, `xgboost_model.py` |
| **TASK-5** | PyTorch ST-GCN | Implement Spatio-Temporal PyTorch model (GCN + GRU/Temporal Conv), training loop, checkpointing | `pytorch_stgcn.py` |
| **TASK-6** | Mapping & Evaluation | Speed-to-congestion classifier module, metric calculations (MAE/RMSE/F1), confusion matrix | `classifier.py`, `evaluate.py` |
| **TASK-7** | FastAPI Service | Build REST API endpoints (`/health`, `/sensors`, `/predict/sensor`, `/predict/network`), schemas, error handling | `app.py`, `schemas.py` |
| **TASK-8** | Streamlit UI | Build interactive dashboard with traffic map, sensor picker, multi-horizon forecast charts, and model comparison | `streamlit_app.py` |
| **TASK-9** | Verification & Testing | Implement pytest test suite (`test_data.py`, `test_models.py`, `test_api.py`), verify end-to-end execution | 100% passing tests |
