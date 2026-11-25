# STRUCTURA – ML-Based Traffic Intelligence Platform

A full-stack **Machine Learning Traffic Intelligence & Forecasting Platform** combining a premium obsidian dark-mode SaaS dashboard, a **FastAPI** backend, **SQLite** persistent user accounts & prediction logs, and **8 trained ML/DL models** (Multi-Layer Perceptron, Random Forest, XGBoost, and PyTorch ST-GCN).

---

## Key Features

### Operational Workspace
- **Overview Dashboard (`/dashboard`):** Live KPI cards (`Network Speed`, `Active Slowdowns`, `Avg. Delay`, `Best Departure Window`), interactive SVG Corridor Conditions map, Speed Forecast wave curve, and Corridors to Watch — dynamically recalculates whenever you upload a custom CSV dataset.
- **Forecast Explorer:** Multi-horizon speed & congestion forecasts (`+15m`, `+30m`, `+60m`) across 207 Los Angeles freeway loop sensors (METR-LA).
- **Route Comparison:** Departure time slot evaluation (`Leave Now` to `+60m`) with automated best-window recommendation.
- **Data Sources & Reports:** METR-LA telemetry specs, custom sensor CSV validation, sample dataset downloads, and 4-tier severity standards.

### 7 Machine Learning Modules
1. **Traffic Congestion Prediction (MLP Classifier):** Real-time 4-class prediction (`LOW`, `MODERATE`, `HIGH`, `SEVERE`) with softmax confidence & class probability bars.
2. **Vehicle Speed Prediction (MLP Regressor):** Real-time continuous speed prediction in `km/h` and `mph` with a 60-minute projected trajectory.
3. **Peak-Hour Traffic Forecasting:** Multi-horizon (`Now`, `+15m`, `+30m`, `+60m`) volume & speed evolution with peak-window intensity detection.
4. **Dataset Explorer & Preprocessing:** CSV upload, missing-value & duplicate audit, imputation (`Mean`/`Median`/`Drop`), normalization, Train/Val/Test split controls, distribution histograms, and correlation heatmap.
5. **MLP Model Evaluation:** Live Test Accuracy, Precision, Recall, F1, $R^2$, MAE, RMSE, MSE, 4×4 Confusion Matrix, and Training Loss convergence curve.
6. **Model Comparison:** Fair side-by-side benchmarking of **MLP vs. Random Forest vs. XGBoost** on identical dataset splits.
7. **Feature Importance & Prediction History:** Permutation feature importance analysis + persistent user-scoped SQLite prediction history with search, filter, and CSV export.

### Persistent User Accounts & Auth (`/login`)
- PBKDF2-HMAC-SHA256 salted password authentication with Bearer session tokens stored in SQLite (`data/structura.db`).
- User profiles, form inputs, unit preferences, uploaded datasets, and prediction histories persist across logout and re-login.

---

## Model Benchmark Results

### ML Platform Models (Balanced 4-Class Split — `models_saved/ml_platform_benchmarks.json`)

| Model | Test Accuracy | F1-Score | Regression $R^2$ | MAE (km/h) | Inference Latency |
|---|---:|---:|---:|---:|---:|
| **MLP `(64, 32)`** | **99.52%** | 0.9952 | **0.9854** | 2.66 km/h | **0.003 ms** *(Fastest)* |
| **Random Forest** | **100.00%** | 1.0000 | **0.9902** | 2.17 km/h | 0.008 ms |
| **XGBoost** | **99.90%** | 0.9990 | **0.9907** | **2.12 km/h** | 0.006 ms |

### METR-LA Multi-Horizon Models (207 LA Sensors — `models_saved/evaluation_metrics.json`)

| Model | Overall Accuracy | Overall MAE | Overall RMSE | Overall MAPE |
|---|---:|---:|---:|---:|
| **XGBoost (METR-LA)** | **95.08%** | **1.802 mph** | **2.315 mph** | **3.47%** |
| **Historical Average** | 94.16% | 2.377 mph | 3.073 mph | 4.52% |
| **PyTorch ST-GCN** | 92.28% | 2.355 mph | 3.218 mph | 4.70% |
| **Persistence Baseline** | 81.60% | 4.488 mph | 6.301 mph | 9.33% |

---

## Quickstart (Run Locally)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Retrain ML Platform Models
```bash
python src/models/train_ml_platform.py
```

### 3. Start the FastAPI Server
```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

### 4. Open in Browser
- **Dashboard:** `http://localhost:8000/dashboard`
- **Login / Create Account:** `http://localhost:8000/login`
- **Interactive API Docs (Swagger):** `http://localhost:8000/docs`

**Default Demo Account:** `alex.taylor@structura.io` / `structura2026` (or create your own account via **Create an account**).

---

## Sample Test Datasets (`data/test_samples/`)
- `ml_dataset_clean.csv` — 200 balanced rows (50 per congestion tier) for **Dataset Explorer**
- `ml_dataset_with_missing_and_dupes.csv` — 120 rows with 13 nulls & 5 duplicates for testing preprocessing
- `sensor_speed_upload.csv` — 60 timestamped intervals × 5 LA sensors for **Data Sources**

---

## Technology Stack
- **Frontend:** HTML5, Custom CSS3 Obsidian Design System, Vanilla ES6 JavaScript, Inline SVG Charts
- **Backend:** Python 3.12, FastAPI, Uvicorn, Pydantic v2, SQLite3
- **Machine Learning:** Scikit-learn, XGBoost, PyTorch, Pandas, NumPy, Joblib
