# Traffic Congestion Prediction System — Comprehensive Project Solution

## 1. Problem Statement and Why It Matters

### The Core Problem
Traffic congestion makes it difficult for people to know how long a journey will take. Unexpected slowdowns lead to wasted time and fuel, missed personal and professional appointments, supply chain bottlenecks, and critical delays for public transport or emergency response services. Drivers and transit operators often have to make high-stakes travel decisions without knowing how busy a given roadway corridor is likely to be 15, 30, or 60 minutes into the future.

### Why It Matters
- **Economic & Environmental Impact:** Urban traffic delays result in billions of dollars lost annually in wasted idling fuel, vehicle wear-and-tear, and increased greenhouse gas emissions.
- **Commuter Quality of Life:** Commute unpredictability creates chronic stress. Enabling travelers to compare departure times (e.g. leaving 15 minutes earlier or 30 minutes later) can save significant transit time.
- **Municipal & Public Transit Operations:** Transit agencies and municipal traffic managers can proactively adjust dispatch intervals, activate variable message signs (VMS), and re-time signals before congestion peaks.

> **Important Nature of This Tool:**  
> This project is a prediction and visualization system built upon the empirical **METR-LA** benchmark dataset. Unless connected to a live PeMS Caltrans telemetry API stream, the information is clearly presented as **sample and historical-data-based estimates**, not guaranteed live traffic feeds.

---

## 2. Project Objectives and Intended Users

### Main Goal
To enable a user to choose a highway road segment, date, and forecast horizon, and view its observed traffic conditions alongside estimated future congestion, travel time, and delay in an intuitive, accessible dashboard.

### Specific Objectives
1. **Multi-Horizon Forecasting:** Estimate traffic speeds at **15 minutes**, **30 minutes**, and **60 minutes** ahead.
2. **Standardized Severity Mapping:** Classify road status into four clear tiers:
   - **Low (Free Flow):** Speed $\ge 50\text{ mph}$ (Green)
   - **Moderate:** $35 \le \text{Speed} < 50\text{ mph}$ (Amber)
   - **High (Heavy):** $20 \le \text{Speed} < 35\text{ mph}$ (Orange)
   - **Severe Bottleneck:** $\text{Speed} < 20\text{ mph}$ (Red)
3. **Trip Duration & Delay Estimation:** Calculate realistic travel times and delays against standard 65 mph highway limits.
4. **Actionable Route & Departure Comparison:** Compare alternative highway routes (e.g., I-405 vs. I-10 vs. US-101 vs. I-110 vs. I-5) and departure time offsets (Now vs. +15m vs. +30m vs. +45m vs. +60m) to highlight optimal departure windows.
5. **Data Handling & CSV Upload:** Provide transparent data specifications and client-side/server-side CSV validation with plain-language feedback.

### Intended Users
- **Daily Commuters & Drivers:** People seeking to know the best time to leave and what delays to expect.
- **Fleet & Logistics Dispatchers:** Delivery and logistics planners optimizing departure times.
- **Transportation Engineers & City Planners:** Analysts studying recurring congestion patterns and spatial bottleneck spillover.
- **Students & Researchers:** Developers studying spatio-temporal time-series forecasting.

---

## 3. Proposed Solution and User Interaction Workflow

The web application is an end-to-end analytics dashboard with a light neutral aesthetic, dark readable typography, and restrained traffic color indicators.

### User Interaction Workflow
1. **Dashboard & Road Network View:**
   - Review top KPI summary cards: Overall Congestion, Average Network Speed, Average Corridor Travel Time, and Model Status.
   - Interact with the SVG road map of the 5 major Los Angeles freeway corridors (I-405, I-10, US-101, I-110, I-5) with 207 real sensor nodes.
   - Click any road segment or sensor pin to immediately load its 60-minute historical speed curve and multi-horizon (+15m, +30m, +60m) forecast curve.
2. **Explore & Predict Traffic:**
   - Filter by highway corridor or select any individual sensor.
   - Choose observation timestamps (e.g., Weekday AM Peak, Midday, PM Peak, or Latest Sample).
   - Select the forecasting engine: Historical Average Baseline, Persistence Baseline, XGBoost Regressor, or PyTorch ST-GCN.
   - Instantly see predicted speed, congestion severity tier, estimated travel time, and data sufficiency status.
3. **Compare Routes & Departure Times:**
   - **Compare Departure Times:** For a chosen road, inspect a table comparing departing Now vs. +15m, +30m, +45m, or +60m, with automated advice (e.g., *"Departing at 10:45 AM saves an estimated 3.2 minutes"*).
   - **Compare Alternative Corridors:** Compare distances, average speeds, travel times, and delays across all 5 highways to identify the fastest route.
4. **Data Handling & CSV Upload:**
   - Review underlying dataset parameters: 207 sensors, 5-minute intervals, 4,032 time steps (March 1–14, 2024).
   - Drag and drop custom sensor speed CSVs. The system validates the timestamp column, numeric sensor values, value ranges ($0\text{–}100\text{ mph}$), and nulls, reporting missing or invalid columns in plain English.
5. **Documentation & Problem Statement:**
   - Read the embedded problem statement, glossary of units (mph, km/h, min delay), and explicit notes on assumptions and limitations.
6. **User Preferences & Units:**
   - Toggle between Imperial ($\text{mph}$) and Metric ($\text{km/h}$) units across the entire application with one click.

---

## 4. Data Requirements, Preparation, and Assumptions

### Dataset Overview (METR-LA Benchmark)
- **Source:** Inductive loop detectors on the Los Angeles County highway network (Caltrans PeMS).
- **Spatial Topology:** 207 selected sensors across I-405, I-10, US-101, I-110, and I-5.
- **Spatial Adjacency:** Pairwise sensor distance matrix $\mathbf{A} \in \mathbb{R}^{207 \times 207}$, transformed via a thresholded Gaussian kernel:
  $$A_{ij} = \exp\left(-\frac{\text{dist}(v_i, v_j)^2}{\sigma^2}\right) \quad \text{for } \text{dist}(v_i, v_j) \le \kappa$$
- **Temporal Cadence:** Aggregated into continuous 5-minute readings ($12\text{ intervals/hour} = 288\text{ intervals/day}$).
- **Data Volume:** 4,032 consecutive time intervals spanning March 1 to March 14, 2024.

### Data Preparation Pipeline
1. **Cleaning & Missing Value Imputation:** Missing readings are forward-filled for short gaps ($\le 15\text{ min}$) and imputed via empirical historical profile for larger gaps.
2. **Sequential Temporal Split:** To prevent data leakage in time-series forecasting, data is split chronologically:
   - **Training Set (70%):** First ~10 days (2,822 steps).
   - **Validation Set (10%):** Next 1.4 days (403 steps).
   - **Testing Set (20%):** Final 2.8 days (807 steps).
3. **Feature Engineering:**
   - 12 lag steps: $v_{t-11}, v_{t-10}, \dots, v_t$ (past 60 minutes).
   - Rolling statistics: 15-min and 60-min moving averages and standard deviations.
   - Cyclical temporal encodings: $\sin(2\pi \cdot \text{hour}/24)$, $\cos(2\pi \cdot \text{hour}/24)$, $\sin(2\pi \cdot \text{day}/7)$, $\cos(2\pi \cdot \text{day}/7)$.
   - Upstream/downstream neighbor speeds from the adjacency matrix.

### Key Assumptions
- **Regular Commute Patterns:** Baseline models assume recurring daily traffic rhythms (morning and evening rush hours repeat cyclically).
- **Normal Operating Conditions:** Assumes standard weather and the absence of catastrophic multi-lane blockages. Non-recurring incidents will cause real-world traffic to deviate from model estimates.
- **Mainline Representation:** Sensors measure mainline highway speeds; on-ramp metering queues and local arterial surface streets are not captured.

---

## 5. Prediction Methods (Plain-Language Explanation)

The application separates prediction logic from the UI and supports four complementary forecasting approaches:

### 1. Historical Average (HA) Baseline (Explainable Core Baseline)
- **How it works:** Divides the week into $7 \times 288 = 2,016$ discrete 5-minute slots. For any future date and time, the model looks up the historical average speed recorded by that exact sensor on the same day of the week at that specific time of day.
- **Why it matters:** It is simple, transparent, fully explainable, and requires zero black-box parameters. It captures the regular commute wave (e.g. 8:00 AM rush on I-10 East) with high reliability.

### 2. Persistence Baseline (Naive Benchmark)
- **How it works:** Assumes conditions remain static into the future:
  $$\hat{v}_{t+15} = \hat{v}_{t+30} = \hat{v}_{t+60} = v_t$$
- **Purpose:** Serves as the zero-intelligence benchmark against which all predictive algorithms must prove their value.

### 3. XGBoost Multi-Horizon Regressor
- **How it works:** Trains separate gradient boosted decision trees for each horizon (15 min, 30 min, 60 min). It uses historical speed lags, rolling momentum, neighbor sensor conditions, and cyclical time features to detect when traffic is accelerating or deteriorating.

### 4. PyTorch Spatio-Temporal Graph Neural Network (ST-GCN)
- **How it works:** Combines spectral graph convolutions over the road network topology matrix with temporal Gated Recurrent Units (GRU) to model how traffic waves physically propagate along highway corridors.

---

## 6. System Architecture and Components

```
                          ┌────────────────────────────────────────────────────────┐
                          │               Browser (Frontend Client)               │
                          │   app/static/index.html · app.js · styles.css         │
                          └──────────────────────────▲─────────────────────────────┘
                                                     │ REST API / JSON
                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             FastAPI Application (src/api/app.py)                         │
├──────────────────────────────┬────────────────────────────┬──────────────────────────────┤
│      Telemetry & Overview    │    Prediction & Trajectory │    Comparisons & CSV Upload  │
│  • GET /api/overview         │  • GET /api/sensor/traj    │  • POST /api/compare/dep     │
│  • GET /api/corridors        │  • POST /predict/sensor    │  • POST /api/compare/roads   │
│  • GET /sensors              │  • POST /predict/network   │  • POST /api/upload-csv      │
└──────────────────────────────┴─────────────▲──────────────┴──────────────────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
         ┌─────────────▼─────────────┐               ┌─────────────▼─────────────┐
         │   Data & Feature Engine   │               │     Model Zoo Engines     │
         │  • src/data/loader.py     │               │  • HistoricalAverage (HA) │
         │  • src/data/preprocessor  │               │  • Persistence Baseline   │
         │  • src/features/build     │               │  • XGBoost Regressors     │
         └─────────────▲─────────────┘               │  • PyTorch ST-GCN         │
                       │                             └─────────────▲─────────────┘
                       │                                           │
         ┌─────────────▼───────────────────────────────────────────▼─────────────┐
         │                    Data & Pretrained Model Artifacts                  │
         │  • data/raw/metr_la_speeds.csv      • models_saved/xgboost_model      │
         │  • data/raw/sensor_metadata.csv     • models_saved/pytorch_stgcn.pt   │
         │  • data/raw/adj_mx.pkl              • models_saved/eval_metrics.json  │
         └───────────────────────────────────────────────────────────────────────┘
```

---

## 7. How Prediction Quality Is Evaluated

Prediction quality is evaluated using both regression error metrics (continuous speed in mph) and classification metrics (severity tier accuracy):

### Regression Metrics
- **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$ (interpretable error in mph).
- **Root Mean Squared Error (RMSE):** Penalizes large deviations heavily.
- **Mean Absolute Percentage Error (MAPE):** Relative error percentage.

### Classification Metrics
- **Classification Accuracy:** Proportion of time the predicted congestion tier (Low, Moderate, High, Severe) matches reality.
- **Macro F1 Score:** Balanced harmonic mean of precision and recall across all four congestion categories, ensuring rare severe bottlenecks are detected accurately.

### Benchmark Evaluation (Official METR-LA Test Split)
| Model | Overall MAE | Overall RMSE | Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Persistence (Naive)** | 4.49 mph | 6.30 mph | 81.6% | 46.5% |
| **Historical Average (HA)** | 2.38 mph | 3.07 mph | 94.2% | 65.2% |
| **PyTorch ST-GCN** | 2.35 mph | 3.22 mph | 92.3% | 61.5% |
| **XGBoost Regressor** | **1.80 mph** | **2.31 mph** | **95.1%** | **66.3%** |

---

## 8. Limitations, Risks, and Future Improvements

### Limitations & Risks
1. **Unpredictable Road Incidents:** Traffic crashes, vehicle stalls, and emergency closures cannot be anticipated purely from historical pattern profiles without live incident feeds.
2. **Weather Sensitivities:** Heavy precipitation alters free-flow speeds; models trained on dry conditions may underestimate delays during adverse weather.
3. **Loop Detector Dropout:** Physical induction loops occasionally experience hardware faults, requiring robust imputation fallbacks.

### Possible Future Improvements
1. **Live Caltrans PeMS API Connector:** Integrate live PeMS REST feeds to enable genuine real-time traffic monitoring alongside historical forecasting.
2. **Weather API Fusion:** Incorporate NOAA precipitation and visibility telemetry as dynamic covariates.
3. **Turn-by-Turn GPS Navigation Integration:** Connect routing engines (e.g. OpenStreetMap / OSRM) for door-to-door journey time optimization.

---

## 9. How to Install and Run Locally

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- pip

### 1. Install Dependencies
```bash
cd d:\projects\mlp@12
pip install -r requirements.txt
```

### 2. Launch the Application Server
Run the FastAPI application with Uvicorn:
```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access in Browser
- **Interactive Web Application:** [http://localhost:8000](http://localhost:8000)
- **Interactive API Documentation (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health Check:** [http://localhost:8000/health](http://localhost:8000/health)



---

## 10. Login & Account Flow (Local Demonstration Mode)

### Purpose & Structure
To provide a secure, realistic commuter and reviewer access flow while operating strictly on localhost:
- **Hero Column (Left):** Explains project motivation, key forecasting capabilities (15m, 30m, 60m), standardized congestion tiers, and dataset background.
- **Sign-In Form (Right):** Includes email validation, password field with **Show/Hide password toggle** (`eye` icon), "Remember me" option, and "Forgot password?" modal.
- **One-Click Instant Demo Login:** A dedicated button (`⚡ One-Click Demo Sign In`) that immediately logs the user in with demo credentials (`demo@trafficpredictor.org` / `demo1234`).
- **Create Account Toggle:** Allows reviewers to switch in-place to registration mode (Full Name, Email, Password, Confirm Password).
- **Session Management:** Stores session status in `localStorage`, displays user initials and email in the top bar chip, and provides a **Sign Out** button that returns to the sign-in screen.
- **Security Notice:** Explicitly discloses that no plain-text passwords or external authentication credentials are transmitted.

---

## 11. Reviewer & Commuter Questions Answered on First Screen

1. **"What does this system do?"**  
   Answered immediately in the top purpose callout:  
   *“This dashboard uses historical traffic speed telemetry from 207 inductive loop sensors on Los Angeles freeways to predict future road conditions at 15, 30, and 60 minutes ahead so you can make informed travel decisions.”*
2. **"Which road and time am I viewing?"**  
   Answered in the persistent compact top bar pill:  
   *`Viewing: I-10 Santa Monica Fwy · Sensor 7100075 · Forecast +15 min`* and synchronized across the map pin, side prediction panel, and chart title.
3. **"How certain or limited is this estimate?"**  
   Answered in the top disclaimer badge, KPI status card, and prediction telemetry:  
   *`Historical Benchmark Estimates · Not Live Radar · 94.6% Avg Confidence · 12 consecutive historical steps observed`*.
