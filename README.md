# SafeRoute AI
### *An Intelligent Traffic Congestion and Accident Risk Prediction Platform*

![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=flat&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas&logoColor=white)
![Leaflet](https://img.shields.io/badge/Leaflet-1.9+-199900?style=flat&logo=leaflet&logoColor=white)

---

## 📌 Project Overview
**SafeRoute AI** is an academic AI/ML decision-support system designed to forecast urban traffic congestion, evaluate accident severity risk probabilities, and discover geographical accident hotspots. 

Framed strictly as a **Decision-Support System**, the platform provides predictive metrics, analytical visualizations, and preventive decision recommendations to assist traffic safety authorities and urban commuters.

> **Decision-Support Disclaimer**: SafeRoute AI provides probabilistic risk scoring and congestion forecasting for decision support. It does not replace live traffic control directives or assert direct accident prevention.

---

## 🎯 Key Objectives
1. **Traffic Volume Regression**: Predict continuous hourly vehicle counts using environmental and temporal features.
2. **Accident Risk Classification**: Classify binary accident severity risk levels ($1 = \text{High Risk}$, $0 = \text{Low Risk}$) from road, weather, and time attributes.
3. **Spatial Hotspot Clustering**: Compare K-Means and DBSCAN algorithms to detect spatial density clusters of accidents.
4. **Model Comparison**: Empirical benchmarking across 5 regression models and 5 classification algorithms.
5. **Flask Deployment & API**: Expose trained ML pipelines through a REST API and interactive web interface with Leaflet.js maps and Plotly charts.
6. **SQLite History Logging**: Persist real-time prediction logs and spatial cluster centroids into an SQLite database.

---

## 📊 Datasets & Sources

| Dataset Name | Source | Target Variable | Primary ML Task |
| :--- | :--- | :--- | :--- |
| **UCI Metro Interstate Traffic Volume** | [UCI Repository](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) | `traffic_volume` | Continuous Regression |
| **US Accidents (2016-2023)** | [Kaggle US Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) | `Severity` / `high_risk_target` | Classification & Spatial Clustering |

*Note: The datasets are treated as complementary pipelines integrated at the application level rather than merged row-by-row.*

---

## 🛠️ Technology Stack
- **Backend Framework**: Python 3.13, Flask 3.0
- **Machine Learning & Pipeline**: Scikit-Learn, NumPy, Pandas, Joblib
- **Database**: SQLite3
- **Visualization**: Plotly.js, Folium / Leaflet.js, Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Automated Testing**: Pytest

---

## 📁 Directory Structure
```
SafeRoute-AI/
│
├── app.py                      # Flask Application Server & REST Endpoints
├── config.py                   # Central Project Configuration
├── requirements.txt            # Dependency Requirements
├── README.md                   # Project Documentation
├── .gitignore                  # Git Ignore Definitions
│
├── data/
│   ├── README.md               # Data Directory Overview
│   ├── raw/                    # Metro_Interstate_Traffic_Volume.csv & US_Accidents_Sample.csv
│   ├── processed/              # Feature-engineered CSVs
│   └── external/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_traffic_regression.ipynb
│   ├── 04_accident_classification.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_clustering_hotspots.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load_data.py        # CSV Data Loaders
│   │   ├── clean_data.py       # Cleaning & Imputation Pipeline
│   │   ├── feature_engineering.py # Temporal & Target Transformers
│   │   └── download_data.py    # Automatic Dataset Downloader
│   │
│   ├── models/
│   │   ├── traffic_model.py    # Regression Training & Evaluation Pipeline
│   │   ├── accident_model.py   # Classification Training Pipeline
│   │   ├── clustering_model.py # K-Means vs DBSCAN Hotspot Clustering
│   │   └── model_comparison.py # Unified Benchmark Aggregator
│   │
│   ├── evaluation/
│   │   ├── regression_metrics.py    # MAE, MSE, RMSE, R² Metrics
│   │   └── classification_metrics.py # Accuracy, Precision, Recall, F1, ROC-AUC
│   │
│   ├── visualization/
│   │   ├── traffic_charts.py   # Plotly Traffic Charts
│   │   ├── accident_charts.py  # Plotly Accident Severity Charts
│   │   └── hotspot_map.py      # Leaflet Map Generator
│   │
│   └── utils/
│       └── helpers.py          # Decision Support Recommendation Logic
│
├── models/                     # Saved Joblib Pipelines
│   ├── traffic_model.pkl
│   ├── accident_model.pkl
│   ├── scaler.pkl
│   └── clustering_model.pkl
│
├── templates/                  # Jinja2 HTML Templates
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   ├── prediction.html
│   ├── hotspots.html
│   ├── analytics.html
│   ├── model_comparison.html
│   └── about.html
│
├── static/
│   ├── css/                    # Custom Stylesheets
│   └── js/                     # Client-side Interactive Scripts
│
├── database/
│   ├── schema.sql              # SQLite DDL Schema
│   ├── init_db.py              # SQLite Initialization Script
│   └── saferoute.db            # SQLite Database File
│
└── tests/
    ├── test_models.py          # Pipeline Unit Tests
    └── test_routes.py          # Flask Web & API Route Tests
```

---

## ⚡ Setup & Installation Instructions

### 1. Clone & Set Up Virtual Environment
```bash
# Navigate to project directory
cd SafeRoute-AI

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate virtual environment (Linux/macOS)
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Fetch Datasets & Train Models
Run the automated data downloader and model training scripts:
```bash
# 1. Download/prepare datasets
python src/data/download_data.py

# 2. Train Traffic Volume Regression Models (Saves models/traffic_model.pkl)
python src/models/traffic_model.py

# 3. Train Accident Risk Classifiers (Saves models/accident_model.pkl)
python src/models/accident_model.py

# 4. Run Hotspot Clustering (Saves models/clustering_model.pkl)
python src/models/clustering_model.py

# 5. Initialize SQLite Database
python database/init_db.py
```

---

## 🚀 Running the Web Application & Server
To start the Flask deployment server:
```bash
python app.py
```
Open your web browser and navigate to: **`http://localhost:5000`**

### Web Routes:
- `GET /` : Landing page with project overview and features
- `GET /dashboard` : Executive dashboard with KPI cards and recent predictions
- `GET /predict` : Interactive prediction form for custom inputs
- `GET /hotspots` : Interactive Leaflet.js spatial map with cluster markers
- `GET /analytics` : Multi-chart EDA & risk analytics dashboard
- `GET /models` : Unified benchmark tables for all algorithms
- `GET /about` : Academic write-up, mathematical background, and methodology

---

## 📡 REST API Documentation

### POST `/api/predict`
Calculates traffic volume and accident risk probability for custom input features.

#### Request Body (`JSON`):
```json
{
  "location": "Interstate 94 Westbound Corridor",
  "date_time": "2026-10-01 08:30:00",
  "temp": 18.5,
  "rain_1h": 0.0,
  "snow_1h": 0.0,
  "clouds_all": 25,
  "weather_main": "Clear",
  "holiday": "None",
  "visibility": 10.0,
  "humidity": 45.0,
  "wind_speed": 5.0
}
```

#### Response Body (`JSON`):
```json
{
  "status": "success",
  "timestamp": "2026-10-01 08:30:00",
  "location": "Interstate 94 Westbound Corridor",
  "traffic_prediction": 4350.2,
  "traffic_level": "Heavy Traffic Congestion",
  "accident_probability": 0.6845,
  "risk_level": "High Risk",
  "recommendations": [
    "Traffic Management: Activate variable message signs (VMS) to suggest alternative arterial routes.",
    "Commuter Advice: Expect delays of 15-30 minutes. Consider shifting departure times by 45 minutes.",
    "Safety Dispatch: Increase highway patrol monitoring in high-density corridors."
  ],
  "disclaimer": "DISCLAIMER: SafeRoute AI is an intelligent decision-support system..."
}
```

---

## 🧪 Running Automated Tests
Execute the Pytest suite to verify dataset pipelines, ML inference, and Flask routes:
```bash
pytest tests/ -v
```

---

## 🤖 Algorithms & Evaluation Summary

### 1. Traffic Volume Regression Models
- **Algorithms**: Linear Regression, Polynomial Regression (Degree 2), Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor.
- **Evaluation Metrics**: MAE, MSE, RMSE, R² score.
- **Top Model**: Gradient Boosting Regressor ($R^2 = 0.9528$, $\text{RMSE} = 429.9$).

### 2. Accident Risk Severity Classification
- **Algorithms**: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Support Vector Machine (SGD Classifier), K-Nearest Neighbors (KNN).
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix.
- **Top Model**: Random Forest Classifier ($\text{Accuracy} = 0.894$, $\text{F1} = 0.862$, $\text{ROC-AUC} = 0.941$).

### 3. Hotspot Spatial Clustering
- **Algorithms**: K-Means ($k=8$) vs DBSCAN ($\epsilon=0.3, \text{min\_samples}=15$).
- **Key Finding**: K-Means provided distinct urban centroid partitions, while DBSCAN excelled at isolating spatial noise points (outliers) along corridor segments.

---

## ⚖️ Limitations & Future Scope
- **Current Limitations**: Geographical accident data relies on static spatial snapshots; real-time streaming sensor telemetry is simulated.
- **Future Scope**: Integration of deep learning time-series forecasting (LSTM / Transformer architectures), live OpenStreetMap routing API integration, and edge device telemetry deployment.

---

## 📝 License
This project is developed for academic AI/ML laboratory demonstration purposes.
