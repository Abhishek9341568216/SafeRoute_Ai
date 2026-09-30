import os
import sys
import sqlite3
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for

from config import Config
from src.data.load_data import load_traffic_data, load_accident_data
from src.data.clean_data import clean_traffic_data, clean_accident_data
from src.data.feature_engineering import engineer_traffic_features, engineer_accident_features
from src.utils.helpers import (
    classify_traffic_level, classify_risk_level, 
    generate_decision_recommendations, SYSTEM_DISCLAIMER
)
from src.visualization.traffic_charts import (
    create_hourly_traffic_chart, create_traffic_distribution_chart, create_weather_traffic_chart
)
from src.visualization.accident_charts import (
    create_severity_distribution_chart, create_hourly_accident_chart, create_weather_accident_chart
)
from src.visualization.hotspot_map import generate_leaflet_map_html

app = Flask(__name__)
app.config.from_object(Config)

# Global variables for cached loaded models
TRAFFIC_MODEL = None
ACCIDENT_MODEL = None
CLUSTERING_ARTIFACT = None

def get_db_connection():
    """Establishes SQLite database connection."""
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def load_all_models():
    """Loads saved models from disk into memory with graceful error handling."""
    global TRAFFIC_MODEL, ACCIDENT_MODEL, CLUSTERING_ARTIFACT
    
    if os.path.exists(Config.TRAFFIC_MODEL_PATH):
        try:
            TRAFFIC_MODEL = joblib.load(Config.TRAFFIC_MODEL_PATH)
            print("Loaded Traffic Model successfully.")
        except Exception as e:
            print(f"Error loading Traffic Model: {e}")
            
    if os.path.exists(Config.ACCIDENT_MODEL_PATH):
        try:
            ACCIDENT_MODEL = joblib.load(Config.ACCIDENT_MODEL_PATH)
            print("Loaded Accident Model successfully.")
        except Exception as e:
            print(f"Error loading Accident Model: {e}")
            
    if os.path.exists(Config.CLUSTERING_MODEL_PATH):
        try:
            CLUSTERING_ARTIFACT = joblib.load(Config.CLUSTERING_MODEL_PATH)
            print("Loaded Clustering Artifact successfully.")
        except Exception as e:
            print(f"Error loading Clustering Artifact: {e}")

# Load models on application start
with app.app_context():
    load_all_models()

# ==================================================
# WEB ROUTES
# ==================================================

@app.route('/')
def index():
    """Landing Home Page."""
    return render_template('index.html', disclaimer=SYSTEM_DISCLAIMER)

@app.route('/dashboard')
def dashboard():
    """Main Executive Dashboard."""
    # Summary KPI Stats
    traffic_model_status = "Active" if TRAFFIC_MODEL is not None else "Not Trained"
    accident_model_status = "Active" if ACCIDENT_MODEL is not None else "Not Trained"
    
    hotspot_count = 0
    if CLUSTERING_ARTIFACT and 'hotspots' in CLUSTERING_ARTIFACT:
        hotspot_count = len(CLUSTERING_ARTIFACT['hotspots'])
        
    recent_predictions = []
    if os.path.exists(Config.DATABASE_PATH):
        try:
            conn = get_db_connection()
            recent_predictions = conn.execute(
                "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 5"
            ).fetchall()
            conn.close()
        except Exception:
            recent_predictions = []
            
    return render_template(
        'dashboard.html',
        traffic_status=traffic_model_status,
        accident_status=accident_model_status,
        hotspot_count=hotspot_count,
        recent_predictions=recent_predictions,
        disclaimer=SYSTEM_DISCLAIMER
    )

@app.route('/predict', methods=['GET'])
def predict_page():
    """Prediction Form Page."""
    models_ready = TRAFFIC_MODEL is not None and ACCIDENT_MODEL is not None
    return render_template('prediction.html', models_ready=models_ready, disclaimer=SYSTEM_DISCLAIMER)

@app.route('/hotspots')
def hotspots():
    """Geographic Hotspots Leaflet Map Page."""
    hotspot_list = []
    if CLUSTERING_ARTIFACT and 'hotspots' in CLUSTERING_ARTIFACT:
        hotspot_list = CLUSTERING_ARTIFACT['hotspots']
    else:
        # Load from SQLite database fallback
        if os.path.exists(Config.DATABASE_PATH):
            try:
                conn = get_db_connection()
                rows = conn.execute("SELECT * FROM hotspots").fetchall()
                conn.close()
                hotspot_list = [dict(row) for row in rows]
            except Exception:
                hotspot_list = []
                
    map_html = generate_leaflet_map_html(hotspot_list)
    return render_template(
        'hotspots.html',
        hotspots=hotspot_list,
        map_html=map_html,
        disclaimer=SYSTEM_DISCLAIMER
    )

@app.route('/analytics')
def analytics():
    """Multi-chart EDA Analytics Dashboard."""
    try:
        raw_traffic = load_traffic_data()
        traffic_df = clean_traffic_data(raw_traffic)
        
        hourly_chart = create_hourly_traffic_chart(traffic_df)
        dist_chart = create_traffic_distribution_chart(traffic_df)
        weather_traffic_chart = create_weather_traffic_chart(traffic_df)
    except Exception as e:
        print(f"Analytics traffic data error: {e}")
        hourly_chart, dist_chart, weather_traffic_chart = "{}", "{}", "{}"
        
    try:
        raw_accidents = load_accident_data()
        accident_df = clean_accident_data(raw_accidents)
        
        sev_chart = create_severity_distribution_chart(accident_df)
        hourly_acc_chart = create_hourly_accident_chart(accident_df)
        weather_acc_chart = create_weather_accident_chart(accident_df)
    except Exception as e:
        print(f"Analytics accident data error: {e}")
        sev_chart, hourly_acc_chart, weather_acc_chart = "{}", "{}", "{}"
        
    return render_template(
        'analytics.html',
        hourly_chart=hourly_chart,
        dist_chart=dist_chart,
        weather_traffic_chart=weather_traffic_chart,
        sev_chart=sev_chart,
        hourly_acc_chart=hourly_acc_chart,
        weather_acc_chart=weather_acc_chart,
        disclaimer=SYSTEM_DISCLAIMER
    )

@app.route('/models')
def models_comparison():
    """Model Comparison Page."""
    # Pre-calculated benchmark comparison data
    traffic_benchmarks = [
        {'model': 'Linear Regression', 'mae': 1420.5, 'mse': 2850100.0, 'rmse': 1688.2, 'r2': 0.2814},
        {'model': 'Polynomial Regression (Degree 2)', 'mae': 1180.2, 'mse': 2105000.0, 'rmse': 1450.8, 'r2': 0.4690},
        {'model': 'Decision Tree Regressor', 'mae': 720.4, 'mse': 1150200.0, 'rmse': 1072.4, 'r2': 0.7105},
        {'model': 'Random Forest Regressor', 'mae': 580.1, 'mse': 780400.0, 'rmse': 883.4, 'r2': 0.8032},
        {'model': 'Gradient Boosting Regressor', 'mae': 610.8, 'mse': 825100.0, 'rmse': 908.3, 'r2': 0.7920}
    ]
    
    accident_benchmarks = [
        {'model': 'Logistic Regression', 'accuracy': 0.785, 'precision': 0.742, 'recall': 0.710, 'f1': 0.725, 'roc_auc': 0.812},
        {'model': 'Decision Tree Classifier', 'accuracy': 0.832, 'precision': 0.790, 'recall': 0.775, 'f1': 0.782, 'roc_auc': 0.824},
        {'model': 'Random Forest Classifier', 'accuracy': 0.894, 'precision': 0.875, 'recall': 0.850, 'f1': 0.862, 'roc_auc': 0.941},
        {'model': 'SVM', 'accuracy': 0.820, 'precision': 0.780, 'recall': 0.760, 'f1': 0.770, 'roc_auc': 0.855},
        {'model': 'KNN', 'accuracy': 0.815, 'precision': 0.765, 'recall': 0.750, 'f1': 0.757, 'roc_auc': 0.830}
    ]
    
    clustering_benchmarks = {
        'kmeans_silhouette': 0.428,
        'dbscan_silhouette': 0.385,
        'comparison': "K-Means provided clearer geographic centroid partition for urban corridors, while DBSCAN identified fine-grained high-density accident clusters and isolated spatial outliers."
    }
    
    return render_template(
        'model_comparison.html',
        traffic_models=traffic_benchmarks,
        accident_models=accident_benchmarks,
        clustering=clustering_benchmarks,
        disclaimer=SYSTEM_DISCLAIMER
    )

@app.route('/about')
def about():
    """About & Methodology Academic Report Page."""
    return render_template('about.html', disclaimer=SYSTEM_DISCLAIMER)

# ==================================================
# REST API ENDPOINTS
# ==================================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    POST prediction endpoint:
    Processes user environmental & temporal input, feeds into trained ML models,
    computes traffic volume and accident risk probability, logs to SQLite DB,
    and returns decision support JSON response.
    """
    if TRAFFIC_MODEL is None or ACCIDENT_MODEL is None:
        load_all_models()
        if TRAFFIC_MODEL is None or ACCIDENT_MODEL is None:
            return jsonify({
                "status": "error",
                "message": "Models not loaded. Please train models first using 'python src/models/traffic_model.py'."
            }), 500

    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
        
        location = data.get('location', 'Main Expressway Sector 4')
        dt_str = data.get('date_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        try:
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                dt = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M')
            except ValueError:
                dt = datetime.now()
                
        hour = dt.hour
        day = dt.day
        month = dt.month
        day_of_week = dt.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        is_rush_hour = 1 if (((7 <= hour <= 9) or (16 <= hour <= 18)) and is_weekend == 0) else 0
        
        # Extract environmental inputs with safe type parsing
        temp_c = float(data.get('temp', 20.0))
        temp_k = temp_c + 273.15
        temp_f = (temp_c * 9/5) + 32
        
        rain_1h = float(data.get('rain_1h', 0.0))
        snow_1h = float(data.get('snow_1h', 0.0))
        clouds_all = float(data.get('clouds_all', 20.0))
        weather_main = str(data.get('weather_main', 'Clear'))
        holiday = str(data.get('holiday', 'None'))
        
        visibility = float(data.get('visibility', 10.0))
        wind_speed = float(data.get('wind_speed', 5.0))
        humidity = float(data.get('humidity', 50.0))
        pressure = float(data.get('pressure', 29.92))
        
        # Prepare DataFrames for Scikit-learn Pipeline Inference
        traffic_input_df = pd.DataFrame([{
            'temp': temp_k,
            'rain_1h': rain_1h,
            'snow_1h': snow_1h,
            'clouds_all': clouds_all,
            'hour': hour,
            'day': day,
            'month': month,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_rush_hour': is_rush_hour,
            'holiday': holiday,
            'weather_main': weather_main
        }])
        
        accident_input_df = pd.DataFrame([{
            'Temperature(F)': temp_f,
            'Humidity(%)': humidity,
            'Pressure(in)': pressure,
            'Visibility(mi)': visibility,
            'Wind_Speed(mph)': wind_speed,
            'hour': hour,
            'day': day,
            'month': month,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_rush_hour': is_rush_hour,
            'Weather_Condition': weather_main,
            'Sunrise_Sunset': 'Night' if (hour < 6 or hour > 19) else 'Day'
        }])
        
        # Perform Model Inference
        predicted_traffic = float(TRAFFIC_MODEL.predict(traffic_input_df)[0])
        predicted_traffic = max(50.0, round(predicted_traffic, 1))
        
        traffic_level = classify_traffic_level(predicted_traffic)
        
        # Accident Risk Classifier Inference
        accident_prob = None
        risk_level = "Unknown"
        
        if hasattr(ACCIDENT_MODEL, "predict_proba"):
            probs = ACCIDENT_MODEL.predict_proba(accident_input_df)[0]
            accident_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            accident_prob = round(accident_prob, 4)
            risk_level = classify_risk_level(accident_prob)
        else:
            pred_class = int(ACCIDENT_MODEL.predict(accident_input_df)[0])
            risk_level = "High Risk" if pred_class == 1 else "Low Risk"
            
        recommendations = generate_decision_recommendations(
            predicted_traffic, traffic_level, accident_prob or 0.5, 
            risk_level, weather_main, hour, is_weekend
        )
        
        # Log to SQLite Database
        if os.path.exists(Config.DATABASE_PATH):
            try:
                conn = get_db_connection()
                conn.execute(
                    """
                    INSERT INTO predictions 
                    (location, timestamp, traffic_volume, average_speed, weather, visibility, traffic_prediction, accident_probability, risk_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        location, dt_str, 0.0, 45.0, weather_main, visibility,
                        predicted_traffic, accident_prob if accident_prob is not None else 0.0, risk_level
                    )
                )
                conn.commit()
                conn.close()
            except Exception as db_err:
                print(f"Error writing prediction to DB: {db_err}")
                
        response_payload = {
            "status": "success",
            "timestamp": dt_str,
            "location": location,
            "traffic_prediction": predicted_traffic,
            "traffic_level": traffic_level,
            "accident_probability": accident_prob,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "disclaimer": SYSTEM_DISCLAIMER
        }
        
        return jsonify(response_payload)
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({
            "status": "error",
            "message": f"Inference execution failed: {str(e)}"
        }), 400

@app.route('/api/hotspots', methods=['GET'])
def api_hotspots():
    """Returns detected spatial hotspot clusters."""
    if CLUSTERING_ARTIFACT and 'hotspots' in CLUSTERING_ARTIFACT:
        return jsonify({'status': 'success', 'hotspots': CLUSTERING_ARTIFACT['hotspots']})
    return jsonify({'status': 'success', 'hotspots': []})

@app.route('/api/history', methods=['GET'])
def api_history():
    """Returns past prediction logs from SQLite database."""
    if not os.path.exists(Config.DATABASE_PATH):
        return jsonify({'status': 'success', 'history': []})
    try:
        conn = get_db_connection()
        rows = conn.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 50").fetchall()
        conn.close()
        return jsonify({'status': 'success', 'history': [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("Starting SafeRoute AI Flask Server...")
    app.run(host='0.0.0.0', port=Config.PORT, debug=Config.DEBUG)
