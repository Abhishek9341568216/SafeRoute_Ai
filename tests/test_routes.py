import os
import sys
import json
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config
from app import app
from database.init_db import init_db
from src.data.download_data import prepare_all_data
from src.models.traffic_model import train_and_evaluate_traffic_models
from src.models.accident_model import train_and_evaluate_accident_models
from src.models.clustering_model import train_and_evaluate_clustering

@pytest.fixture(scope='module', autouse=True)
def setup_app_state():
    """Initializes datasets, loads models, and sets up SQLite DB before web route testing."""
    prepare_all_data()
    if not os.path.exists(Config.TRAFFIC_MODEL_PATH):
        train_and_evaluate_traffic_models()
    if not os.path.exists(Config.ACCIDENT_MODEL_PATH):
        train_and_evaluate_accident_models()
    if not os.path.exists(Config.CLUSTERING_MODEL_PATH):
        train_and_evaluate_clustering()
    init_db()

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b"SafeRoute AI" in rv.data

def test_dashboard_route(client):
    rv = client.get('/dashboard')
    assert rv.status_code == 200
    assert b"Executive Dashboard" in rv.data

def test_predict_page_route(client):
    rv = client.get('/predict')
    assert rv.status_code == 200
    assert b"Traffic & Risk Prediction Engine" in rv.data

def test_hotspots_route(client):
    rv = client.get('/hotspots')
    assert rv.status_code == 200
    assert b"Accident Hotspot" in rv.data

def test_analytics_route(client):
    rv = client.get('/analytics')
    assert rv.status_code == 200
    assert b"Analytics Dashboard" in rv.data

def test_models_route(client):
    rv = client.get('/models')
    assert rv.status_code == 200
    assert b"Model Comparison" in rv.data

def test_about_route(client):
    rv = client.get('/about')
    assert rv.status_code == 200
    assert b"About & Academic Methodology" in rv.data

def test_api_predict_success(client):
    payload = {
        "location": "Test Highway Corridor",
        "date_time": "2026-10-01T08:30:00",
        "temp": 22.5,
        "rain_1h": 0.0,
        "snow_1h": 0.0,
        "clouds_all": 10,
        "weather_main": "Clear",
        "holiday": "None",
        "visibility": 10.0,
        "humidity": 45.0,
        "wind_speed": 5.0
    }
    rv = client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert data['status'] == 'success'
    assert 'traffic_prediction' in data
    assert 'accident_probability' in data
    assert 'risk_level' in data
    assert isinstance(data['recommendations'], list)

def test_api_predict_invalid_input(client):
    payload = {
        "temp": "invalid_number_string"
    }
    rv = client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
    assert rv.status_code == 400
    data = json.loads(rv.data)
    assert data['status'] == 'error'
