import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config
from src.data.download_data import prepare_all_data
from src.data.load_data import load_traffic_data, load_accident_data
from src.data.clean_data import clean_traffic_data, clean_accident_data
from src.data.feature_engineering import engineer_traffic_features, engineer_accident_features
from src.models.traffic_model import train_and_evaluate_traffic_models
from src.models.accident_model import train_and_evaluate_accident_models
from src.models.clustering_model import train_and_evaluate_clustering

@pytest.fixture(scope='module', autouse=True)
def setup_datasets():
    """Ensures datasets and raw data exist before tests run."""
    prepare_all_data()

def test_load_traffic_data():
    df = load_traffic_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'traffic_volume' in df.columns

def test_load_accident_data():
    df = load_accident_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'Severity' in df.columns

def test_clean_and_engineer_traffic():
    raw = load_traffic_data()
    clean = clean_traffic_data(raw)
    feat = engineer_traffic_features(clean)
    assert 'hour' in feat.columns
    assert 'is_weekend' in feat.columns
    assert 'is_rush_hour' in feat.columns
    assert 'temp_celsius' in feat.columns

def test_clean_and_engineer_accidents():
    raw = load_accident_data()
    clean = clean_accident_data(raw)
    feat = engineer_accident_features(clean)
    assert 'high_risk_target' in feat.columns
    assert set(feat['high_risk_target'].unique()).issubset({0, 1})

def test_traffic_model_training():
    metrics_df, best_model = train_and_evaluate_traffic_models()
    assert isinstance(metrics_df, pd.DataFrame)
    assert not metrics_df.empty
    assert os.path.exists(Config.TRAFFIC_MODEL_PATH)
    assert os.path.exists(Config.SCALER_PATH)

def test_accident_model_training():
    metrics_df, best_model = train_and_evaluate_accident_models()
    assert isinstance(metrics_df, pd.DataFrame)
    assert not metrics_df.empty
    assert os.path.exists(Config.ACCIDENT_MODEL_PATH)

def test_clustering_training():
    hotspots, k_sil, db_sil = train_and_evaluate_clustering()
    assert isinstance(hotspots, list)
    assert os.path.exists(Config.CLUSTERING_MODEL_PATH)
