import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'saferoute-ai-lab-secret-key-2026'
    
    # Data Paths
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')
    
    TRAFFIC_DATA_PATH = os.path.join(RAW_DATA_DIR, 'Metro_Interstate_Traffic_Volume.csv')
    ACCIDENT_DATA_PATH = os.path.join(RAW_DATA_DIR, 'US_Accidents_Sample.csv')
    
    # Model Paths
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    TRAFFIC_MODEL_PATH = os.path.join(MODELS_DIR, 'traffic_model.pkl')
    ACCIDENT_MODEL_PATH = os.path.join(MODELS_DIR, 'accident_model.pkl')
    SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
    CLUSTERING_MODEL_PATH = os.path.join(MODELS_DIR, 'clustering_model.pkl')
    TRAFFIC_PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'traffic_preprocessor.pkl')
    ACCIDENT_PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'accident_preprocessor.pkl')
    
    # Database Path
    DATABASE_DIR = os.path.join(BASE_DIR, 'database')
    DATABASE_PATH = os.path.join(DATABASE_DIR, 'saferoute.db')
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH}"
    
    # App Settings
    DEBUG = True
    PORT = 5000
