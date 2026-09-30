import os
import pandas as pd
from config import Config

def load_traffic_data(file_path=None):
    """Loads Metro Interstate Traffic Volume raw dataset."""
    path = file_path or Config.TRAFFIC_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Traffic dataset not found at '{path}'. "
            "Please run 'python src/data/download_data.py' or download Metro_Interstate_Traffic_Volume.csv."
        )
    df = pd.read_csv(path)
    return df

def load_accident_data(file_path=None):
    """Loads US Accidents raw dataset."""
    path = file_path or Config.ACCIDENT_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Accident dataset not found at '{path}'. "
            "Please run 'python src/data/download_data.py' or download US_Accidents_Sample.csv."
        )
    df = pd.read_csv(path)
    return df
