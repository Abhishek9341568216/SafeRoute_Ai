import pandas as pd
import numpy as np

def engineer_traffic_features(df):
    """
    Engineers temporal and environmental features for Metro Interstate Traffic dataset:
    - hour (0-23)
    - day (1-31)
    - month (1-12)
    - day_of_week (0=Mon, 6=Sun)
    - is_weekend (0 or 1)
    - is_rush_hour (0 or 1)
    - temp_celsius (Celsius conversion from Kelvin)
    """
    data = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(data['date_time']):
        data['date_time'] = pd.to_datetime(data['date_time'])
        
    data['hour'] = data['date_time'].dt.hour
    data['day'] = data['date_time'].dt.day
    data['month'] = data['date_time'].dt.month
    data['day_of_week'] = data['date_time'].dt.dayofweek
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    
    # Peak commuting hours: 7-9 AM and 4-6 PM on weekdays
    data['is_rush_hour'] = (
        ((data['hour'].between(7, 9)) | (data['hour'].between(16, 18))) & 
        (data['is_weekend'] == 0)
    ).astype(int)
    
    # Temperature in Celsius
    data['temp_celsius'] = data['temp'] - 273.15
    
    return data

def engineer_accident_features(df):
    """
    Engineers temporal, environmental, and binary target features for US Accidents dataset.
    Target Construction Methodology:
    - High Risk (1): Severity >= 3 (significant traffic impact / severe injury)
    - Low/Moderate Risk (0): Severity < 3 (minor delay / localized impact)
    """
    data = df.copy()
    
    if 'Start_Time' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['Start_Time']):
        data['Start_Time'] = pd.to_datetime(data['Start_Time'])
        
    if 'Start_Time' in data.columns:
        data['hour'] = data['Start_Time'].dt.hour
        data['day'] = data['Start_Time'].dt.day
        data['month'] = data['Start_Time'].dt.month
        data['day_of_week'] = data['Start_Time'].dt.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_rush_hour'] = (
            ((data['hour'].between(7, 9)) | (data['hour'].between(16, 18))) & 
            (data['is_weekend'] == 0)
        ).astype(int)
        
    # Construct binary risk classification target from Severity
    if 'Severity' in data.columns:
        data['high_risk_target'] = (data['Severity'] >= 3).astype(int)
        
    # Convert boolean infrastructure flags to integers if present
    bool_cols = ['Junction', 'Traffic_Signal', 'Crossing', 'Amenity', 'Bump', 'Give_Way', 'No_Exit', 'Railway', 'Stop']
    for col in bool_cols:
        if col in data.columns:
            data[col] = data[col].astype(int)
            
    return data
