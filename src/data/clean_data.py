import pandas as pd
import numpy as np

def clean_traffic_data(df):
    """
    Cleans raw Metro Interstate Traffic Volume dataset:
    - Handles temperature outliers (0 Kelvin converted to median/NaN)
    - Removes duplicates
    - Formats datetime
    - Imputes numerical/categorical missing values
    """
    data = df.copy()
    
    # Format date_time
    data['date_time'] = pd.to_datetime(data['date_time'])
    
    # Remove duplicate date_time entries
    data = data.drop_duplicates(subset=['date_time']).reset_index(drop=True)
    
    # Handle temperature outliers (e.g. 0 Kelvin is physically invalid for outdoor traffic measurements)
    if 'temp' in data.columns:
        median_temp = data[data['temp'] > 100]['temp'].median()
        if pd.isna(median_temp):
            median_temp = 288.15  # Default ~15°C / 59°F in Kelvin
        data.loc[data['temp'] < 100, 'temp'] = median_temp
        
    # Impute missing values if present
    num_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume']
    for col in num_cols:
        if col in data.columns and data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())
            
    cat_cols = ['holiday', 'weather_main', 'weather_description']
    for col in cat_cols:
        if col in data.columns and data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].mode()[0])
            
    return data

def clean_accident_data(df):
    """
    Cleans raw US Accidents dataset:
    - Formats timestamps
    - Filters invalid geographical coordinates
    - Imputes missing numerical and categorical values
    """
    data = df.copy()
    
    # Timestamp parsing
    if 'Start_Time' in data.columns:
        data['Start_Time'] = pd.to_datetime(data['Start_Time'])
        
    # Remove rows with invalid coordinates
    if 'Start_Lat' in data.columns and 'Start_Lng' in data.columns:
        valid_coords = (
            (data['Start_Lat'].between(24.0, 50.0)) & 
            (data['Start_Lng'].between(-125.0, -65.0))
        )
        data = data[valid_coords].reset_index(drop=True)
        
    # Impute numerical features
    num_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
    for col in num_cols:
        if col in data.columns and data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())
            
    # Impute categorical features
    cat_cols = ['Weather_Condition', 'Sunrise_Sunset', 'City', 'State']
    for col in cat_cols:
        if col in data.columns and data[col].isnull().sum() > 0:
            data[col] = data[col].fillna('Unknown')
            
    return data
