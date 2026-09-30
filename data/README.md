# Data Directory

This directory contains datasets used by **SafeRoute AI**.

## Directory Structure
- `data/raw/`: Contains original, un-transformed raw datasets.
- `data/processed/`: Contains preprocessed and feature-engineered CSV files ready for modeling.
- `data/external/`: Contains supplementary geographic metadata or reference mappings.

## Datasets Overview

### 1. UCI Metro Interstate Traffic Volume Dataset
- **Source**: [UCI Machine Learning Repository - Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
- **Target File Path**: `data/raw/Metro_Interstate_Traffic_Volume.csv`
- **Description**: Hourly Interstate 94 traffic volume for WB I-94 at MN 280, Minneapolis-St. Paul, MN from 2012 to 2018, including weather details and holidays.
- **Key Columns**:
  - `holiday`: Categorical national holiday indicator
  - `temp`: Numeric temperature in Kelvin
  - `rain_1h`: Numeric mm of rain in hour
  - `snow_1h`: Numeric mm of snow in hour
  - `clouds_all`: Numeric percentage of cloud cover (0-100)
  - `weather_main`: Short textual weather summary (Clear, Rain, Snow, Drizzle, etc.)
  - `weather_description`: Detailed weather description
  - `date_time`: Datetime format YYYY-MM-DD HH:MM:SS
  - `traffic_volume`: Target continuous variable (Vehicles per hour)

### 2. US Accidents (2016 - 2023) Dataset
- **Source**: [Kaggle - US Accidents (2016-2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **Target File Path**: `data/raw/US_Accidents_Sample.csv` (or full dataset `US_Accidents_Dec23.csv`)
- **Description**: Countrywide traffic accident dataset covering the contiguous United States, collected from multiple API providers providing streaming traffic data.
- **Key Columns**:
  - `Severity`: Impact severity scale (1 to 4, where 4 has longest delay impact)
  - `Start_Time`, `End_Time`: Event timestamps
  - `Start_Lat`, `Start_Lng`: Geographic latitude and longitude
  - `Distance(mi)`: Roadway length affected
  - `City`, `County`, `State`: Location breakdown
  - `Temperature(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Speed(mph)`, `Precipitation(in)`
  - `Weather_Condition`: Categorical weather state
  - Road infrastructure indicators (`Crossing`, `Junction`, `Traffic_Signal`, `Stop`, etc.)

## Data Fetching & Automated Pipeline
Run the data fetching script to download or prepare datasets automatically:
```bash
python src/data/download_data.py
```
If network access is unavailable, `download_data.py` will generate realistic synthetic sample datasets adhering strictly to the official schemas so all ML pipelines, clustering models, SQLite databases, web endpoints, and unit tests execute end-to-end without errors.
