import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import Config
from src.data.load_data import load_traffic_data
from src.data.clean_data import clean_traffic_data
from src.data.feature_engineering import engineer_traffic_features
from src.evaluation.regression_metrics import evaluate_regression

def train_and_evaluate_traffic_models():
    """
    Executes the Traffic Volume Regression Pipeline:
    1. Loads, cleans, and engineers Metro Interstate Traffic dataset
    2. Constructs Sklearn Preprocessing Pipeline
    3. Trains and evaluates 5 Regression Algorithms:
       - Linear Regression
       - Polynomial Regression (Degree 2)
       - Decision Tree Regressor
       - Random Forest Regressor
       - Gradient Boosting Regressor
    4. Compares metrics (MAE, MSE, RMSE, R²)
    5. Persists best model & scaler to models/ using Joblib
    """
    print("=== TRAFFIC VOLUME REGRESSION PIPELINE ===", flush=True)
    
    # 1. Load & Preprocess
    raw_df = load_traffic_data()
    clean_df = clean_traffic_data(raw_df)
    
    # Subsample if dataset is very large for rapid training performance
    if len(clean_df) > 10000:
        clean_df = clean_df.sample(n=10000, random_state=42).reset_index(drop=True)
        
    df = engineer_traffic_features(clean_df)
    
    # Define features and target
    num_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'day', 'month', 'day_of_week', 'is_weekend', 'is_rush_hour']
    cat_features = ['holiday', 'weather_main']
    target = 'traffic_volume'
    
    X = df[num_features + cat_features]
    y = df[target]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Build ColumnTransformer Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )
    
    # Save standalone scaler for general reference
    standalone_scaler = StandardScaler()
    standalone_scaler.fit(df[num_features])
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    joblib.dump(standalone_scaler, Config.SCALER_PATH)
    
    # 3. Model Definitions
    models = {
        'Linear Regression': Pipeline([
            ('prep', preprocessor),
            ('reg', LinearRegression())
        ]),
        'Polynomial Regression': Pipeline([
            ('prep', preprocessor),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('reg', LinearRegression())
        ]),
        'Decision Tree Regressor': Pipeline([
            ('prep', preprocessor),
            ('reg', DecisionTreeRegressor(max_depth=10, random_state=42))
        ]),
        'Random Forest Regressor': Pipeline([
            ('prep', preprocessor),
            ('reg', RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1))
        ]),
        'Gradient Boosting Regressor': Pipeline([
            ('prep', preprocessor),
            ('reg', GradientBoostingRegressor(n_estimators=30, max_depth=5, random_state=42))
        ])
    }
    
    results = []
    trained_pipelines = {}
    best_r2 = -float('inf')
    best_model_name = None
    
    print("\nTraining models and evaluating performance...", flush=True)
    for name, pipeline in models.items():
        print(f"Training {name}...", flush=True)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        metrics = evaluate_regression(y_test, y_pred, model_name=name)
        results.append(metrics)
        trained_pipelines[name] = pipeline
        
        print(f"  -> R²: {metrics['r2']} | RMSE: {metrics['rmse']} | MAE: {metrics['mae']}", flush=True)
        
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model_name = name
            
    print(f"\nBest Traffic Regression Model: {best_model_name} (R² = {best_r2})", flush=True)
    
    # 4. Save Best Model Pipeline
    best_pipeline = trained_pipelines[best_model_name]
    joblib.dump(best_pipeline, Config.TRAFFIC_MODEL_PATH)
    joblib.dump(preprocessor, Config.TRAFFIC_PREPROCESSOR_PATH)
    print(f"Persisted best traffic model to '{Config.TRAFFIC_MODEL_PATH}'", flush=True)
    
    comparison_df = pd.DataFrame(results)
    return comparison_df, best_pipeline

if __name__ == '__main__':
    train_and_evaluate_traffic_models()
