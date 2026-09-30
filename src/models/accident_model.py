import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import Config
from src.data.load_data import load_accident_data
from src.data.clean_data import clean_accident_data
from src.data.feature_engineering import engineer_accident_features
from src.evaluation.classification_metrics import evaluate_classification

def train_and_evaluate_accident_models():
    """
    Executes the Accident Risk Severity Classification Pipeline:
    1. Loads, cleans, and engineers US Accidents dataset
    2. Target: high_risk_target (Severity >= 3 -> 1, else 0)
    3. Preprocessing Pipeline with StandardScaler & OneHotEncoder
    4. Trains and evaluates 5 Classification Algorithms:
       - Logistic Regression
       - Decision Tree Classifier
       - Random Forest Classifier
       - SVM (Linear SGD Classifier with log_loss)
       - K-Nearest Neighbors (KNN)
    5. Computes Accuracy, Precision, Recall, F1, ROC-AUC & Confusion Matrix
    6. Persists best classification model to models/ using Joblib
    """
    print("=== ACCIDENT RISK SEVERITY CLASSIFICATION PIPELINE ===", flush=True)
    
    # 1. Load & Preprocess Data
    raw_df = load_accident_data()
    clean_df = clean_accident_data(raw_df)
    df = engineer_accident_features(clean_df)
    
    num_features = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'hour', 'day', 'month', 'day_of_week', 'is_weekend', 'is_rush_hour']
    cat_features = ['Weather_Condition', 'Sunrise_Sunset']
    
    for col in cat_features:
        if col not in df.columns:
            df[col] = 'Unknown'
            
    target = 'high_risk_target'
    
    X = df[num_features + cat_features]
    y = df[target]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Build ColumnTransformer Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )
    
    # 3. Define Classifiers
    models = {
        'Logistic Regression': Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(max_iter=500, random_state=42))
        ]),
        'Decision Tree Classifier': Pipeline([
            ('prep', preprocessor),
            ('clf', DecisionTreeClassifier(max_depth=10, random_state=42))
        ]),
        'Random Forest Classifier': Pipeline([
            ('prep', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1))
        ]),
        'SVM': Pipeline([
            ('prep', preprocessor),
            ('clf', SGDClassifier(loss='log_loss', max_iter=1000, random_state=42))
        ]),
        'KNN': Pipeline([
            ('prep', preprocessor),
            ('clf', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
        ])
    }
    
    results = []
    trained_pipelines = {}
    best_f1 = -float('inf')
    best_model_name = None
    
    print("\nTraining classification models...", flush=True)
    for name, pipeline in models.items():
        print(f"Training {name}...", flush=True)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        y_prob = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_prob = pipeline.predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = None
                
        metrics = evaluate_classification(y_test, y_pred, y_prob=y_prob, model_name=name)
        results.append(metrics)
        trained_pipelines[name] = pipeline
        
        print(f"  -> Accuracy: {metrics['accuracy']} | F1: {metrics['f1']} | Precision: {metrics['precision']} | Recall: {metrics['recall']}", flush=True)
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_name = name
            
    print(f"\nBest Accident Classifier: {best_model_name} (F1 = {best_f1})", flush=True)
    
    # 4. Save Best Model Pipeline
    best_pipeline = trained_pipelines[best_model_name]
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipeline, Config.ACCIDENT_MODEL_PATH)
    joblib.dump(preprocessor, Config.ACCIDENT_PREPROCESSOR_PATH)
    print(f"Persisted best accident model to '{Config.ACCIDENT_MODEL_PATH}'", flush=True)
    
    comparison_df = pd.DataFrame(results)
    return comparison_df, best_pipeline

if __name__ == '__main__':
    train_and_evaluate_accident_models()
