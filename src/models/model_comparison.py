import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.traffic_model import train_and_evaluate_traffic_models
from src.models.accident_model import train_and_evaluate_accident_models
from src.models.clustering_model import train_and_evaluate_clustering

def run_full_model_comparison():
    """
    Runs complete model comparison suite across:
    1. Traffic Volume Regression (Linear, Poly, Decision Tree, Random Forest, Gradient Boosting)
    2. Accident Risk Classification (Logistic Reg, Decision Tree, Random Forest, SVM, KNN)
    3. Spatial Hotspot Clustering (K-Means vs DBSCAN)
    """
    traffic_metrics_df, _ = train_and_evaluate_traffic_models()
    accident_metrics_df, _ = train_and_evaluate_accident_models()
    hotspots, kmeans_sil, dbscan_sil = train_and_evaluate_clustering()
    
    clustering_summary = {
        'kmeans_silhouette': kmeans_sil,
        'dbscan_silhouette': dbscan_sil,
        'total_hotspots': len(hotspots)
    }
    
    return {
        'traffic_regression': traffic_metrics_df.to_dict(orient='records'),
        'accident_classification': accident_metrics_df.to_dict(orient='records'),
        'spatial_clustering': clustering_summary
    }

if __name__ == '__main__':
    res = run_full_model_comparison()
    print("Full Model Comparison Results:")
    print(res)
