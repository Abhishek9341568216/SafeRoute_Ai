import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import Config
from src.data.load_data import load_accident_data
from src.data.clean_data import clean_accident_data

def train_and_evaluate_clustering():
    """
    Executes Spatial Accident Hotspot Detection:
    1. Loads cleaned US Accidents coordinates (Start_Lat, Start_Lng)
    2. Scales spatial features with StandardScaler
    3. Runs K-Means Clustering (e.g. k=8)
    4. Runs DBSCAN Clustering (eps=0.3, min_samples=15)
    5. Evaluates Silhouette Scores & noise point detection
    6. Extracts Hotspot summary (centroid coordinates, accident counts, risk scores)
    7. Persists K-Means model object to models/clustering_model.pkl
    """
    print("=== GEOGRAPHIC ACCIDENT HOTSPOT CLUSTERING PIPELINE ===")
    
    raw_df = load_accident_data()
    df = clean_accident_data(raw_df)
    
    coords = df[['Start_Lat', 'Start_Lng']].dropna()
    if len(coords) < 10:
        raise ValueError("Insufficient spatial coordinates for clustering.")
        
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coords)
    
    # 1. K-Means Clustering
    n_clusters = min(8, len(coords) // 100 + 2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(scaled_coords)
    kmeans_silhouette = silhouette_score(scaled_coords, kmeans_labels) if len(set(kmeans_labels)) > 1 else 0.0
    
    # 2. DBSCAN Clustering
    dbscan = DBSCAN(eps=0.3, min_samples=15)
    dbscan_labels = dbscan.fit_predict(scaled_coords)
    
    # Count noise points (-1)
    n_noise = list(dbscan_labels).count(-1)
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    
    dbscan_silhouette = 0.0
    if n_dbscan_clusters > 1:
        non_noise_mask = dbscan_labels != -1
        if np.sum(non_noise_mask) > 10:
            dbscan_silhouette = silhouette_score(scaled_coords[non_noise_mask], dbscan_labels[non_noise_mask])
            
    print("\nClustering Results Comparison:")
    print(f"  -> K-Means (k={n_clusters}): Silhouette Score = {round(kmeans_silhouette, 4)}")
    print(f"  -> DBSCAN (eps=0.3, min_samples=15): {n_dbscan_clusters} clusters, {n_noise} noise points, Silhouette Score = {round(dbscan_silhouette, 4)}")
    
    # Extract Cluster Hotspots details for SQLite and UI Leaflet Map
    coords_df = coords.copy()
    coords_df['cluster'] = kmeans_labels
    if 'Severity' in df.columns:
        coords_df['Severity'] = df['Severity']
    else:
        coords_df['Severity'] = 2
        
    hotspots = []
    for cid in range(n_clusters):
        cluster_points = coords_df[coords_df['cluster'] == cid]
        if len(cluster_points) == 0:
            continue
        center_lat = float(cluster_points['Start_Lat'].mean())
        center_lng = float(cluster_points['Start_Lng'].mean())
        acc_count = len(cluster_points)
        avg_severity = float(cluster_points['Severity'].mean())
        
        risk_level = "High" if avg_severity >= 2.8 or acc_count > 500 else ("Medium" if avg_severity >= 2.2 else "Low")
        
        hotspots.append({
            'cluster_id': int(cid),
            'latitude': round(center_lat, 6),
            'longitude': round(center_lng, 6),
            'accident_count': int(acc_count),
            'avg_severity': round(avg_severity, 2),
            'risk_level': risk_level
        })
        
    # Persist model artifact
    clustering_artifact = {
        'kmeans': kmeans,
        'scaler': scaler,
        'n_clusters': n_clusters,
        'kmeans_silhouette': kmeans_silhouette,
        'dbscan_silhouette': dbscan_silhouette,
        'n_dbscan_clusters': n_dbscan_clusters,
        'hotspots': hotspots
    }
    
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    joblib.dump(clustering_artifact, Config.CLUSTERING_MODEL_PATH)
    print(f"Saved clustering artifact to '{Config.CLUSTERING_MODEL_PATH}'")
    
    return hotspots, kmeans_silhouette, dbscan_silhouette

if __name__ == '__main__':
    train_and_evaluate_clustering()
