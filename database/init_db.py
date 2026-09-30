import os
import sys
import sqlite3
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

def init_db():
    """Initializes SQLite database and populates hotspot table from trained model."""
    os.makedirs(Config.DATABASE_DIR, exist_ok=True)
    db_path = Config.DATABASE_PATH
    schema_path = os.path.join(Config.DATABASE_DIR, 'schema.sql')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    cursor.executescript(schema_sql)
    conn.commit()
    print(f"Database schema initialized successfully at '{db_path}'.")
    
    # Populate hotspots table if clustering model exists
    if os.path.exists(Config.CLUSTERING_MODEL_PATH):
        try:
            artifact = joblib.load(Config.CLUSTERING_MODEL_PATH)
            hotspots = artifact.get('hotspots', [])
            
            # Clear existing hotspots
            cursor.execute("DELETE FROM hotspots")
            
            for h in hotspots:
                cursor.execute(
                    """
                    INSERT INTO hotspots (latitude, longitude, accident_count, risk_level, cluster_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (h['latitude'], h['longitude'], h['accident_count'], h['risk_level'], h['cluster_id'])
                )
            conn.commit()
            print(f"Populated {len(hotspots)} hotspot clusters into SQLite database.")
        except Exception as e:
            print(f"Warning populating database hotspots: {e}")
            
    conn.close()

if __name__ == '__main__':
    init_db()
