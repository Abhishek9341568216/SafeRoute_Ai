-- SafeRoute AI Database Schema

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    traffic_volume REAL,
    average_speed REAL,
    weather TEXT,
    visibility REAL,
    traffic_prediction REAL,
    accident_probability REAL,
    risk_level TEXT
);

CREATE TABLE IF NOT EXISTS hotspots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    accident_count INTEGER NOT NULL,
    risk_level TEXT NOT NULL,
    cluster_id INTEGER NOT NULL
);
