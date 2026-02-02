CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    decision TEXT NOT NULL,
    probability REAL NOT NULL,
    latency_ms REAL NOT NULL,
    input_mean REAL,
    input_std REAL,
    model_version TEXT NOT NULL
);

