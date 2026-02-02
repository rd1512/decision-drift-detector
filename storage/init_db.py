import sqlite3
from pathlib import Path

db_path = Path(__file__).resolve().parent / "predictions.db"
schema_path = Path(__file__).resolve().parent / "schema.sql"

conn = sqlite3.connect(db_path)

with open(schema_path, "r") as f:
    conn.executescript(f.read())

conn.commit()
conn.close()

print("Database initialized successfully.")
