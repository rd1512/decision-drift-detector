import json
import numpy as np
from pathlib import Path
from storage.database import get_db_connection


BASELINE_PATH = Path(__file__).resolve().parent / "baseline_model_v0.1.0.json"
NUM_BINS = 5


def fetch_initial_predictions(limit=1000):
    """
    Fetch first N predictions to create baseline
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT probability FROM predictions
        ORDER BY timestamp ASC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [row[0] for row in rows]


def create_histogram(predictions):
    """
    Create normalized histogram from predictions
    """
    counts, bins = np.histogram(
        predictions,
        bins=NUM_BINS,
        range=(0.0, 1.0),
        density=False
    )

    return {
        "bins": bins.tolist(),
        "counts": counts.tolist()
    }


def save_baseline(histogram):
    """
    Save baseline distribution as JSON
    """
    with open(BASELINE_PATH, "w") as f:
        json.dump(histogram, f, indent=2)


def create_baseline():
    predictions = fetch_initial_predictions()

    if len(predictions) == 0:
        raise ValueError("No predictions found to create baseline.")

    histogram = create_histogram(predictions)
    save_baseline(histogram)

    print("Baseline distribution created and saved.")
    print(histogram)


if __name__ == "__main__":
    create_baseline()
