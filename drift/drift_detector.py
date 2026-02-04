import sqlite3
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List
from monitoring.slack_alerts import send_slack_alert
from prometheus_client import Gauge, start_http_server
import time



# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "storage", "predictions.db")
NUM_BINS = 5
EPSILON = 1e-6
MIN_SAMPLES = 30

# Drift threshold
DRIFT_THRESHOLD = 0.15  # tunable

kl_metric = Gauge(
    "model_kl_divergence",
    "KL divergence between baseline and recent predictions"
)


# Database helpers
def fetch_recent_predictions(hours: int = 24) -> List[float]:
    """
    Fetch prediction probabilities from the last `hours`.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    since_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    cursor.execute(
        """
        SELECT probability
        FROM predictions
        WHERE timestamp >= ?
        """,
        (since_time,)
    )

    rows = cursor.fetchall()
    conn.close()

    return [row[0] for row in rows]


def fetch_baseline_predictions(limit: int = 500) -> List[float]:
    """
    Fetch baseline predictions (earliest N rows).
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT probability
        FROM predictions
        ORDER BY timestamp ASC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()
    conn.close()

    return [row[0] for row in rows]


# Distribution logic
def build_distribution(probabilities: List[float]) -> np.ndarray:
    """
    Convert probabilities into a normalized histogram distribution.
    """
    hist, _ = np.histogram(
        probabilities,
        bins=NUM_BINS,
        range=(0.0, 1.0)
    )

    # Convert to float and add epsilon to avoid zeros
    hist = hist.astype(float) + EPSILON

    # Normalize
    distribution = hist / np.sum(hist)

    return distribution


# Drift calculation
def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL divergence: KL(P || Q)
    """
    return np.sum(p * np.log(p / q))


# Main runner
def run_drift_detection():
    baseline_probs = fetch_baseline_predictions()
    recent_probs = fetch_recent_predictions()

    print(f"Baseline samples: {len(baseline_probs)}")
    print(f"Recent samples: {len(recent_probs)}")

    if len(recent_probs) < MIN_SAMPLES:
        print("Not enough recent data to compute drift.")
        return

    baseline_dist = build_distribution(baseline_probs)
    recent_dist = build_distribution(recent_probs)

    kl_divergence = compute_kl_divergence(baseline_dist, recent_dist)
    kl_metric.set(kl_divergence)

    drift_detected = kl_divergence > DRIFT_THRESHOLD

    print(f"KL Divergence (baseline || recent): {kl_divergence:.6f}")

    if drift_detected:
        drift_excess = kl_divergence - DRIFT_THRESHOLD
        print(
            f"Drift detected: exceeded threshold by "
            f"{drift_excess:.6f} (threshold={DRIFT_THRESHOLD})"
        )
        send_slack_alert(
            kl_divergence,
            DRIFT_THRESHOLD,
            drift_excess
        )
    else:
        print("No significant drift detected")


if __name__ == "__main__":
    start_http_server(8001)
    run_drift_detection()

    # keep process alive so /metrics stays accessible
    while True:
        time.sleep(10)
