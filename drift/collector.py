import uuid
from datetime import datetime

from storage.database import get_db_connection
from app.config import MODEL_VERSION


def log_prediction(
    decision: str,
    probability: float,
    latency_ms: float,
    input_summary: dict
) -> str:
    prediction_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions
        (prediction_id, timestamp, decision, probability, latency_ms,
         input_mean, input_std, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prediction_id,
            timestamp,
            decision,
            probability,
            latency_ms,
            input_summary["mean"],
            input_summary["std"],
            MODEL_VERSION
        )
    )

    conn.commit()
    conn.close()

    return prediction_id
