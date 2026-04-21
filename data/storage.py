"""SQLite storage layer for predictions, drift reports, and performance metrics."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

import pandas as pd
from loguru import logger

from core.config import get_settings

settings = get_settings()

_CREATE_PREDICTIONS_LOG = """
CREATE TABLE IF NOT EXISTS predictions_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    feature_values  TEXT NOT NULL,
    prediction      INTEGER NOT NULL,
    probability     REAL NOT NULL,
    actual_label    INTEGER
)
"""

_CREATE_DRIFT_REPORTS = """
CREATE TABLE IF NOT EXISTS drift_reports (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    window_start      TEXT NOT NULL,
    window_end        TEXT NOT NULL,
    drift_score       REAL NOT NULL,
    drifted_features  TEXT NOT NULL,
    report_type       TEXT NOT NULL,
    severity          TEXT NOT NULL
)
"""

_CREATE_PERFORMANCE_METRICS = """
CREATE TABLE IF NOT EXISTS performance_metrics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    window_start TEXT NOT NULL,
    window_end   TEXT NOT NULL,
    accuracy     REAL,
    precision    REAL,
    recall       REAL,
    f1           REAL,
    roc_auc      REAL,
    n_samples    INTEGER NOT NULL
)
"""

_CREATE_ALERTS = """
CREATE TABLE IF NOT EXISTS alerts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    alert_type        TEXT NOT NULL,
    severity          TEXT NOT NULL,
    message           TEXT NOT NULL,
    affected_features TEXT NOT NULL,
    acknowledged      INTEGER NOT NULL DEFAULT 0
)
"""


@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they don't exist."""
    with _get_conn() as conn:
        conn.execute(_CREATE_PREDICTIONS_LOG)
        conn.execute(_CREATE_DRIFT_REPORTS)
        conn.execute(_CREATE_PERFORMANCE_METRICS)
        conn.execute(_CREATE_ALERTS)
    logger.info("Database initialized at {}", settings.db_path)


def log_prediction(
    timestamp: datetime,
    feature_values: dict[str, Any],
    prediction: int,
    probability: float,
    actual_label: int | None = None,
) -> None:
    """Insert a single prediction record."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO predictions_log (timestamp, feature_values, prediction, probability, actual_label) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                timestamp.isoformat(),
                json.dumps(feature_values),
                int(prediction),
                float(probability),
                actual_label,
            ),
        )


def save_drift_report(
    timestamp: datetime,
    window_start: datetime,
    window_end: datetime,
    drift_score: float,
    drifted_features: list[str],
    report_type: str,
    severity: str,
) -> None:
    """Persist a drift detection result."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO drift_reports "
            "(timestamp, window_start, window_end, drift_score, drifted_features, report_type, severity) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp.isoformat(),
                window_start.isoformat(),
                window_end.isoformat(),
                float(drift_score),
                json.dumps(drifted_features),
                report_type,
                severity,
            ),
        )


def save_performance_metrics(
    timestamp: datetime,
    window_start: datetime,
    window_end: datetime,
    n_samples: int,
    accuracy: float | None = None,
    precision: float | None = None,
    recall: float | None = None,
    f1: float | None = None,
    roc_auc: float | None = None,
) -> None:
    """Persist a performance metrics snapshot."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO performance_metrics "
            "(timestamp, window_start, window_end, accuracy, precision, recall, f1, roc_auc, n_samples) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp.isoformat(),
                window_start.isoformat(),
                window_end.isoformat(),
                accuracy,
                precision,
                recall,
                f1,
                roc_auc,
                n_samples,
            ),
        )


def get_predictions_window(start: datetime, end: datetime) -> pd.DataFrame:
    """Return all predictions in [start, end]."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions_log WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
            (start.isoformat(), end.isoformat()),
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        r = dict(row)
        r["feature_values"] = json.loads(r["feature_values"])
        records.append(r)

    return pd.DataFrame(records)


def get_drift_history(days: int) -> list[dict]:
    """Return drift reports from the last `days` days, ordered by timestamp."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM drift_reports "
            "WHERE timestamp >= datetime('now', ?) "
            "ORDER BY timestamp",
            (f"-{days} days",),
        ).fetchall()

    result = []
    for row in rows:
        r = dict(row)
        r["drifted_features"] = json.loads(r["drifted_features"])
        result.append(r)
    return result


def get_performance_history(days: int) -> list[dict]:
    """Return performance metrics from the last `days` days."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM performance_metrics "
            "WHERE timestamp >= datetime('now', ?) "
            "ORDER BY timestamp",
            (f"-{days} days",),
        ).fetchall()
    return [dict(row) for row in rows]


def save_alert(
    timestamp: datetime,
    alert_type: str,
    severity: str,
    message: str,
    affected_features: list[str],
) -> None:
    """Persist an alert."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO alerts (timestamp, alert_type, severity, message, affected_features) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                timestamp.isoformat(),
                alert_type,
                severity,
                message,
                json.dumps(affected_features),
            ),
        )


def get_active_alerts(last_n_hours: int = 24) -> list[dict]:
    """Return unacknowledged alerts from the last N hours."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts "
            "WHERE acknowledged = 0 AND timestamp >= datetime('now', ?) "
            "ORDER BY timestamp DESC",
            (f"-{last_n_hours} hours",),
        ).fetchall()
    result = []
    for row in rows:
        r = dict(row)
        r["affected_features"] = json.loads(r["affected_features"])
        result.append(r)
    return result


def acknowledge_alert(alert_id: int) -> None:
    """Mark an alert as acknowledged."""
    with _get_conn() as conn:
        conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
