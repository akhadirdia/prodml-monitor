"""Daily monitoring pipeline — generates a production batch, detects drift, evaluates alerts.

Usage:
    python run_monitoring.py --scenario stable
    python run_monitoring.py --scenario gradual_drift --date 2025-01-15
    python run_monitoring.py --scenario sudden_drift  --samples 300

Pre-fill the database with 60 days of history:
    python run_monitoring.py --prefill
"""

import argparse
import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from loguru import logger

import data.storage as storage
from core.config import get_settings
from data.simulator import generate_production_batch
from models.baseline_model import BaselineModel
from models.model_registry import get_current_model_info
from monitoring.alerting import AlertManager
from monitoring.drift_detector import DriftDetector
from monitoring.performance_tracker import PerformanceTracker

settings = get_settings()

_model = BaselineModel()
_detector = DriftDetector()
_tracker = PerformanceTracker()
_alert_manager = AlertManager()


def run_daily(scenario: str, batch_date: date, n_samples: int = 200) -> dict:
    """Execute one full monitoring cycle for a given production date.

    Returns a summary dict with drift scores, alert count, and performance metrics.
    """
    logger.info("=== Monitoring cycle | date={} | scenario={} ===", batch_date, scenario)
    now = datetime.now(timezone.utc)

    # 1. Generate production batch
    batch = generate_production_batch(scenario, batch_date, n_samples=n_samples)
    feature_df = batch.drop(columns=["batch_date", settings.target_column], errors="ignore")

    # 2. Predict
    predictions = _model.predict(feature_df)
    probabilities = _model.predict_proba(feature_df)[:, 1]

    # 3. Log predictions (simulate ~70% labeled with slight delay — real labels from target col)
    actual_labels = batch[settings.target_column].values if settings.target_column in batch.columns else None
    for i in range(len(feature_df)):
        # Simulate realistic label availability: only 70% of rows get a ground-truth label
        label = int(actual_labels[i]) if (actual_labels is not None and i % 10 < 7) else None
        storage.log_prediction(
            timestamp=datetime.combine(batch_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                      + timedelta(minutes=i),
            feature_values=feature_df.iloc[i].to_dict(),
            prediction=int(predictions[i]),
            probability=float(probabilities[i]),
            actual_label=label,
        )
    logger.info("Logged {} predictions ({} with labels)", len(feature_df), sum(1 for i in range(len(feature_df)) if i % 10 < 7))

    # 4. Fetch last 7 days from SQLite for drift window
    window_end = datetime.combine(batch_date, datetime.max.time()).replace(tzinfo=timezone.utc)
    window_start = window_end - timedelta(days=settings.short_window_days)
    predictions_df = storage.get_predictions_window(window_start, window_end)

    if predictions_df.empty or len(predictions_df) < settings.min_samples_for_metrics:
        logger.warning("Not enough data in window ({} rows) — skipping drift/performance.", len(predictions_df))
        return {"skipped": True, "reason": "insufficient_window_data"}

    # Reconstruct feature DataFrame from stored JSON
    feature_rows = predictions_df["feature_values"].apply(
        lambda v: v if isinstance(v, dict) else json.loads(v)
    )
    current_features = pd.DataFrame(feature_rows.tolist())

    # Load reference data for Evidently comparison
    reference_df = pd.read_csv(settings.reference_data_path)
    reference_features = reference_df[settings.feature_names]

    # 5. Detect drift (data + concept)
    data_drift = _detector.detect_data_drift(
        reference_data=reference_features,
        current_data=current_features,
        window_start=window_start,
        window_end=window_end,
    )
    ref_preds = _model.predict(reference_features)
    cur_preds = predictions_df["prediction"].values
    concept_drift = _detector.detect_concept_drift(
        reference_data=reference_features,
        current_data=current_features,
        predictions_ref=ref_preds,
        predictions_current=cur_preds.astype(int),
        window_start=window_start,
        window_end=window_end,
    )

    # 6. Performance metrics (requires real labels)
    model_info = get_current_model_info()
    ref_metrics = model_info["reference_metrics"]
    perf_metrics = _tracker.compute_metrics(predictions_df, window_start, window_end)

    # 7. Evaluate alerts
    last_labeled_row = (
        predictions_df.dropna(subset=["actual_label"])
        .sort_values("timestamp")
        .iloc[-1]
        if not predictions_df.dropna(subset=["actual_label"]).empty
        else None
    )
    last_label_ts = (
        datetime.fromisoformat(last_labeled_row["timestamp"])
        if last_labeled_row is not None
        else None
    )
    alerts = _alert_manager.evaluate_alerts(
        drift_report=data_drift,
        performance_metrics=perf_metrics,
        reference_metrics=ref_metrics,
        last_label_timestamp=last_label_ts,
    )

    # 8. Persist drift reports
    storage.save_drift_report(
        timestamp=now,
        window_start=window_start,
        window_end=window_end,
        drift_score=data_drift.drift_score,
        drifted_features=data_drift.drifted_features,
        report_type="data_drift",
        severity=data_drift.severity,
    )
    storage.save_drift_report(
        timestamp=now,
        window_start=window_start,
        window_end=window_end,
        drift_score=concept_drift.drift_score,
        drifted_features=concept_drift.drifted_features,
        report_type="concept_drift",
        severity=concept_drift.severity,
    )
    if perf_metrics is not None:
        storage.save_performance_metrics(
            timestamp=now,
            window_start=window_start,
            window_end=window_end,
            n_samples=perf_metrics.n_samples,
            accuracy=perf_metrics.accuracy,
            precision=perf_metrics.precision,
            recall=perf_metrics.recall,
            f1=perf_metrics.f1,
            roc_auc=perf_metrics.roc_auc,
        )

    # 9. Summary log
    summary = {
        "date": batch_date.isoformat(),
        "scenario": scenario,
        "data_drift_score": round(data_drift.drift_score, 3),
        "data_drift_severity": data_drift.severity,
        "drifted_features": data_drift.drifted_features,
        "concept_drift_severity": concept_drift.severity,
        "perf_f1": perf_metrics.f1 if perf_metrics else None,
        "alert_count": len(alerts),
        "alert_types": [a.alert_type for a in alerts],
    }
    logger.success(
        "Cycle complete | drift={} ({}) | perf_f1={} | alerts={}",
        summary["data_drift_score"],
        summary["data_drift_severity"],
        summary["perf_f1"],
        summary["alert_count"],
    )
    return summary


def prefill_database(start_date: date | None = None) -> None:
    """Generate 60 days of history: 30 stable + 30 gradual_drift."""
    if start_date is None:
        start_date = date(2025, 1, 1)
    logger.info("Pre-filling database from {} — 60 days of history", start_date)
    for i in range(30):
        d = start_date + timedelta(days=i)
        run_daily("stable", d, n_samples=150)
    for i in range(30):
        d = start_date + timedelta(days=30 + i)
        run_daily("gradual_drift", d, n_samples=150)
    logger.success("Pre-fill complete. Database ready for dashboard.")


def main() -> None:
    parser = argparse.ArgumentParser(description="ML Model Monitoring — daily pipeline")
    parser.add_argument("--scenario", choices=["stable", "gradual_drift", "sudden_drift"],
                        default="stable", help="Data generation scenario")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(),
                        help="Production date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of production samples to generate")
    parser.add_argument("--prefill", action="store_true",
                        help="Pre-fill database with 60 days of history and exit")
    args = parser.parse_args()

    storage.init_db()

    if args.prefill:
        prefill_database()
        return

    run_daily(args.scenario, args.batch_date if hasattr(args, "batch_date") else args.date, args.samples)


if __name__ == "__main__":
    main()
