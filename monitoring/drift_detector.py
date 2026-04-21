"""Drift detection using Evidently AI."""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from evidently.report import Report
from loguru import logger

from core.config import get_settings

settings = get_settings()


@dataclass
class DriftReport:
    """Result of a single drift detection run."""

    report_type: str           # "data_drift" | "concept_drift"
    window_start: datetime
    window_end: datetime
    drift_score: float         # fraction of features that drifted (0.0–1.0)
    drifted_features: list[str]
    severity: str              # "ok" | "warning" | "critical"
    feature_scores: dict[str, float] = field(default_factory=dict)
    raw_report: dict = field(default_factory=dict, repr=False)


class DriftDetector:
    """Detects data drift and concept drift using Evidently."""

    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        window_start: datetime,
        window_end: datetime,
    ) -> DriftReport:
        """Compare feature distributions between reference and current data.

        Returns a DriftReport with drift_score (share of drifted features),
        per-feature drift scores, severity, and the raw Evidently report dict.
        """
        feature_cols = [c for c in settings.feature_names if c in reference_data.columns]
        ref = reference_data[feature_cols].copy()
        cur = current_data[feature_cols].copy()

        metrics = [DatasetDriftMetric()] + [ColumnDriftMetric(col) for col in feature_cols]
        report = Report(metrics=metrics)
        report.run(reference_data=ref, current_data=cur)
        raw = report.as_dict()

        # Overall drift score = share of drifted columns
        dataset_result = raw["metrics"][0]["result"]
        drift_score = float(dataset_result.get("share_of_drifted_columns", 0.0))

        # Per-feature drift scores
        feature_scores: dict[str, float] = {}
        drifted_features: list[str] = []
        for metric_entry in raw["metrics"][1:]:
            res = metric_entry["result"]
            col = res["column_name"]
            score = float(res.get("drift_score", 0.0))
            feature_scores[col] = score
            if res.get("drift_detected", False):
                drifted_features.append(col)

        severity = self._compute_severity(drift_score)
        logger.info(
            "Data drift | score={:.3f} | drifted={} | severity={}",
            drift_score, drifted_features, severity,
        )
        return DriftReport(
            report_type="data_drift",
            window_start=window_start,
            window_end=window_end,
            drift_score=drift_score,
            drifted_features=drifted_features,
            severity=severity,
            feature_scores=feature_scores,
            raw_report=raw,
        )

    def detect_concept_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        predictions_ref: np.ndarray,
        predictions_current: np.ndarray,
        window_start: datetime,
        window_end: datetime,
    ) -> DriftReport:
        """Detect if the feature→prediction relationship has changed.

        Uses TargetDriftPreset — compares the distribution of model predictions
        between the reference window and the current window.
        """
        target_col = "prediction"
        feature_cols = [c for c in settings.feature_names if c in reference_data.columns]

        ref = reference_data[feature_cols].copy()
        ref[target_col] = predictions_ref.astype(float)

        cur = current_data[feature_cols].copy()
        cur[target_col] = predictions_current.astype(float)

        report = Report(metrics=[TargetDriftPreset()])
        report.run(reference_data=ref, current_data=cur, column_mapping=None)
        raw = report.as_dict()

        # TargetDriftPreset first metric is ColumnDriftMetric on the target
        target_result = raw["metrics"][0]["result"]
        drift_score = float(target_result.get("drift_score", 0.0))
        drift_detected = bool(target_result.get("drift_detected", False))

        drifted_features = [target_col] if drift_detected else []
        # Normalise to 0–1 for severity logic (treat as single-feature drift share)
        severity = self._compute_severity(1.0 if drift_detected else 0.0)

        logger.info(
            "Concept drift | score={:.3f} | detected={} | severity={}",
            drift_score, drift_detected, severity,
        )
        return DriftReport(
            report_type="concept_drift",
            window_start=window_start,
            window_end=window_end,
            drift_score=drift_score,
            drifted_features=drifted_features,
            severity=severity,
            feature_scores={target_col: drift_score},
            raw_report=raw,
        )

    def _compute_severity(self, drift_score: float) -> str:
        if drift_score >= settings.drift_critical_threshold:
            return "critical"
        if drift_score >= settings.drift_warning_threshold:
            return "warning"
        return "ok"
