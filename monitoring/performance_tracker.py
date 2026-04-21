"""Performance metric tracking for the deployed baseline model."""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from core.config import get_settings

settings = get_settings()


@dataclass
class PerformanceMetrics:
    """Snapshot of model performance over a time window."""

    window_start: datetime
    window_end: datetime
    n_samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


class PerformanceTracker:
    """Computes and compares model performance metrics."""

    def compute_metrics(
        self,
        predictions_df: pd.DataFrame,
        window_start: datetime,
        window_end: datetime,
    ) -> PerformanceMetrics | None:
        """Compute accuracy, precision, recall, F1, ROC-AUC over the window.

        Only rows where actual_label is not NULL are used. Returns None when
        fewer than min_samples_for_metrics labeled rows are available — this is
        the normal production state when ground-truth labels arrive with a delay.
        """
        if predictions_df.empty or "actual_label" not in predictions_df.columns:
            logger.debug("No predictions data or missing actual_label column.")
            return None

        labeled = predictions_df.dropna(subset=["actual_label"])
        n = len(labeled)

        if n < settings.min_samples_for_metrics:
            logger.info(
                "Only {} labeled samples in window — need at least {}. Skipping metrics.",
                n, settings.min_samples_for_metrics,
            )
            return None

        y_true = labeled["actual_label"].astype(int)
        y_pred = labeled["prediction"].astype(int)
        y_prob = labeled["probability"].astype(float)

        metrics = PerformanceMetrics(
            window_start=window_start,
            window_end=window_end,
            n_samples=n,
            accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
            precision=round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            recall=round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            f1=round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            roc_auc=round(float(roc_auc_score(y_true, y_prob)), 4),
        )
        logger.info(
            "Performance metrics | n={} | accuracy={} | f1={} | roc_auc={}",
            n, metrics.accuracy, metrics.f1, metrics.roc_auc,
        )
        return metrics

    def compute_performance_degradation(
        self,
        current_metrics: PerformanceMetrics,
        reference_metrics: dict,
    ) -> dict[str, float]:
        """Return % degradation per metric versus the reference (positive = worse).

        Args:
            current_metrics: Metrics computed on recent production data.
            reference_metrics: Dict from model_metadata.json (reference_metrics key).

        Returns:
            Dict mapping metric name → degradation as a fraction (e.g. 0.05 = 5% drop).
        """
        keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        result: dict[str, float] = {}
        for key in keys:
            ref_val = float(reference_metrics.get(key, 0.0))
            cur_val = float(getattr(current_metrics, key, 0.0))
            if ref_val > 0:
                result[key] = round((ref_val - cur_val) / ref_val, 4)
            else:
                result[key] = 0.0
        return result
