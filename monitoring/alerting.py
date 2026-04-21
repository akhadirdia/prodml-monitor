"""Alert evaluation and retrieval for the ML monitoring system."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger

import data.storage as storage
from core.config import get_settings
from monitoring.drift_detector import DriftReport
from monitoring.performance_tracker import PerformanceMetrics

settings = get_settings()


@dataclass
class Alert:
    """A single monitoring alert."""

    alert_type: str        # "data_drift" | "concept_drift" | "performance_degradation" | "missing_labels"
    severity: str          # "warning" | "critical"
    message: str
    timestamp: datetime
    affected_features: list[str] = field(default_factory=list)


class AlertManager:
    """Evaluates monitoring conditions and surfaces actionable alerts."""

    def evaluate_alerts(
        self,
        drift_report: DriftReport,
        performance_metrics: PerformanceMetrics | None = None,
        reference_metrics: dict | None = None,
        last_label_timestamp: datetime | None = None,
    ) -> list[Alert]:
        """Evaluate all alert conditions and return the list of active alerts.

        Args:
            drift_report: Latest drift detection result.
            performance_metrics: Latest performance snapshot, or None if labels unavailable.
            reference_metrics: Dict from model_metadata.json used for degradation comparison.
            last_label_timestamp: Timestamp of the most recent prediction with a real label.

        Returns:
            List of Alert objects. Empty list means everything is healthy.
        """
        alerts: list[Alert] = []
        now = datetime.now(timezone.utc)

        # --- Drift alerts ---
        if drift_report.severity in ("warning", "critical"):
            drift_type = drift_report.report_type  # "data_drift" or "concept_drift"
            if drift_type == "data_drift":
                pct = round(drift_report.drift_score * 100)
                if drift_report.severity == "critical":
                    msg = (
                        f"Alerte critique : {pct}% des variables d'entrée du modèle ont changé "
                        f"de distribution par rapport aux données d'entraînement. "
                        f"Variables concernées : {', '.join(drift_report.drifted_features)}. "
                        "Le modèle risque de produire des prédictions inexactes."
                    )
                else:
                    msg = (
                        f"Avertissement : {pct}% des variables d'entrée montrent des signes "
                        f"de dérive. Variables concernées : {', '.join(drift_report.drifted_features)}. "
                        "Situation à surveiller — aucune action immédiate requise."
                    )
            else:
                if drift_report.severity == "critical":
                    msg = (
                        "Alerte critique : le comportement du modèle a changé de façon significative. "
                        "Les prédictions produites aujourd'hui diffèrent de celles attendues sur des "
                        "données similaires. Une révision du modèle est recommandée."
                    )
                else:
                    msg = (
                        "Avertissement : de légères variations dans le comportement du modèle "
                        "ont été détectées. Surveillance renforcée recommandée."
                    )

            alerts.append(Alert(
                alert_type=drift_type,
                severity=drift_report.severity,
                message=msg,
                timestamp=now,
                affected_features=drift_report.drifted_features,
            ))

        # --- Performance degradation alert ---
        if performance_metrics is not None and reference_metrics is not None:
            ref_f1 = float(reference_metrics.get("f1", 0.0))
            cur_f1 = performance_metrics.f1
            if ref_f1 > 0:
                f1_drop = (ref_f1 - cur_f1) / ref_f1
                if f1_drop >= settings.performance_degradation_threshold:
                    pct_drop = round(f1_drop * 100, 1)
                    severity = "critical" if f1_drop >= 0.20 else "warning"
                    alerts.append(Alert(
                        alert_type="performance_degradation",
                        severity=severity,
                        message=(
                            f"La qualité des prédictions a baissé de {pct_drop}% "
                            f"(F1 actuel : {cur_f1:.2f} vs référence : {ref_f1:.2f}). "
                            "Vérifier si la dérive des données en est la cause et envisager "
                            "un réentraînement du modèle."
                        ),
                        timestamp=now,
                        affected_features=[],
                    ))

        # --- Missing labels alert ---
        if last_label_timestamp is not None:
            # Normalise to UTC-aware for comparison
            if last_label_timestamp.tzinfo is None:
                last_label_timestamp = last_label_timestamp.replace(tzinfo=timezone.utc)
            hours_since = (now - last_label_timestamp).total_seconds() / 3600
            if hours_since >= settings.missing_labels_alert_hours:
                alerts.append(Alert(
                    alert_type="missing_labels",
                    severity="warning",
                    message=(
                        f"Aucun résultat réel reçu depuis {int(hours_since)} heures. "
                        "Sans données de vérité terrain, il est impossible de mesurer la "
                        "performance réelle du modèle. Vérifier le pipeline de labellisation."
                    ),
                    timestamp=now,
                    affected_features=[],
                ))

        # Persist each alert to SQLite
        for alert in alerts:
            try:
                storage.save_alert(
                    timestamp=alert.timestamp,
                    alert_type=alert.alert_type,
                    severity=alert.severity,
                    message=alert.message,
                    affected_features=alert.affected_features,
                )
            except Exception as exc:
                logger.warning("Could not persist alert to SQLite: {}", exc)

        if alerts:
            logger.info("{} alert(s) generated: {}", len(alerts), [a.alert_type for a in alerts])
        else:
            logger.info("No alerts — system healthy.")

        return alerts

    def get_active_alerts(self, last_n_hours: int = 24) -> list[Alert]:
        """Return unacknowledged alerts from the last N hours."""
        try:
            rows = storage.get_active_alerts(last_n_hours)
        except Exception as exc:
            logger.warning("Could not retrieve alerts from SQLite: {}", exc)
            return []

        return [
            Alert(
                alert_type=row["alert_type"],
                severity=row["severity"],
                message=row["message"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                affected_features=row["affected_features"],
            )
            for row in rows
        ]
