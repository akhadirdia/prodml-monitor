from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Alert thresholds
    drift_warning_threshold: float = 0.15
    drift_critical_threshold: float = 0.30
    performance_degradation_threshold: float = 0.10  # 10% F1 drop triggers alert
    missing_labels_alert_hours: int = 48
    min_samples_for_metrics: int = 30

    # Monitoring windows (days)
    short_window_days: int = 7
    long_window_days: int = 30

    # Paths
    artifacts_dir: Path = BASE_DIR / "artifacts"
    model_path: Path = BASE_DIR / "artifacts" / "baseline_model.joblib"
    reference_data_path: Path = BASE_DIR / "artifacts" / "reference_data.csv"
    model_metadata_path: Path = BASE_DIR / "artifacts" / "model_metadata.json"
    db_path: Path = BASE_DIR / "monitoring.db"

    # Feature names used across the project
    feature_names: list[str] = [
        "feature_age",
        "feature_income",
        "feature_tenure",
        "feature_num_products",
        "feature_credit_score",
        "feature_balance",
        "feature_is_active",
        "feature_num_transactions",
        "feature_avg_transaction",
        "feature_debt_ratio",
    ]
    target_column: str = "target"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
