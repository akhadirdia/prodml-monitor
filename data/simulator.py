"""Synthetic data generator simulating three production drift scenarios."""

from datetime import date
import numpy as np
import pandas as pd

from core.config import get_settings

settings = get_settings()

# Reference distribution parameters (mirrors training data)
_REF_PARAMS: dict[str, dict] = {
    "feature_age":             {"mean": 40.0,    "std": 10.0},
    "feature_income":          {"mean": 55000.0, "std": 15000.0},
    "feature_tenure":          {"mean": 5.0,     "std": 3.0},
    "feature_num_products":    {"mean": 2.5,     "std": 1.0},
    "feature_credit_score":    {"mean": 680.0,   "std": 60.0},
    "feature_balance":         {"mean": 12000.0, "std": 8000.0},
    "feature_is_active":       {"p": 0.70},          # Bernoulli
    "feature_num_transactions": {"mean": 20.0,   "std": 8.0},
    "feature_avg_transaction": {"mean": 250.0,   "std": 80.0},
    "feature_debt_ratio":      {"mean": 0.35,    "std": 0.15},
}

# Features that drift in gradual/sudden scenarios and their drift magnitudes
_DRIFT_FEATURES = {
    "feature_income":       {"direction": 1.0,  "scale": 0.8},   # income increases
    "feature_credit_score": {"direction": -1.0, "scale": 0.6},   # scores drop
    "feature_balance":      {"direction": 1.0,  "scale": 0.7},
    "feature_debt_ratio":   {"direction": 1.0,  "scale": 0.5},
}


def _sample_feature(name: str, params: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    if "p" in params:
        return rng.binomial(1, params["p"], size=n).astype(float)
    return rng.normal(params["mean"], params["std"], size=n)


def _build_dataframe(feature_arrays: dict[str, np.ndarray], rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(feature_arrays)
    # Simple deterministic target for demo: high income + good credit → churn=0
    score = (
        (df["feature_credit_score"] - 680) / 60
        - (df["feature_debt_ratio"] - 0.35) / 0.15
        + df["feature_is_active"] * 0.5
    )
    prob = 1 / (1 + np.exp(-score))
    df[settings.target_column] = rng.binomial(1, prob).astype(int)
    return df


def generate_production_batch(
    scenario: str,
    batch_date: date,
    n_samples: int = 200,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Generate a production data batch for a given date and drift scenario.

    Args:
        scenario: One of "stable", "gradual_drift", "sudden_drift".
        batch_date: The production date for this batch.
        n_samples: Number of rows to generate.
        random_state: Optional seed for reproducibility.

    Returns:
        DataFrame with feature columns and target column.
    """
    if scenario not in ("stable", "gradual_drift", "sudden_drift"):
        raise ValueError(f"Unknown scenario: {scenario}. Choose stable/gradual_drift/sudden_drift.")

    seed = random_state if random_state is not None else (batch_date.toordinal() % 2**31)
    rng = np.random.default_rng(seed)

    drift_intensity = _compute_drift_intensity(scenario, batch_date)

    arrays: dict[str, np.ndarray] = {}
    for name, ref in _REF_PARAMS.items():
        params = dict(ref)
        if drift_intensity > 0 and name in _DRIFT_FEATURES:
            drift_cfg = _DRIFT_FEATURES[name]
            shift = drift_intensity * drift_cfg["scale"] * drift_cfg["direction"]
            if "mean" in params:
                params["mean"] = params["mean"] * (1 + shift)
            elif "p" in params:
                params["p"] = float(np.clip(params["p"] + shift * 0.3, 0.05, 0.95))
        arrays[name] = _sample_feature(name, params, n_samples, rng)

    df = _build_dataframe(arrays, rng)
    df["batch_date"] = batch_date.isoformat()
    return df


def _compute_drift_intensity(scenario: str, batch_date: date) -> float:
    """Return a 0–1 drift intensity for the given scenario and date."""
    if scenario == "stable":
        return 0.0

    if scenario == "gradual_drift":
        # Drift linearly increases over 30 days starting from day 1
        day_index = (batch_date - date(batch_date.year, batch_date.month, 1)).days
        return min(day_index / 29.0, 1.0)

    if scenario == "sudden_drift":
        # Full drift starting on the 15th of the month
        cutoff = date(batch_date.year, batch_date.month, 15)
        return 1.0 if batch_date >= cutoff else 0.0

    return 0.0
