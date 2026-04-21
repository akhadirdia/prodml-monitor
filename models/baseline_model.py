"""Wrapper around the saved baseline model."""

import numpy as np
import pandas as pd
import joblib
from loguru import logger

from core.config import get_settings

settings = get_settings()

_model = None


def _load_model():
    global _model
    if _model is None:
        if not settings.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {settings.model_path}. "
                "Run notebooks/01_baseline_training.ipynb first to generate it."
            )
        _model = joblib.load(settings.model_path)
        logger.info("Baseline model loaded from {}", settings.model_path)
    return _model


class BaselineModel:
    """Wrapper for the trained baseline classifier."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (0 or 1) for each row in X."""
        model = _load_model()
        return model.predict(X[settings.feature_names])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities. Column 1 is P(positive class)."""
        model = _load_model()
        return model.predict_proba(X[settings.feature_names])
