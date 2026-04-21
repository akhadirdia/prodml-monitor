"""Model registry — reads metadata from artifacts/model_metadata.json."""

import json
from functools import lru_cache

from loguru import logger

from core.config import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def get_current_model_info() -> dict:
    """Return model metadata including version, training date, and reference metrics.

    Raises:
        FileNotFoundError: if model_metadata.json has not been generated yet.
    """
    if not settings.model_metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at {settings.model_metadata_path}. "
            "Run notebooks/01_baseline_training.ipynb first."
        )
    data = json.loads(settings.model_metadata_path.read_text())
    logger.debug("Model registry loaded: version {}", data.get("version"))
    return data
