"""Application configuration loaded from environment variables."""

import os
from functools import lru_cache


class Settings:
    """Runtime settings resolved from environment variables with defaults."""

    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./loan_lens.db")
    model_version: str = os.getenv("MODEL_VERSION", "1.0.0")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    drift_window: int = int(os.getenv("DRIFT_WINDOW", "100"))
    drift_p_value_threshold: float = float(os.getenv("DRIFT_P_VALUE_THRESHOLD", "0.05"))
    retrain_drift_threshold: int = int(os.getenv("RETRAIN_DRIFT_THRESHOLD", "3"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
