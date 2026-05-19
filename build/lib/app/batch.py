"""Batch prediction runner for Loan-Lens."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.model import predict

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 500
MAX_WORKERS = 4


def batch_predict(applications: list[dict]) -> list[dict]:
    """Score a list of loan applications concurrently.

    Returns results in the same order as the input list.
    """
    if len(applications) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size {len(applications)} exceeds limit of {MAX_BATCH_SIZE}")

    results: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(predict, app): idx
            for idx, app in enumerate(applications)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error("batch_predict failed idx=%d error=%s", idx, exc)
                results[idx] = {"error": str(exc), "probability": None, "prediction": None, "risk_level": None}

    return [results[i] for i in range(len(applications))]
