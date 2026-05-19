"""A/B testing framework for comparing model versions in production."""

import hashlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelVariant:
    """A registered model variant with a traffic allocation weight."""

    name: str
    weight: float
    model_path: str = "model.joblib"
    metadata: dict = field(default_factory=dict)


class ABRouter:
    """Routes prediction requests to model variants using consistent hashing.

    Consistent hashing ensures the same applicant_id always routes to the
    same variant, enabling fair A/B comparisons.
    """

    def __init__(self, variants: list[ModelVariant]) -> None:
        total = sum(v.weight for v in variants)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Variant weights must sum to 1.0, got {total:.4f}")
        self._variants = variants

    def route(self, applicant_id: str) -> ModelVariant:
        """Return the variant assigned to applicant_id via consistent hashing."""
        digest = int(hashlib.md5(applicant_id.encode()).hexdigest(), 16)
        bucket = (digest % 10_000) / 10_000.0

        cumulative = 0.0
        for variant in self._variants:
            cumulative += variant.weight
            if bucket < cumulative:
                logger.debug("ab_route applicant=%s variant=%s bucket=%.4f", applicant_id, variant.name, bucket)
                return variant

        return self._variants[-1]

    def split_summary(self) -> dict:
        """Return configured variant names and their weights."""
        return {v.name: v.weight for v in self._variants}


_DEFAULT_ROUTER = ABRouter([
    ModelVariant(name="control", weight=0.9),
    ModelVariant(name="challenger", weight=0.1),
])


def get_router() -> ABRouter:
    return _DEFAULT_ROUTER
