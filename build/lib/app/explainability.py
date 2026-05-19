"""SHAP-based model explainability for loan default predictions."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_shap_explanation(features: dict) -> dict:
    """Return top-5 SHAP feature contributions for a prediction.

    Falls back to a gradient-free approximation if shap is not installed.
    """
    try:
        import shap

        from app.features import FEATURE_COLUMNS
        from app.model import load_model

        pipe, engineer = load_model()
        df = pd.DataFrame([features])
        X_eng = engineer.transform(df)
        X_feat = X_eng[FEATURE_COLUMNS].values

        try:
            explainer = shap.TreeExplainer(pipe.named_steps["model"].estimators_[0])
            shap_values = explainer.shap_values(X_feat)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            contributions = dict(zip(FEATURE_COLUMNS, shap_values[0].tolist(), strict=False))
            top5 = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            return {"method": "shap", "top_features": [{"feature": k, "contribution": round(v, 4)} for k, v in top5]}
        except Exception as inner:
            logger.debug("SHAP tree explainer failed: %s", inner)
    except ImportError:
        pass

    # Gradient-free fallback: feature sensitivity via finite differences
    return _finite_diff_explanation(features)


def _finite_diff_explanation(features: dict) -> dict:
    """Approximate feature importance via finite differences on the ensemble."""
    from app.model import predict

    baseline = predict(features)["probability"]
    contributions = []

    numeric_keys = [k for k, v in features.items() if isinstance(v, (int, float))]
    for key in numeric_keys:
        perturbed = dict(features)
        delta = max(abs(features[key]) * 0.01, 1e-4)
        perturbed[key] = features[key] + delta
        try:
            new_prob = predict(perturbed)["probability"]
            sensitivity = (new_prob - baseline) / delta
            contributions.append({"feature": key, "contribution": round(sensitivity, 4)})
        except Exception:
            continue

    top5 = sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)[:5]
    return {"method": "finite_diff", "top_features": top5}
