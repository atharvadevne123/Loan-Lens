"""FastAPI application for Loan-Lens credit risk scoring API."""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db, init_db
from app.explainability import get_shap_explanation
from app.middleware import RateLimitMiddleware
from app.model import predict as _predict
from app.monitoring import (
    compute_feature_drift,
    compute_prediction_stats,
    get_model_metrics,
    log_prediction,
)
from app.schemas import (
    DriftResponseSchema,
    HealthResponseSchema,
    LoanApplicationSchema,
    MetricsResponseSchema,
    PredictionResponseSchema,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()
MODEL_VERSION = settings.model_version


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Loan-Lens API started version=%s", MODEL_VERSION)
    yield
    logger.info("Loan-Lens API shutting down")


app = FastAPI(
    title="Loan-Lens",
    description=(
        "Production-grade credit risk scoring and loan default prediction API. "
        "Ensemble ML model with drift detection, explainability, and fairness metrics."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = round((time.monotonic() - start) * 1000, 2)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
    logger.info(
        "request path=%s status=%d ms=%s corr=%s",
        request.url.path, response.status_code, elapsed_ms, correlation_id,
    )
    return response


# ── API v1 endpoints ──────────────────────────────────────────────────────────

@app.get(
    "/api/v1/health",
    response_model=HealthResponseSchema,
    tags=["ops"],
    summary="Service liveness check",
    description="Returns HTTP 200 when the API is running and ready to serve predictions.",
)
def health() -> dict:
    return {"status": "ok", "version": MODEL_VERSION}


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponseSchema,
    tags=["prediction"],
    summary="Score a loan application for default risk",
    description=(
        "Accepts a loan application and returns a default probability (0–1), "
        "binary prediction, and risk tier (low/medium/high). "
        "Every prediction is persisted to the database."
    ),
)
def predict(
    application: LoanApplicationSchema,
    request: Request,
    db: Session = Depends(get_db),
) -> dict:
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    try:
        result = _predict(application.model_dump())
    except Exception as exc:
        logger.exception("prediction_failed corr=%s", correlation_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    log_prediction(
        db=db,
        correlation_id=correlation_id,
        input_features=application.model_dump(),
        probability=result["probability"],
        prediction=result["prediction"],
        model_version=MODEL_VERSION,
    )
    return {
        "correlation_id": correlation_id,
        "probability": result["probability"],
        "prediction": result["prediction"],
        "risk_level": result["risk_level"],
        "model_version": MODEL_VERSION,
    }


@app.post(
    "/api/v1/explain",
    tags=["prediction"],
    summary="Explain a loan default prediction",
    description="Returns the top-5 feature contributions driving the prediction using SHAP or finite differences.",
)
def explain(application: LoanApplicationSchema) -> dict:
    try:
        return get_shap_explanation(application.model_dump())
    except Exception as exc:
        logger.exception("explanation_failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/api/v1/metrics",
    response_model=MetricsResponseSchema,
    tags=["ops"],
    summary="Model performance and prediction statistics",
    description="Returns training AUC, recent default rate, and probability percentiles.",
)
def metrics(db: Session = Depends(get_db)) -> dict:
    return {
        "model": get_model_metrics(),
        "predictions": compute_prediction_stats(db),
        "version": MODEL_VERSION,
    }


@app.get(
    "/api/v1/drift",
    response_model=DriftResponseSchema,
    tags=["monitoring"],
    summary="Feature drift report",
    description=(
        "Runs KS-tests comparing the training reference distribution against "
        "the last 100 production predictions. Reports p-values and drift flags per feature."
    ),
)
def drift(db: Session = Depends(get_db)) -> dict:
    results = compute_feature_drift(db)
    drifted = [r for r in results if r["drift_detected"]]
    return {
        "features_checked": len(results),
        "features_drifted": len(drifted),
        "drift_details": results,
    }


# ── Batch prediction endpoint ─────────────────────────────────────────────────

class BatchRequest(BaseModel):
    applications: list[LoanApplicationSchema] = Field(..., min_length=1, max_length=500)


@app.post(
    "/api/v1/batch",
    tags=["prediction"],
    summary="Score a batch of loan applications",
    description="Accepts up to 500 applications and returns risk scores for all in parallel.",
)
def batch_predict_endpoint(batch: BatchRequest) -> dict:
    from app.batch import batch_predict

    results = batch_predict([a.model_dump() for a in batch.applications])
    return {"count": len(results), "results": results}
