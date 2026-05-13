"""FastAPI application for Loan-Lens credit risk scoring."""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.model import get_model_metrics as _get_model_metrics, predict as _predict
from app.monitoring import (
    compute_feature_drift,
    compute_prediction_stats,
    get_model_metrics,
    log_prediction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Loan-Lens API started, DB initialised")
    yield
    logger.info("Loan-Lens API shutting down")


app = FastAPI(
    title="Loan-Lens",
    description="Credit risk scoring and loan default prediction API",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Correlation-ID middleware ──────────────────────────────────────────────────
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 2)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    logger.info("path=%s status=%d ms=%s corr=%s", request.url.path, response.status_code, elapsed, correlation_id)
    return response


# ── Request / Response schemas ────────────────────────────────────────────────
class LoanApplication(BaseModel):
    loan_amount: float = Field(..., gt=0, le=500000, description="Requested loan amount in USD")
    annual_income: float = Field(..., gt=0, description="Applicant annual income in USD")
    installment: float = Field(..., gt=0, description="Proposed monthly installment in USD")
    interest_rate: float = Field(..., ge=1, le=40, description="Annual interest rate (%)")
    loan_term_months: int = Field(..., ge=12, le=84, description="Loan term in months (12-84)")
    fico_score: int = Field(..., ge=300, le=850, description="FICO credit score")
    revolving_utilization: float = Field(..., ge=0, le=1, description="Credit utilization ratio")
    revolving_balance: float = Field(..., ge=0, description="Total revolving credit balance")
    delinquencies_2y: int = Field(0, ge=0, description="Delinquencies in past 2 years")
    credit_history_months: int = Field(..., ge=0, description="Months of credit history")
    open_accounts: int = Field(..., ge=0, description="Number of open credit accounts")
    total_accounts: int = Field(..., ge=0, description="Total number of credit accounts")
    public_records: int = Field(0, ge=0, description="Number of derogatory public records")

    @field_validator("total_accounts")
    @classmethod
    def total_must_gte_open(cls, v: int, info) -> int:
        open_acc = info.data.get("open_accounts", 0)
        if v < open_acc:
            raise ValueError("total_accounts must be >= open_accounts")
        return v


class PredictionResponse(BaseModel):
    correlation_id: str
    probability: float
    prediction: int
    risk_level: str
    model_version: str


# ── API v1 endpoints ──────────────────────────────────────────────────────────
@app.get("/api/v1/health", tags=["ops"], summary="Service health check")
def health():
    return {"status": "ok", "version": MODEL_VERSION}


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Score a loan application for default risk",
)
def predict(
    application: LoanApplication,
    request: Request,
    db: Session = Depends(get_db),
):
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    try:
        result = _predict(application.model_dump())
    except Exception as exc:
        logger.exception("Prediction failed corr=%s", correlation_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    log_prediction(
        db=db,
        correlation_id=correlation_id,
        input_features=application.model_dump(),
        probability=result["probability"],
        prediction=result["prediction"],
        model_version=MODEL_VERSION,
    )
    return PredictionResponse(
        correlation_id=correlation_id,
        probability=result["probability"],
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        model_version=MODEL_VERSION,
    )


@app.get("/api/v1/metrics", tags=["ops"], summary="Model performance and prediction statistics")
def metrics(db: Session = Depends(get_db)):
    model_metrics = get_model_metrics()
    pred_stats = compute_prediction_stats(db)
    return {
        "model": model_metrics,
        "predictions": pred_stats,
        "version": MODEL_VERSION,
    }


@app.get("/api/v1/drift", tags=["monitoring"], summary="Feature drift report using KS-test")
def drift(db: Session = Depends(get_db)):
    results = compute_feature_drift(db)
    drifted = [r for r in results if r["drift_detected"]]
    return {
        "features_checked": len(results),
        "features_drifted": len(drifted),
        "drift_details": results,
    }
