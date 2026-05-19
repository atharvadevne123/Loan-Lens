"""Pydantic request/response schemas for Loan-Lens API."""

from pydantic import BaseModel, Field, field_validator


class LoanApplicationSchema(BaseModel):
    """Input schema for a loan default risk assessment request."""

    loan_amount: float = Field(..., gt=0, le=500000, description="Requested loan amount in USD")
    annual_income: float = Field(..., gt=0, description="Applicant annual income in USD")
    installment: float = Field(..., gt=0, description="Proposed monthly installment in USD")
    interest_rate: float = Field(..., ge=1, le=40, description="Annual interest rate (%)")
    loan_term_months: int = Field(..., ge=12, le=84, description="Loan term in months (12–84)")
    fico_score: int = Field(..., ge=300, le=850, description="FICO credit score (300–850)")
    revolving_utilization: float = Field(..., ge=0, le=1, description="Credit utilization ratio (0–1)")
    revolving_balance: float = Field(..., ge=0, description="Total revolving credit balance in USD")
    delinquencies_2y: int = Field(0, ge=0, description="Number of delinquencies in past 2 years")
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

    model_config = {"json_schema_extra": {
        "example": {
            "loan_amount": 10000.0, "annual_income": 60000.0, "installment": 320.0,
            "interest_rate": 12.5, "loan_term_months": 36, "fico_score": 700,
            "revolving_utilization": 0.25, "revolving_balance": 5000.0,
            "delinquencies_2y": 0, "credit_history_months": 84,
            "open_accounts": 5, "total_accounts": 12, "public_records": 0,
        }
    }}


class PredictionResponseSchema(BaseModel):
    """Output schema for a loan risk prediction."""

    correlation_id: str = Field(..., description="Unique request identifier")
    probability: float = Field(..., ge=0, le=1, description="Default probability (0–1)")
    prediction: int = Field(..., description="Binary prediction: 1=default, 0=non-default")
    risk_level: str = Field(..., description="Risk tier: low | medium | high")
    model_version: str = Field(..., description="Model version used for inference")


class HealthResponseSchema(BaseModel):
    """Health check response."""

    status: str
    version: str


class MetricsResponseSchema(BaseModel):
    """Model and prediction metrics response."""

    model: dict
    predictions: dict
    version: str


class DriftResponseSchema(BaseModel):
    """Feature drift analysis response."""

    features_checked: int
    features_drifted: int
    drift_details: list[dict]
