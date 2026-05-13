# Changelog

All notable changes to Loan-Lens are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [1.0.0] - 2025-05-13

### Added

- FastAPI REST API with `/api/v1/predict`, `/api/v1/health`, `/api/v1/metrics`, `/api/v1/drift`
- Three-model soft-voting ensemble: XGBoost + LightGBM + RandomForest
- 26-feature engineering pipeline with ratios, log transforms, bins, and interaction terms
- 5-fold cross-validation AUC-ROC reporting
- KS-test based feature drift detection
- Prediction logging to SQLAlchemy / PostgreSQL
- Automated retraining pipeline (Airflow DAG + standalone runner)
- Correlation-ID middleware for request tracing
- Pydantic v2 input validation with custom validators
- Docker + docker-compose deployment with PostgreSQL
- GitHub Actions CI (ruff lint + pytest)
- pytest suite with 30+ test cases and parametrized tests
- Architecture diagram

---

## [Unreleased]

- SHAP explainability per prediction
- MLflow model registry integration
- Fairness metrics (demographic parity, equalized odds)
- Prometheus metrics endpoint
- A/B testing framework for model versions
