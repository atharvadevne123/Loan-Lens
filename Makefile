.PHONY: install test lint run docker-up docker-down retrain diagram clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

lint:
	ruff check .
	ruff format --check .

lint-fix:
	ruff check . --fix
	ruff format .

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

retrain:
	python3 -c "from pipelines.retrain_dag import retrain; print(retrain())"

diagram:
	python3 scripts/generate_diagram.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -f loan_lens.db test_loan_lens.db model.joblib metrics.json reference_data.json
