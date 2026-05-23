.PHONY: install test test-unit test-integration test-security test-all lint format run clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/unit/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-security:
	pytest tests/security/ -v

test-all:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

run:
	python app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info; \
	rm -f data/*.db logs/*.log logs/*.jsonl
