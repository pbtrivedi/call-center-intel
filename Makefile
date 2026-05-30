.PHONY: install test test-unit test-integration test-security test-all lint format run stop clean clean-db eval-components eval-e2e

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

stop:
	@kill $$(lsof -ti :$$(python -c "from src.config.loader import get_settings; print(get_settings().app_port)")) 2>/dev/null && echo "App stopped." || echo "No app running on that port."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info; \
	rm -f logs/*.log logs/*.jsonl

eval-components:
	python evals/eval_qa_scoring.py
	python evals/eval_summarization.py

eval-e2e:
	python evals/eval_e2e_golden.py

clean-db:
	rm -f data/*.db
	@echo "Database wiped."
