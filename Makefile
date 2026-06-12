.PHONY: install test lint format clean run-api help

install:
	pip install -r requirements.txt

test:
	pytest -v tests/

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

run-api:
	uvicorn web.app:app --reload --port 8000

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

help:
	@echo "Betting Intelligence System"
	@echo ""
	@echo "  make install   Install dependencies"
	@echo "  make run-api   Start web dashboard at localhost:8000"
	@echo "  make test      Run tests"
	@echo "  make lint      Run linter"
	@echo "  make clean     Remove build artifacts"
