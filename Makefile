.PHONY: install dev test test-slow test-fast lint format clean clean-output \
  ci yaml-validate typecheck \
  run-api run-pipeline \
  docker-build docker-up docker-down docker-logs \
  db-upgrade db-downgrade db-migrate db-history db-current \
  refresh-data refresh-data-fast refresh-data-backfill verify-data help

# ── Installation ──────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

dev:
	pip install -e ".[dev]"

# ── Testing ───────────────────────────────────────────────────────────────
test:
	pytest -v --cov=src/betting_intel --cov-report=term-missing

test-slow:
	pytest -v --cov=src/betting_intel -m slow

test-fast:
	pytest -v -m "not slow"

# ── Linting ───────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# ── CI Checks (mirrors .github/workflows/ci.yml locally) ───────────────
ci: yaml-validate lint typecheck test-fast
	@echo "✅ All CI checks passed"

yaml-validate:
	@echo "🔍 Validating YAML workflow files..."
	@for f in .github/workflows/*.yml; do \
		python -c "import yaml, sys; yaml.safe_load(open('$$f', encoding='utf-8')); sys.stderr.write('  ✓ ' + '$$f' + '\n')" 2>&1 || \
		(echo "  ✗ $$f — INVALID" && exit 1); \
	done
	@echo "✅ All YAML files valid"

typecheck:
	@echo "🔍 Running mypy type checker..."
	mypy src/betting_intel/ --ignore-missing-imports --warn-unused-ignores
	@echo "✅ Type check complete"

# ── Running ───────────────────────────────────────────────────────────────
run-pipeline:
	betting-intel run-pipeline

run-api:
	betting-api

# ── Database (Alembic Migrations) ───────────────────────────────────────
db-upgrade:
	PYTHONPATH=src alembic upgrade head

db-downgrade:
	PYTHONPATH=src alembic downgrade -1

db-migrate:
	PYTHONPATH=src alembic revision --autogenerate -m "$(msg)"

db-history:
	PYTHONPATH=src alembic history

db-current:
	PYTHONPATH=src alembic current

# ── Docker ────────────────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# ── Cleanup ───────────────────────────────────────────────────────────────
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-output:
	rm -f output/*.csv output/*.json output/*.txt

# ── Real NBA Data ────────────────────────────────────────────────────────
refresh-data:
	python scripts/fetch_real_nba_data.py

refresh-data-fast:
	python scripts/fetch_real_nba_data.py --fast

refresh-data-backfill:
	python scripts/fetch_real_nba_data.py --backfill

verify-data:
	python scripts/fetch_real_nba_data.py --verify

# ── Help ──────────────────────────────────────────────────────────────────
help:
	@echo "Betting Intelligence System - Makefile"
	@echo ""
	@echo "Installation:"
	@echo "  make install     Install package with all dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test        Run all tests with coverage"
	@echo "  make test-fast   Run fast tests only (excludes @pytest.mark.slow)"
	@echo "  make test-slow   Run slow tests only"
	@echo ""
	@echo "CI Checks (local):"
	@echo "  make ci              Run all CI checks: YAML → Ruff → mypy → pytest"
	@echo "  make yaml-validate   Validate all .github/workflows/*.yml files"
	@echo "  make lint            Run ruff linter + format check"
	@echo "  make typecheck       Run mypy type checker"
	@echo ""
	@echo "Running:"
	@echo "  make run-pipeline   Execute full data pipeline"
	@echo "  make run-api        Start FastAPI server"
	@echo "  make run-dashboard  Start Streamlit dashboard"
	@echo ""
	@echo "Data:"
	@echo "  make refresh-data        Fetch all NBA seasons from ESPN (full)"
	@echo "  make refresh-data-fast   Scoreboard only (no boxscore backfill)"
	@echo "  make verify-data         Verify database contents"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all services"
	@echo ""
	@echo "Database:"
	@echo "  make db-upgrade       Apply all pending Alembic migrations"
	@echo "  make db-downgrade     Roll back the last migration"
	@echo "  make db-migrate msg=  Create a new auto-generated migration"
	@echo "  make db-history      Show migration history"
	@echo "  make db-current      Show current migration version"
