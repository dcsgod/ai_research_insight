# ============================================================
# AI Research Intelligence Platform - Makefile
# ============================================================
# Targets for development, testing, Docker operations, and CI.
# Run `make help` to see all available targets.
#
# Requirements:
#   - Python 3.11+ with pip
#   - Docker and Docker Compose v2+
#   - Node.js 20+ with npm
#   - GNU Make (on Windows: use "nmake" or WSL)
# ============================================================

.PHONY: help install install-dev dev docker-up docker-down docker-build \
        docker-logs docker-ps docker-clean init-db ingest ingest-arxiv \
        ingest-github test test-cov lint format type-check frontend-dev \
        frontend-build frontend-install celery-worker celery-beat flower \
        migrate migrate-create alembic-head clean

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
PYTHON         := python
PIP            := pip
DOCKER_COMPOSE := docker compose
PYTEST         := pytest
RUFF           := ruff
MYPY           := mypy
UVICORN        := uvicorn

BACKEND_MODULE := backend.main:app
CELERY_APP     := backend.celery_app

BACKEND_HOST   := 0.0.0.0
BACKEND_PORT   := 8000
FRONTEND_DIR   := frontend

-include .env
export

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:  ## Show this help message
	@echo ""
	@echo "  AI Research Intelligence Platform — Makefile"
	@echo "  ============================================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install:  ## Install Python production dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --no-cache-dir -r requirements.txt

install-dev:  ## Install Python dev + production dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --no-cache-dir -r requirements.txt
	$(PIP) install --no-cache-dir pre-commit
	pre-commit install

frontend-install:  ## Install Node.js dependencies for the frontend
	cd $(FRONTEND_DIR) && npm ci

# ---------------------------------------------------------------------------
# Development (local, no Docker)
# ---------------------------------------------------------------------------
dev:  ## Start infrastructure containers + run FastAPI with hot-reload
	$(DOCKER_COMPOSE) up -d postgres redis qdrant
	@echo "Waiting for services to be ready..."
	@sleep 3
	$(UVICORN) $(BACKEND_MODULE) \
		--host $(BACKEND_HOST) \
		--port $(BACKEND_PORT) \
		--reload \
		--log-level debug

frontend-dev:  ## Start Next.js development server with hot-reload
	cd $(FRONTEND_DIR) && npm run dev

# ---------------------------------------------------------------------------
# Docker Compose
# ---------------------------------------------------------------------------
docker-build:  ## Build all Docker images
	$(DOCKER_COMPOSE) build --no-cache

docker-up:  ## Start all services in the background
	$(DOCKER_COMPOSE) up -d

docker-down:  ## Stop and remove all containers (keep volumes)
	$(DOCKER_COMPOSE) down

docker-down-volumes:  ## Stop containers AND remove all volumes (destructive!)
	$(DOCKER_COMPOSE) down -v

docker-restart:  ## Restart all services
	$(DOCKER_COMPOSE) restart

docker-logs:  ## Follow logs for all services (Ctrl+C to stop)
	$(DOCKER_COMPOSE) logs -f --tail=100

docker-logs-backend:  ## Follow backend logs only
	$(DOCKER_COMPOSE) logs -f backend

docker-logs-worker:  ## Follow celery worker logs only
	$(DOCKER_COMPOSE) logs -f celery-worker

docker-ps:  ## Show status of all running containers
	$(DOCKER_COMPOSE) ps

docker-clean:  ## Remove stopped containers, dangling images, and unused networks
	docker system prune -f
	docker volume prune -f

frontend-build:  ## Build the Next.js production image
	$(DOCKER_COMPOSE) build frontend

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
init-db:  ## Initialize database: run migrations + create Qdrant collections + seed data
	$(PYTHON) scripts/init_db.py

init-db-no-seed:  ## Initialize database without seeding sample data
	$(PYTHON) scripts/init_db.py --skip-seed

migrate:  ## Run Alembic migrations (upgrade to head)
	alembic upgrade head

migrate-create:  ## Create a new Alembic migration (usage: make migrate-create MSG="add_users_table")
	alembic revision --autogenerate -m "$(MSG)"

alembic-head:  ## Show current migration head
	alembic current

migrate-downgrade:  ## Downgrade by 1 revision (usage: make migrate-downgrade)
	alembic downgrade -1

# ---------------------------------------------------------------------------
# Data Ingestion
# ---------------------------------------------------------------------------
ingest:  ## Run full ingestion (all sources)
	$(PYTHON) scripts/run_ingestion.py --sources all

ingest-arxiv:  ## Ingest from arXiv only
	$(PYTHON) scripts/run_ingestion.py --sources arxiv --verbose

ingest-github:  ## Ingest from GitHub only
	$(PYTHON) scripts/run_ingestion.py --sources github --verbose

ingest-huggingface:  ## Ingest from Hugging Face only
	$(PYTHON) scripts/run_ingestion.py --sources huggingface --verbose

ingest-reddit:  ## Ingest from Reddit only
	$(PYTHON) scripts/run_ingestion.py --sources reddit --verbose

ingest-pwc:  ## Ingest from Papers With Code only
	$(PYTHON) scripts/run_ingestion.py --sources paperswithcode --verbose

ingest-dry-run:  ## Run ingestion in dry-run mode (no writes)
	$(PYTHON) scripts/run_ingestion.py --sources all --dry-run

# ---------------------------------------------------------------------------
# Celery
# ---------------------------------------------------------------------------
celery-worker:  ## Start a Celery worker locally (requires Redis)
	celery -A $(CELERY_APP) worker \
		--loglevel=info \
		--concurrency=4 \
		--queues=ingestion,ranking,forecasting,default

celery-beat:  ## Start Celery beat scheduler locally
	celery -A $(CELERY_APP) beat --loglevel=info

flower:  ## Start Flower (Celery monitoring UI) on port 5555
	celery -A $(CELERY_APP) flower --port=5555

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test:  ## Run the full test suite with verbose output
	$(PYTEST) backend/tests/ -v --tb=short

test-cov:  ## Run tests with HTML coverage report
	$(PYTEST) backend/tests/ -v \
		--cov=backend \
		--cov-report=html:coverage_html \
		--cov-report=term-missing \
		--tb=short
	@echo "Coverage report: coverage_html/index.html"

test-fast:  ## Run tests, stop on first failure
	$(PYTEST) backend/tests/ -x --tb=short

test-watch:  ## Run tests in watch mode (requires pytest-watch)
	ptw backend/tests/ -- -v --tb=short

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------
lint:  ## Run Ruff linter and Mypy type checker
	$(RUFF) check .
	$(MYPY) backend/ --ignore-missing-imports

lint-fix:  ## Run Ruff with auto-fix enabled
	$(RUFF) check . --fix

format:  ## Format all Python files with Ruff
	$(RUFF) format .

format-check:  ## Check formatting without making changes
	$(RUFF) format . --check

type-check:  ## Run Mypy type checking only
	$(MYPY) backend/ --ignore-missing-imports --show-error-codes

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:  ## Remove Python cache files, coverage reports, and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf coverage_html/ .coverage htmlcov/ 2>/dev/null || true
	@echo "Cleaned up cache and build artifacts."
