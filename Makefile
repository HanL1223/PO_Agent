#Provde short commands for common task e.g.
#make tes
#make lint
#make docker-up




.PHONY: help install lint format test test-unit test-integration typecheck \
        serve dev docker-up docker-down ingest clean

# Default target: show help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'


install: ## Install all dependencies (Python + Node)
	pip install -e ".[dev]"
	cd frontend && npm ci


lint: ## Run linter (ruff)
	ruff check src/ backend/ tests/ scripts/

format: ## Format code (ruff)
	ruff format src/ backend/ tests/ scripts/

typecheck: ## Run type checker (mypy)
	mypy src/ backend/ --ignore-missing-imports


test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v -m "not integration"

test-integration: ## Run integration tests (requires API keys)
	pytest tests/integration/ -v -m integration

test-coverage: ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term


serve: ## Start the backend API server
	uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

dev: ## Start frontend dev server
	cd frontend && npm run dev



ingest: ## Ingest and index all spaces from data/raw/
	python scripts/ingest_and_index.py --data-dir data/raw

ingest-force: ## Force rebuild index from scratch
	python scripts/ingest_and_index.py --data-dir data/raw --force



docker-up: ## Start all services with Docker Compose
	docker compose -f infra/docker/docker-compose.yml up --build

docker-down: ## Stop all Docker services
	docker compose -f infra/docker/docker-compose.yml down

docker-build: ## Build Docker images without starting
	docker compose -f infra/docker/docker-compose.yml build



clean: ## Remove generated files and caches
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage
	rm -rf data/cache/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
