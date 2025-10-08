# Makefile for Pixel Voice API & MCP Server (using uv)

.PHONY: help setup install dev-install clean test lint format type-check run-api run-mcp run-examples docker-build docker-up docker-down

# Default target
help:
	@echo "Pixel Voice API & MCP Server - uv Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup          - Run complete setup (install uv, create venv, install deps)"
	@echo "  install        - Install project dependencies"
	@echo "  dev-install    - Install development dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linting (flake8)"
	@echo "  format         - Format code (black)"
	@echo "  type-check     - Run type checking (mypy)"
	@echo "  clean          - Clean up generated files"
	@echo ""
	@echo "Run Commands:"
	@echo "  run-api        - Start the API server"
	@echo "  run-mcp        - Start the MCP server"
	@echo "  run-examples   - Run example scripts"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build   - Build Docker images"
	@echo "  docker-up      - Start services with docker-compose"
	@echo "  docker-down    - Stop services with docker-compose"

# Setup commands
setup:
	@echo "Running complete setup..."
	python setup.py

install:
	@echo "Installing project dependencies..."
	uv pip install -e .

dev-install:
	@echo "Installing development dependencies..."
	uv pip install -e .[dev]

# Development commands
test:
	@echo "Running tests..."
	uv run pytest

test-cov:
	@echo "Running tests with coverage..."
	uv run pytest --cov=pixel_voice --cov-report=html

lint:
	@echo "Running linting..."
	uv run flake8 pixel_voice/

format:
	@echo "Formatting code..."
	uv run black pixel_voice/

format-check:
	@echo "Checking code format..."
	uv run black --check pixel_voice/

type-check:
	@echo "Running type checking..."
	uv run mypy pixel_voice/

quality: format lint type-check
	@echo "Running all quality checks..."

# Run commands
run-api:
	@echo "Starting API server..."
	uv run python start_api.py

run-mcp:
	@echo "Starting MCP server..."
	uv run python start_mcp.py

run-examples:
	@echo "Running examples..."
	uv run python run_examples.py

# Docker commands
docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-up:
	@echo "Starting services with Docker..."
	docker-compose up -d

docker-down:
	@echo "Stopping services..."
	docker-compose down

docker-logs:
	@echo "Showing Docker logs..."
	docker-compose logs -f

# Production deployment commands
deploy-staging:
	@echo "Deploying to staging..."
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/storage.yaml
	kubectl apply -f k8s/postgres.yaml
	kubectl apply -f k8s/redis.yaml
	sed 's|pixel-voice:latest|$(REGISTRY)/pixel-voice:staging|g' k8s/api-deployment.yaml | kubectl apply -f -
	kubectl rollout status deployment/pixel-voice-api -n pixel-voice

deploy-production:
	@echo "Deploying to production..."
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/storage.yaml
	kubectl apply -f k8s/postgres.yaml
	kubectl apply -f k8s/redis.yaml
	sed 's|pixel-voice:latest|$(REGISTRY)/pixel-voice:latest|g' k8s/api-deployment.yaml | kubectl apply -f -
	kubectl rollout status deployment/pixel-voice-api -n pixel-voice

deploy-monitoring:
	@echo "Deploying monitoring stack..."
	kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f monitoring/prometheus.yml
	kubectl apply -f monitoring/alert_rules.yml

k8s-status:
	@echo "Checking Kubernetes status..."
	kubectl get pods -n pixel-voice
	kubectl get services -n pixel-voice
	kubectl get ingress -n pixel-voice

k8s-logs:
	@echo "Showing Kubernetes logs..."
	kubectl logs -f deployment/pixel-voice-api -n pixel-voice

# Utility commands
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

venv-create:
	@echo "Creating virtual environment..."
	uv venv .venv

venv-activate:
	@echo "To activate virtual environment, run:"
	@echo "source .venv/bin/activate"

# Environment info
info:
	@echo "Environment Information:"
	@echo "======================="
	@which uv && uv --version || echo "uv not found"
	@which python && python --version || echo "python not found"
	@echo "Virtual environment: $${VIRTUAL_ENV:-Not activated}"
	@echo "Current directory: $$(pwd)"

# Install pre-commit hooks
pre-commit-install:
	@echo "Installing pre-commit hooks..."
	uv run pre-commit install

pre-commit-run:
	@echo "Running pre-commit on all files..."
	uv run pre-commit run --all-files
