# ============================================================================
# MAE - Makefile for Common Development Tasks
# ============================================================================
#
# Usage:
#   make help              Show this help message
#   make setup             Initial project setup
#   make dev               Start development environment
#   make test              Run all tests
#   make build             Build Docker images
#   make deploy            Deploy to local Kubernetes
#
# ============================================================================

.PHONY: help setup dev test clean build deploy

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m  # No Color

# Project variables
PROJECT_NAME := mae
PYTHON := python3
PIP := pip3
DOCKER_COMPOSE := docker-compose
KUBECTL := kubectl
HELM := helm

# ============================================================================
# Help
# ============================================================================

help:  ## Show this help message
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  MAE - Mycelial Agent Engine - Development Commands$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ============================================================================
# Setup and Installation
# ============================================================================

setup:  ## Initial project setup (install dependencies)
	@echo "$(BLUE)Setting up MAE development environment...$(NC)"
	@$(PYTHON) -m pip install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"
	@if [ ! -f .env ]; then cp .env.example .env && echo "$(GREEN)✓ Created .env file$(NC)"; fi
	@echo "$(GREEN)✓ Setup complete!$(NC)"

install:  ## Install Python dependencies
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev:  ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	@$(PIP) install -r requirements.txt
	@$(PIP) install pre-commit
	@pre-commit install
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

# ============================================================================
# Development
# ============================================================================

dev:  ## Start development environment with Docker Compose
	@echo "$(BLUE)Starting MAE development environment...$(NC)"
	@$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "  - Redis: http://localhost:6379"
	@echo "  - ChromaDB: http://localhost:8000"
	@echo "  - API: http://localhost:8080"
	@echo "  - API Docs: http://localhost:8080/docs"

dev-monitoring:  ## Start development environment with monitoring
	@echo "$(BLUE)Starting MAE with monitoring stack...$(NC)"
	@$(DOCKER_COMPOSE) --profile monitoring up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "  - API: http://localhost:8080"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"
	@echo "  - Jaeger: http://localhost:16686"

dev-debug:  ## Start development environment with debug tools
	@echo "$(BLUE)Starting MAE with debug tools...$(NC)"
	@$(DOCKER_COMPOSE) --profile debug up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "  - Redis Commander: http://localhost:8081"

stop:  ## Stop all Docker Compose services
	@echo "$(BLUE)Stopping MAE services...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Services stopped$(NC)"

restart:  ## Restart all services
	@make stop
	@make dev

logs:  ## Show logs from all services
	@$(DOCKER_COMPOSE) logs -f

logs-api:  ## Show logs from API service
	@$(DOCKER_COMPOSE) logs -f mae-api

# ============================================================================
# Testing
# ============================================================================

test:  ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	@$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-unit:  ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	@$(PYTHON) -m pytest tests/unit/ -v
	@echo "$(GREEN)✓ Unit tests complete$(NC)"

test-integration:  ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	@$(PYTHON) -m pytest tests/integration/ -v
	@echo "$(GREEN)✓ Integration tests complete$(NC)"

test-api:  ## Run API tests only
	@echo "$(BLUE)Running API tests...$(NC)"
	@$(PYTHON) -m pytest tests/unit/api/ -v
	@echo "$(GREEN)✓ API tests complete$(NC)"

test-coverage:  ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated: htmlcov/index.html$(NC)"

test-watch:  ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	@$(PYTHON) -m pytest-watch tests/ -v

# ============================================================================
# Code Quality
# ============================================================================

lint:  ## Run code linting
	@echo "$(BLUE)Running linters...$(NC)"
	@$(PYTHON) -m flake8 src/ tests/
	@$(PYTHON) -m pylint src/
	@echo "$(GREEN)✓ Linting complete$(NC)"

format:  ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(PYTHON) -m black src/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

type-check:  ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	@$(PYTHON) -m mypy src/
	@echo "$(GREEN)✓ Type checking complete$(NC)"

check:  ## Run all code quality checks
	@make format
	@make lint
	@make type-check
	@make test

# ============================================================================
# Docker
# ============================================================================

build:  ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	@$(DOCKER_COMPOSE) build
	@echo "$(GREEN)✓ Images built$(NC)"

build-api:  ## Build API Docker image
	@echo "$(BLUE)Building MAE API image...$(NC)"
	@docker build -f docker/Dockerfile.api -t mae-api:latest .
	@echo "$(GREEN)✓ API image built$(NC)"

push:  ## Push Docker images to registry
	@echo "$(BLUE)Pushing images to registry...$(NC)"
	@docker tag mae-api:latest ${REGISTRY}/mae-api:latest
	@docker push ${REGISTRY}/mae-api:latest
	@echo "$(GREEN)✓ Images pushed$(NC)"

# ============================================================================
# Kubernetes
# ============================================================================

k8s-deploy:  ## Deploy to Kubernetes
	@echo "$(BLUE)Deploying to Kubernetes...$(NC)"
	@$(KUBECTL) apply -f k8s/namespace.yaml
	@$(KUBECTL) apply -f k8s/base/
	@echo "$(GREEN)✓ Deployed to Kubernetes$(NC)"

k8s-delete:  ## Delete Kubernetes resources
	@echo "$(BLUE)Deleting Kubernetes resources...$(NC)"
	@$(KUBECTL) delete -f k8s/base/
	@echo "$(GREEN)✓ Resources deleted$(NC)"

k8s-status:  ## Show Kubernetes deployment status
	@echo "$(BLUE)Kubernetes Status:$(NC)"
	@$(KUBECTL) get all -n mae

k8s-logs:  ## Show Kubernetes pod logs
	@$(KUBECTL) logs -f -l app=mae -n mae

# ============================================================================
# Helm
# ============================================================================

helm-install:  ## Install MAE using Helm
	@echo "$(BLUE)Installing MAE with Helm...$(NC)"
	@$(HELM) install mae helm/mae-chart/ -n mae --create-namespace
	@echo "$(GREEN)✓ Helm chart installed$(NC)"

helm-upgrade:  ## Upgrade MAE Helm release
	@echo "$(BLUE)Upgrading MAE Helm release...$(NC)"
	@$(HELM) upgrade mae helm/mae-chart/ -n mae
	@echo "$(GREEN)✓ Helm chart upgraded$(NC)"

helm-uninstall:  ## Uninstall MAE Helm release
	@echo "$(BLUE)Uninstalling MAE Helm release...$(NC)"
	@$(HELM) uninstall mae -n mae
	@echo "$(GREEN)✓ Helm chart uninstalled$(NC)"

helm-test:  ## Test Helm chart
	@echo "$(BLUE)Testing Helm chart...$(NC)"
	@$(HELM) lint helm/mae-chart/
	@echo "$(GREEN)✓ Helm chart valid$(NC)"

# ============================================================================
# Simulation
# ============================================================================

simulate:  ## Run simulation
	@echo "$(BLUE)Running MAE simulation...$(NC)"
	@$(PYTHON) run_simulation.py
	@echo "$(GREEN)✓ Simulation complete$(NC)"

live:  ## Run in live mode
	@echo "$(BLUE)Starting MAE in live mode...$(NC)"
	@$(PYTHON) run_live.py

# ============================================================================
# Cleanup
# ============================================================================

clean:  ## Clean temporary files and caches
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name ".coverage" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-docker:  ## Remove Docker volumes and images
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	@$(DOCKER_COMPOSE) down -v
	@docker system prune -f
	@echo "$(GREEN)✓ Docker cleanup complete$(NC)"

clean-all:  ## Remove all generated files (clean + clean-docker)
	@make clean
	@make clean-docker

# ============================================================================
# Database
# ============================================================================

redis-cli:  ## Open Redis CLI
	@docker exec -it mae_redis redis-cli

redis-flush:  ## Flush Redis database (WARNING: deletes all data)
	@echo "$(RED)WARNING: This will delete all Redis data!$(NC)"
	@docker exec -it mae_redis redis-cli FLUSHALL

# ============================================================================
# Documentation
# ============================================================================

docs:  ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@cd docs && make html
	@echo "$(GREEN)✓ Documentation generated: docs/_build/html/index.html$(NC)"

docs-serve:  ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	@cd docs/_build/html && python -m http.server 8000

# ============================================================================
# Utilities
# ============================================================================

shell:  ## Open Python shell with MAE context
	@$(PYTHON) -i -c "import sys; sys.path.insert(0, 'src'); from src.agents.base_agent import BaseAgent; print('MAE environment loaded')"

version:  ## Show version information
	@echo "$(BLUE)MAE Version Information:$(NC)"
	@echo "  Python: $$($(PYTHON) --version)"
	@echo "  Docker: $$(docker --version)"
	@echo "  Docker Compose: $$($(DOCKER_COMPOSE) --version)"
	@echo "  Kubernetes: $$($(KUBECTL) version --client --short 2>/dev/null || echo 'Not installed')"
	@echo "  Helm: $$($(HELM) version --short 2>/dev/null || echo 'Not installed')"
