# Pixelated Empathy AI submodule — local verification targets.
#
# Source of truth for these targets is `.github/workflows/training-safety-coverage.yml`.
# Keep the local targets in sync with CI thresholds when bumping them.

PYTEST ?= uv run pytest
PYTEST_IGNORE := --ignore=training/tests/test_book_pdf_converter.py

.PHONY: help test test-fast coverage-safety coverage-pilot coverage-all lint clean

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

test: ## Run the full training test suite (matches CI invocation).
	$(PYTEST) training/tests/ $(PYTEST_IGNORE) --no-header -q

test-fast: ## Same as `test` but without -q output for richer diff on failure.
	$(PYTEST) training/tests/ $(PYTEST_IGNORE) --no-header -v

coverage-safety: ## Run the safety-critical coverage gate (95%, branch).
	mkdir -p coverage
	$(PYTEST) training/tests $(PYTEST_IGNORE) \
		--cov=training.shared_config \
		--cov=training.multilingual_safety_checker \
		--cov=training.clinical_safety_checker \
		--cov=training.reward_score \
		--cov-branch --cov-fail-under=95 -q \
		--cov-report=xml:coverage/safety-critical-coverage.xml \
		--cov-report=term

coverage-pilot: ## Run the pilot-module coverage gate (40%, branch).
	mkdir -p coverage
	$(PYTEST) training/tests $(PYTEST_IGNORE) \
		--cov=training.pixelated_production_pilot \
		--cov-branch --cov-fail-under=40 -q \
		--cov-report=xml:coverage/pilot-coverage.xml \
		--cov-report=term

coverage-all: coverage-safety coverage-pilot ## Run every coverage gate CI runs.

lint: ## Run ruff on the test tree.
	uv run ruff check training/tests/

clean: ## Remove local coverage artifacts.
	rm -rf .coverage coverage/
