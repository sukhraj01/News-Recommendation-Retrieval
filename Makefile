.PHONY: help install test test-unit test-integration test-reproducibility format lint data fixtures clean-data clean

help:
	@echo "Assignment 1: Lexical & Semantic Retrieval"
	@echo ""
	@echo "Commands:"
	@echo "  make install              Install dependencies"
	@echo "  make data                 Build the feature store from raw files"
	@echo "  make fixtures             (Re)generate tests/fixtures/*.zip — not committed, run automatically by test/test-unit"
	@echo "  make test                 Run all tests"
	@echo "  make test-unit            Unit tests only"
	@echo "  make test-integration     Integration tests"
	@echo "  make test-reproducibility Reproducibility tests"
	@echo "  make format               Format code"
	@echo "  make lint                 Lint code"
	@echo "  make clean-data           Remove the built feature store"
	@echo "  make clean                Remove build artifacts"

install:
	poetry install

data:
	poetry run python scripts/build_feature_store.py

# tests/fixtures/*.zip are gitignored (Q8: no *.zip in git) but several unit
# tests read them directly (test_ebnerd_loader.py, test_mind_loader.py,
# test_mind_format.py, test_ebnerd_format.py) — regenerated here instead of
# committed. Unconditional (not file-timestamp-gated) because this repo's
# `make` is GNU Make 3.81 (macOS default, confirmed via `make --version`),
# which doesn't support the grouped multi-target (`&:`) rule needed to make
# three output files one real prerequisite cleanly; generation is <1s and
# deterministic, so re-running it on every `make test`/`test-unit` is cheap.
fixtures:
	poetry run python scripts/generate_test_fixtures.py

test: fixtures
	poetry run pytest tests/ -v

test-unit: fixtures
	poetry run pytest tests/unit/ -v

test-integration:
	poetry run pytest tests/integration/ -v

test-reproducibility:
	poetry run pytest tests/reproducibility/ -v

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

lint:
	poetry run flake8 src/ tests/ --max-line-length=100

clean-data:
	rm -rf data/processed data/interim

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/

.DEFAULT_GOAL := help
