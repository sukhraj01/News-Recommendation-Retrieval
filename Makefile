.PHONY: help install test test-unit test-integration test-reproducibility format lint data clean-data clean

help:
	@echo "Assignment 1: Lexical & Semantic Retrieval"
	@echo ""
	@echo "Commands:"
	@echo "  make install              Install dependencies"
	@echo "  make data                 Build the feature store from raw files"
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

test:
	poetry run pytest tests/ -v

test-unit:
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
