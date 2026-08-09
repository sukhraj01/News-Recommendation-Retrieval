.PHONY: help install test test-unit test-integration test-reproducibility format lint data clean

help:
	@echo "Assignment 1: Lexical & Semantic Retrieval"
	@echo ""
	@echo "Commands:"
	@echo "  make install              Install dependencies"
	@echo "  make test                 Run all tests"
	@echo "  make test-unit            Unit tests only"
	@echo "  make test-integration     Integration tests"
	@echo "  make format               Format code"
	@echo "  make lint                 Lint code"
	@echo "  make clean                Remove artifacts"

install:
	poetry install

test:
	poetry run pytest tests/ -v

test-unit:
	poetry run pytest tests/unit/ -v

test-integration:
	poetry run pytest tests/integration/ -v

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

lint:
	poetry run flake8 src/ tests/ --max-line-length=100

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/

.DEFAULT_GOAL := help
