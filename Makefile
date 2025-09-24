# Makefile for Document RAG English Study

.PHONY: help install install-dev test lint format clean run setup

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install production dependencies using uv"
	@echo "  install-dev - Install development dependencies using uv"
	@echo "  test        - Run tests with pytest"
	@echo "  lint        - Run linting with flake8 and mypy"
	@echo "  format      - Format code with black and isort"
	@echo "  clean       - Clean up generated files"
	@echo "  run         - Run the CLI application"
	@echo "  setup       - Initial project setup"

# Install production dependencies
install:
	uv sync --no-dev

# Install development dependencies
install-dev:
	uv sync

# Run tests
test:
	uv run pytest

# Run linting
lint:
	uv run flake8 src/ tests/
	uv run mypy src/

# Format code
format:
	uv run black src/ tests/
	uv run isort src/ tests/

# Clean up generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

# Run the CLI application
run:
	uv run python src/document_rag_english_study/cli/main.py

# Initial project setup
setup: install-dev
	@echo "Creating necessary directories..."
	@mkdir -p data/vector_db data/sessions data/cache logs
	@echo "Copying environment template..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@echo "Setup complete! Edit .env file with your API keys."