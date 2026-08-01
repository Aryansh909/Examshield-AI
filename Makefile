# Developer Ergonomics
.PHONY: help setup run test lint clean benchmark

help: ## Display available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Create virtualenv and install pinned dependencies
	python -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	cp -n .env.example .env || true
	./venv/bin/python scripts/setup.py

run: ## Launch the proctoring server
	./venv/bin/python app.py

test: ## Execute test suite
	./venv/bin/pytest tests/ -v

lint: ## Execute static analysis and PEP8 checks
	./venv/bin/flake8 src/ app.py config.py db.py --max-line-length=120

clean: ## Remove bytecode, caches, and temporary build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
