# ExamShield AI — Developer Makefile
# Usage: make <target>

.PHONY: help setup run test clean lint

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Create virtualenv, install dependencies, initialise directories
	python -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	cp -n .env.example .env || true
	python scripts/setup.py
	@echo "\n[OK] Setup complete. Run 'make run' to start."

run: ## Start the ExamShield AI server
	./venv/bin/python app.py

test: ## Run the test suite
	./venv/bin/pytest tests/ -v

lint: ## Lint the codebase with flake8
	./venv/bin/flake8 app.py camera.py config.py db.py train_models.py --max-line-length=120

clean: ## Remove Python cache and build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
	@echo "[OK] Clean complete"

models: ## Download required model files (MediaPipe tasks)
	bash scripts/download_models.sh
