.PHONY: install test bench run lint clean

PYTHON := python3
PKG := adversarial_prompt_suite

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src/$(PKG) --cov-report=term-missing --cov-report=xml

bench:
	$(PYTHON) benchmarks/bench_evaluator.py

run:
	$(PYTHON) examples/quickstart.py

lint:
	$(PYTHON) -m ruff check src/ tests/ examples/ benchmarks/ || true
	$(PYTHON) -m mypy src/ --ignore-missing-imports || true

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f coverage.xml .coverage report.json report.md
	@echo "Clean complete."
