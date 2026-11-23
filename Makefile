.PHONY: help install test lint format run docker-build docker-run docker-stop cam-test cam-run

help:
	@echo "GeoMind - Development Commands"
	@echo ""
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run all tests"
	@echo "  make test-cam     - Run camera pipeline tests"
	@echo "  make lint         - Run linting"
	@echo "  make format       - Format code"
	@echo "  make run          - Run application locally"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run with docker-compose"
	@echo "  make docker-stop  - Stop docker-compose"
	@echo "  make docker-logs  - View docker-compose logs"
	@echo "  make clean        - Clean build artifacts"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov ruff mypy

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-cam:
	pytest tests/test_cameras.py tests/test_cameras_db.py tests/test_cameras_api.py tests/test_cameras_scheduler.py -v

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

run:
	python -m src.geomind.main

cam-run:
	GEOMIND_CAM_CONFIG_PATH=cameras.yaml python -m src.geomind.main

docker-build:
	docker build -t geomind:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true