.PHONY: install infra-up infra-down dev worker storage-worker langgraph-worker workers-all test test-unit test-integration test-upload-api test-all lint clean migrate migrate-autogenerate migrate-check migrate-stamp

dev:
	uv run uvicorn src.server:app --reload

worker:
	uv run python -m src.worker.run --worker ingestion

storage-worker:
	uv run python -m src.worker.run --worker storage

langgraph-worker:
	uv run python -m src.worker.run --worker langgraph

## Run all three workers in parallel (foreground, Ctrl-C stops all)
workers-all:
	@echo "Starting ingestion, langgraph, and storage workers..."
	@trap 'kill 0' SIGINT; \
		uv run python -m src.worker.run --worker ingestion & \
		uv run python -m src.worker.run --worker langgraph & \
		uv run python -m src.worker.run --worker storage & \
		wait

install:
	uv sync --group dev

infra-up:
	docker compose up -d

infra-down:
	docker compose down

reset-volumes:
	docker compose down -v

test:
	uv run pytest -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v -m integration

test-upload-api:
	uv run pytest tests/integration/test_upload_api.py -v

test-all: infra-up
	@sleep 3
	uv run pytest -v

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

migrate:
	PYTHONPATH=. uv run alembic upgrade head

migrate-autogenerate:
	PYTHONPATH=. uv run alembic revision --autogenerate -m "$(message)"

migrate-check:
	PYTHONPATH=. uv run alembic check

migrate-stamp:
	PYTHONPATH=. uv run alembic stamp $(revision)

clean:
	rm -rf .pytest_cache
	rm -rf tests/.pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
