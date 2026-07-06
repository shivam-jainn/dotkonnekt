import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require external services (Postgres, RabbitMQ, MinIO)",
    )
