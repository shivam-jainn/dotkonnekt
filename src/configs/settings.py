from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_name: str = "dotkonnekt"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/api/v1"

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "dtx_database"
    postgres_user: str = "dtx_user"
    postgres_password: str = "dtx_password"
    database_url: str | None = None
    pool_min_size: int = 2
    pool_max_size: int = 10

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"

    # RabbitMQ
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"
    rabbitmq_vhost: str = "/"
    rabbitmq_queue: str = "ingestion"
    storage_queue: str = "storage"

    # Storage (batched upsert)
    storage_batch_size: int = 100

    # Storage
    storage_provider: str = "minio"
    storage_endpoint_url: str = "http://localhost:9000"
    storage_access_key_id: str = "minioadmin"
    storage_secret_access_key: str = "minioadmin"
    storage_region: str = "us-east-1"
    storage_bucket_name: str = "dotkonnekt"
    storage_use_ssl: bool = False

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
