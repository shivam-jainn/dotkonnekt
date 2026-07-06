from pydantic_settings import BaseSettings, SettingsConfigDict

class DatabaseSettings(BaseSettings):
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "dotkonnekt"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DB_",
        extra="ignore",
    )