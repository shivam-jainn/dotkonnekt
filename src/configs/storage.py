from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageSettings(BaseSettings):
    provider: str = "minio"

    endpoint_url: str = "http://localhost:9000"
    access_key_id: str = "minioadmin"
    secret_access_key: str = "minioadmin"
    region: str = "us-east-1"
    bucket_name: str = "dotkonnekt"
    use_ssl: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="STORAGE_",
        extra="ignore",
    )
