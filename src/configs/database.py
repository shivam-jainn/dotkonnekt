from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class DatabaseSettings(BaseSettings):
    """
    Database settings for the application.
    """

    # Postgres
    DB_HOST: str = Field(default="localhost")
    DB_PORT: int = Field(default=5432)
    DB_NAME: str = Field(default="mydatabase")
    DB_USER: str = Field(default="myuser")
    DB_PASSWORD: str = Field(default="mypassword")

    # Vector
    VECTOR_DB_HOST: str = Field(default="localhost")
    VECTOR_DB_PORT: int = Field(default=8000)
    VECTOR_DB_NAME: str = Field(default="myvector database")
    VECTOR_DB_USER: str = Field(default="myvector user")
    VECTOR_DB_PASSWORD: str = Field(default="myvector password")


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

settings = DatabaseSettings()