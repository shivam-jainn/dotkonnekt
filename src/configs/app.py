from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    APP_NAME: str = "dotkonnekt"
    APP_VERSION: str = Field(default="0.1.0")

    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

settings = AppSettings()