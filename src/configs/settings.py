from .app import AppSettings
from .database import DatabaseSettings


class Settings:
    def __init__(self) -> None:
        self.app = AppSettings()
        self.db = DatabaseSettings()


settings = Settings()