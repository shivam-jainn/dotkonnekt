from .app import AppSettings
from .database import DatabaseSettings
from .storage import StorageSettings


class Settings:
    def __init__(self) -> None:
        self.app = AppSettings()
        self.db = DatabaseSettings()
        self.storage = StorageSettings()


settings = Settings()