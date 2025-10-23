from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    EMBED_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-4o-mini"
    DATA_DIR: str = "./data"
    INDEX_DIR: str = "./indexes"
    MAX_CONTEXT_CHUNKS: int = 6
    ALLOWED_EMAIL_DOMAINS: List[str] = []

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def allowed_domains(self):
        raw = os.getenv("ALLOWED_EMAIL_DOMAINS", "")
        return [d.strip().lower() for d in raw.split(",") if d.strip()]

settings = Settings()
