
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
import os
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class Settings:
    openrouter_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_url: str = "https://openrouter.ai/api/v1"

    @property
    @lru_cache
    def client(self) -> OpenAI:
        return OpenAI(base_url=self.openrouter_url, api_key=self.openrouter_key)

@lru_cache
def get_settings() -> Settings:
    return Settings()
