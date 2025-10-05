from pydantic import BaseSettings, Field
from typing import Optional
import os

class Settings(BaseSettings):
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    chroma_persist_dir: str = Field(default=os.getenv("CHROMA_PERSIST_DIR", ".chroma"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
