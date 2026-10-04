from __future__ import annotations
from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["openai-compatible", "mock"] = "openai-compatible"
    village_model: str = "Qwen/Qwen2.5-7B-Instruct"
    openai_base_url: str = "http://localhost:8000/v1"
    openai_api_key: str = "local"  # client requires one
    temperature: float = 0.7  
    max_tokens: int = 512
    request_timeout: float = 120.0
    max_retries: int = 6  # transient errors (429, 5xx) are retried with backoff by the client
    max_concurrency: int = 8  
    structured_output: Literal["json_schema", "json_object", "none"] = "json_schema"
    openrouter_require_parameters: bool = True
    openrouter_providers: str = ""
    runs_dir: str = "runs"

@lru_cache
def get_settings() -> Settings:
    return Settings()
