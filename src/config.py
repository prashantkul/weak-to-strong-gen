"""
Configuration management for both local development and Colab
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration for the weak-to-strong ICL experiment"""

    openrouter_api_key: str
    openrouter_api_key_backup: str
    weak_model: str = "meta-llama/llama-3.1-8b-instruct"
    strong_model: str = "meta-llama/llama-3.1-405b-instruct"
    num_few_shot_examples: int = 5
    max_parallel_requests: int = 50
    temperature: float = 0.0
    max_tokens: int = 500
    cache_dir: Path = Field(default_factory=lambda: Path("./cache"))

    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> "Config":
        """
        Load configuration from .env file (local) or environment variables (Colab)

        Args:
            env_path: Path to .env file. If None, tries to load from current directory
        """
        # Try to load from .env file if it exists
        try:
            from dotenv import load_dotenv
            if env_path:
                load_dotenv(env_path)
            else:
                load_dotenv()
        except ImportError:
            pass  # dotenv not available, will use env vars directly

        # Try to get from Google Colab secrets if available
        try:
            from google.colab import userdata
            openrouter_key = userdata.get('OPENROUTER_API_KEY')
            openrouter_backup = userdata.get('OPENROUTER_API_KEY_BACKUP')
        except (ImportError, Exception):
            # Not in Colab or secrets not set, use env vars
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            openrouter_backup = os.getenv("OPENROUTER_API_KEY_BACKUP")

        if not openrouter_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. Please set it in .env file or Colab secrets."
            )

        return cls(
            openrouter_api_key=openrouter_key,
            openrouter_api_key_backup=openrouter_backup or openrouter_key,
            weak_model=os.getenv("WEAK_MODEL", "meta-llama/llama-3.1-8b-instruct"),
            strong_model=os.getenv("STRONG_MODEL", "meta-llama/llama-3.1-405b-instruct"),
            num_few_shot_examples=int(os.getenv("NUM_FEW_SHOT_EXAMPLES", "5")),
            max_parallel_requests=int(os.getenv("MAX_PARALLEL_REQUESTS", "50")),
            temperature=float(os.getenv("TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "500")),
            cache_dir=Path(os.getenv("CACHE_DIR", "./cache")),
        )

    def setup_environment(self):
        """Set up environment variables for the API"""
        os.environ["OPENAI_API_KEY"] = "dummy"  # Required by safety-tooling
        os.environ["OPENROUTER_API_KEY"] = self.openrouter_api_key

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
