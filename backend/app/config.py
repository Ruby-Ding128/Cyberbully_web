from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
# 系统环境变量优先；其次读取 backend/.env，再尝试项目根目录 .env。
load_dotenv(ROOT / "backend" / ".env", override=False)
load_dotenv(ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path(
        os.getenv(
            "CYBERBULLYING_MODEL_DIR",
            ROOT / "outputs" / "distilbert_multilabel" / "final_model",
        )
    )
    llm_api_key: str | None = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    llm_base_url: str | None = os.getenv("LLM_BASE_URL")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
    max_model_length: int = int(os.getenv("MAX_MODEL_LENGTH", "256"))
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    auth_db_path: Path = Path(
        os.getenv("AUTH_DB_PATH", ROOT / "backend" / "data" / "users.db")
    )
    jwt_secret: str = os.getenv("JWT_SECRET", "development-only-change-this-secret")
    jwt_expire_minutes: int = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))


settings = Settings()
