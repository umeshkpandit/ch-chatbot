"""Configuration: environment variables first, then Streamlit secrets."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def _get(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value:
        return value
    try:  # Streamlit secrets (Community Cloud); absent when running without secrets.toml
        import streamlit as st

        return str(st.secrets.get(name, default))
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    app_access_key: str
    cv_path: Path
    embedding_model: str
    chat_model: str
    fallback_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_score: float
    max_questions: int
    show_admin_tools: bool
    max_daily_questions: int
    ollama_model: str
    ollama_url: str

    def missing(self) -> list[str]:
        out = []
        if not self.gemini_api_key:
            out.append("GEMINI_API_KEY")
        if not self.app_access_key:
            out.append("APP_ACCESS_KEY")
        return out


def load_settings() -> Settings:
    base = Path(__file__).parent
    return Settings(
        gemini_api_key=_get("GEMINI_API_KEY"),
        app_access_key=_get("APP_ACCESS_KEY"),
        cv_path=base / _get("CV_PATH", "cv.pdf"),
        embedding_model=_get("EMBEDDING_MODEL", "gemini-embedding-001"),
        chat_model=_get("CHAT_MODEL", "gemini-3.1-flash-lite"),
        fallback_model=_get("CHAT_FALLBACK_MODEL", "gemini-3-flash-preview"),
        chunk_size=int(_get("CHUNK_SIZE", "700")),
        chunk_overlap=int(_get("CHUNK_OVERLAP", "120")),
        top_k=int(_get("TOP_K", "4")),
        min_score=float(_get("MIN_SCORE", "0.0")),
        max_questions=int(_get("MAX_QUESTIONS_PER_USER", "10")),
        max_daily_questions=int(_get("MAX_DAILY_QUESTIONS", "200")),
        ollama_model=_get("OLLAMA_MODEL"),
        ollama_url=_get("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        show_admin_tools=_get("SHOW_ADMIN_TOOLS").lower() in ("1", "true", "yes"),
    )
