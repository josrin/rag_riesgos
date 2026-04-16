from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "llama3")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    docs_dir: Path = ROOT / os.getenv("DOCS_DIR", "docs")
    chroma_dir: Path = ROOT / os.getenv("CHROMA_DIR", "data/chroma")
    queries_db: Path = ROOT / os.getenv("QUERIES_DB", "data/queries.db")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    reranker_enabled: bool = os.getenv("RERANKER_ENABLED", "false").lower() in ("true", "1", "yes")
    reranker_pool_size: int = int(os.getenv("RERANKER_POOL_SIZE", "20"))
    reranker_alpha: float = float(os.getenv("RERANKER_ALPHA", "0.5"))
    log_retention_days: int = int(os.getenv("LOG_RETENTION_DAYS", "30"))

    tesseract_cmd: str | None = os.getenv("TESSERACT_CMD") or None
    poppler_path: str | None = os.getenv("POPPLER_PATH") or None

    collection_name: str = "riesgos_corpus"


settings = Settings()
