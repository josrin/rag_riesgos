"""Log de consultas en SQLite con retencion configurable (default 30 dias)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Sequence

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    latency_ms INTEGER,
    llm_model TEXT,
    embedding_model TEXT,
    warnings_json TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_queries_ts ON queries(ts);
"""


def _ensure_db() -> Path:
    """Crea la DB y migra columna `warnings_json` si falta (idempotente)."""
    settings.queries_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.queries_db) as con:
        con.executescript(SCHEMA)
        cols = {r[1] for r in con.execute("PRAGMA table_info(queries)")}
        if "warnings_json" not in cols:
            con.execute("ALTER TABLE queries ADD COLUMN warnings_json TEXT DEFAULT '[]'")
    return settings.queries_db


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    """Context manager que garantiza schema + commit/close + row_factory."""
    _ensure_db()
    con = sqlite3.connect(settings.queries_db)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def log_query(
    question: str,
    answer: str,
    sources: Sequence[dict],
    latency_ms: int,
    warnings: Sequence[dict] | None = None,
) -> None:
    """Persiste una consulta con su respuesta, fuentes, latencia y warnings."""
    with _conn() as con:
        con.execute(
            "INSERT INTO queries(ts, question, answer, sources_json, latency_ms, "
            "llm_model, embedding_model, warnings_json) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                datetime.utcnow().isoformat(timespec="seconds"),
                question,
                answer,
                json.dumps(list(sources), ensure_ascii=False),
                int(latency_ms),
                settings.llm_model,
                settings.embedding_model,
                json.dumps(list(warnings or []), ensure_ascii=False),
            ),
        )


def purge_old() -> int:
    """Borra registros con mas de LOG_RETENTION_DAYS de antiguedad. Devuelve filas eliminadas."""
    cutoff = (datetime.utcnow() - timedelta(days=settings.log_retention_days)).isoformat(timespec="seconds")
    with _conn() as con:
        cur = con.execute("DELETE FROM queries WHERE ts < ?", (cutoff,))
        return cur.rowcount


def recent(limit: int = 50) -> list[dict]:
    """Devuelve las ultimas `limit` consultas, mas recientes primero."""
    with _conn() as con:
        rows = con.execute(
            "SELECT id, ts, question, answer, sources_json, latency_ms, "
            "llm_model, warnings_json FROM queries ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d.pop("sources_json"))
        d["warnings"] = json.loads(d.pop("warnings_json") or "[]")
        out.append(d)
    return out


def all_queries() -> list[dict]:
    """Devuelve todos los registros vigentes para analisis."""
    with _conn() as con:
        rows = con.execute(
            "SELECT id, ts, question, answer, sources_json, latency_ms, "
            "llm_model, embedding_model, warnings_json FROM queries "
            "ORDER BY ts DESC"
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d.pop("sources_json"))
        d["warnings"] = json.loads(d.pop("warnings_json") or "[]")
        out.append(d)
    return out
