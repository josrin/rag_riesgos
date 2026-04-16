import dataclasses
import sqlite3
from datetime import datetime, timedelta

import pytest

from src import logger_db


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Aisla queries.db en un tmp_path y restaura automaticamente."""
    new_settings = dataclasses.replace(
        logger_db.settings,
        queries_db=tmp_path / "queries_test.db",
        log_retention_days=30,
    )
    monkeypatch.setattr(logger_db, "settings", new_settings)
    return new_settings.queries_db


class TestSchema:
    def test_db_created_on_first_access(self, isolated_db):
        assert not isolated_db.exists()
        logger_db.log_query("q", "a", [], 1)
        assert isolated_db.exists()

    def test_warnings_column_added_on_legacy_db(self, isolated_db, tmp_path):
        # Simula una DB preexistente sin warnings_json y verifica migracion.
        with sqlite3.connect(isolated_db) as con:
            con.executescript(
                """
                CREATE TABLE queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL, question TEXT, answer TEXT,
                    sources_json TEXT, latency_ms INTEGER,
                    llm_model TEXT, embedding_model TEXT
                );
                """
            )
        logger_db.log_query("q", "a", [], 1)
        with sqlite3.connect(isolated_db) as con:
            cols = {r[1] for r in con.execute("PRAGMA table_info(queries)")}
        assert "warnings_json" in cols


class TestLogQuery:
    def test_basic_insert(self, isolated_db):
        logger_db.log_query(
            "¿Que es VaR?",
            "Medida de riesgo.",
            [{"source": "doc.pdf", "page": 1}],
            123,
        )
        rows = logger_db.recent(limit=10)
        assert len(rows) == 1
        assert rows[0]["question"] == "¿Que es VaR?"
        assert rows[0]["answer"] == "Medida de riesgo."
        assert rows[0]["latency_ms"] == 123

    def test_utf8_preserved(self, isolated_db):
        q = "¿Qué metodología usa el límite? ñ á é"
        logger_db.log_query(q, "r", [], 1)
        rows = logger_db.recent(limit=1)
        assert rows[0]["question"] == q

    def test_sources_serialized_as_json(self, isolated_db):
        srcs = [
            {"source": "a.pdf", "page": 1, "distance": 0.12},
            {"source": "b.pdf", "page": 3, "distance": 0.34},
        ]
        logger_db.log_query("q", "a", srcs, 50)
        rows = logger_db.recent(limit=1)
        assert rows[0]["sources"] == srcs

    def test_warnings_default_empty_list(self, isolated_db):
        logger_db.log_query("q", "a", [], 1)
        rows = logger_db.recent(limit=1)
        assert rows[0]["warnings"] == []

    def test_warnings_persisted(self, isolated_db):
        w = [{"kind": "numero", "claim": "99.9%"}]
        logger_db.log_query("q", "a", [], 1, warnings=w)
        rows = logger_db.recent(limit=1)
        assert rows[0]["warnings"] == w

    def test_model_metadata_recorded(self, isolated_db):
        logger_db.log_query("q", "a", [], 1)
        with sqlite3.connect(isolated_db) as con:
            row = con.execute(
                "SELECT llm_model, embedding_model FROM queries ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == logger_db.settings.llm_model
        assert row[1] == logger_db.settings.embedding_model


class TestRecent:
    def test_returns_most_recent_first(self, isolated_db):
        # Insertamos con timestamps explicitos para evitar resolucion de 1s.
        with sqlite3.connect(isolated_db) as con:
            con.executescript(logger_db.SCHEMA)
            for i, ts in enumerate(
                ["2026-04-15T10:00:00", "2026-04-16T10:00:00", "2026-04-14T10:00:00"]
            ):
                con.execute(
                    "INSERT INTO queries(ts, question, answer, sources_json, latency_ms, warnings_json) "
                    "VALUES(?,?,?,?,?,?)",
                    (ts, f"q{i}", "a", "[]", 1, "[]"),
                )
            con.commit()
        rows = logger_db.recent(limit=10)
        timestamps = [r["ts"] for r in rows]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_limit_respected(self, isolated_db):
        for i in range(5):
            logger_db.log_query(f"q{i}", "a", [], 1)
        rows = logger_db.recent(limit=2)
        assert len(rows) == 2

    def test_empty_db_returns_empty_list(self, isolated_db):
        logger_db._ensure_db()
        assert logger_db.recent() == []


class TestPurgeOld:
    def _insert_with_ts(self, db_path, ts: str, question: str) -> int:
        with sqlite3.connect(db_path) as con:
            cur = con.execute(
                "INSERT INTO queries(ts, question, answer, sources_json, latency_ms, warnings_json) "
                "VALUES(?,?,?,?,?,?)",
                (ts, question, "a", "[]", 1, "[]"),
            )
            con.commit()
            return cur.lastrowid

    def test_removes_rows_older_than_retention(self, isolated_db):
        logger_db._ensure_db()
        old_ts = (datetime.utcnow() - timedelta(days=45)).isoformat(timespec="seconds")
        new_ts = (datetime.utcnow() - timedelta(days=5)).isoformat(timespec="seconds")
        self._insert_with_ts(isolated_db, old_ts, "antigua")
        self._insert_with_ts(isolated_db, new_ts, "reciente")

        removed = logger_db.purge_old()
        assert removed == 1

        rows = logger_db.recent(limit=10)
        assert len(rows) == 1
        assert rows[0]["question"] == "reciente"

    def test_no_rows_to_purge_returns_zero(self, isolated_db):
        logger_db._ensure_db()
        self._insert_with_ts(
            isolated_db,
            datetime.utcnow().isoformat(timespec="seconds"),
            "reciente",
        )
        assert logger_db.purge_old() == 0

    def test_boundary_exactly_at_retention_removed(self, isolated_db):
        # Registros a 30 dias 1 minuto se borran (ts < cutoff).
        logger_db._ensure_db()
        old_ts = (datetime.utcnow() - timedelta(days=30, minutes=1)).isoformat(timespec="seconds")
        self._insert_with_ts(isolated_db, old_ts, "boundary")
        assert logger_db.purge_old() == 1
