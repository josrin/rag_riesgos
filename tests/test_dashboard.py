"""Tests para las funciones puras de `src/dashboard.py`.

El render de Streamlit no se testea (requiere runtime de la app). Nos
centramos en las agregaciones que alimentan a los widgets: frecuencias,
top preguntas y normalizacion del detector 'no encuentro'.
"""
from datetime import datetime, timezone

import pandas as pd

from src import dashboard


class TestNormalize:
    def test_removes_accents(self):
        assert dashboard._normalize("Información") == "informacion"

    def test_lowercases(self):
        assert dashboard._normalize("NO encUEntro") == "no encuentro"

    def test_no_match_marker_triggers_detection(self):
        # El marcador se usa para contar respuestas sin contexto. Cualquier
        # variacion razonable de acentos/mayusculas debe coincidir.
        for variant in (
            "No encuentro esta información en los documentos",
            "no encuentro esta informacion",
            "NO ENCUENTRO ESTA INFORMACIÓN",
        ):
            norm = dashboard._normalize(variant)
            assert dashboard._NO_MATCH_MARKER in norm


class TestSourceFrequencies:
    def test_empty_dataframe_returns_empty(self):
        out = dashboard._source_frequencies(pd.DataFrame())
        assert out.empty
        assert list(out.columns) == ["source", "citas"]

    def test_counts_citations_across_rows(self):
        df = pd.DataFrame(
            {
                "sources": [
                    [{"source": "a.pdf"}, {"source": "b.pdf"}],
                    [{"source": "a.pdf"}],
                    [{"source": "c.pdf"}, {"source": "a.pdf"}],
                ]
            }
        )
        out = dashboard._source_frequencies(df)
        counts = dict(zip(out["source"], out["citas"]))
        assert counts == {"a.pdf": 3, "b.pdf": 1, "c.pdf": 1}

    def test_sorted_descending(self):
        df = pd.DataFrame(
            {
                "sources": [
                    [{"source": "raro"}],
                    [{"source": "comun"}, {"source": "comun"}, {"source": "comun"}],
                    [{"source": "medio"}, {"source": "medio"}],
                ]
            }
        )
        out = dashboard._source_frequencies(df)
        assert list(out["source"]) == ["comun", "medio", "raro"]


class TestTopQuestions:
    def _df(self, rows):
        return pd.DataFrame(rows)

    def test_empty_returns_empty_with_expected_cols(self):
        out = dashboard._top_questions(pd.DataFrame())
        assert out.empty
        assert set(out.columns) == {"question", "veces", "no_match"}

    def test_counts_and_medians(self):
        df = self._df(
            [
                {"id": 1, "question": "q1", "no_match": False, "latency_ms": 100},
                {"id": 2, "question": "q1", "no_match": True, "latency_ms": 200},
                {"id": 3, "question": "q2", "no_match": False, "latency_ms": 50},
            ]
        )
        out = dashboard._top_questions(df, n=10)
        row_q1 = out[out["question"] == "q1"].iloc[0]
        assert row_q1["veces"] == 2
        assert row_q1["no_match"] == 1
        assert row_q1["p50"] == 150.0  # mediana de [100, 200]

    def test_limit_applied(self):
        rows = [
            {"id": i, "question": f"q{i % 3}", "no_match": False, "latency_ms": 10}
            for i in range(15)
        ]
        out = dashboard._top_questions(self._df(rows), n=2)
        assert len(out) == 2


class TestLoadDataframe:
    def test_empty_when_no_rows(self, monkeypatch):
        monkeypatch.setattr(dashboard.logger_db, "all_queries", lambda: [])
        out = dashboard._load_dataframe()
        assert out.empty

    def test_enriches_rows_with_derived_columns(self, monkeypatch):
        rows = [
            {
                "id": 1,
                "ts": datetime.now(timezone.utc).isoformat(),
                "question": "q1",
                "answer": "Sin contexto. No encuentro esta informacion en los documentos.",
                "sources": [{"source": "a.pdf", "page": 1}],
                "latency_ms": 100,
                "warnings": [{"kind": "numero"}],
                "llm_model": "llama3",
                "embedding_model": "bge-m3",
            },
            {
                "id": 2,
                "ts": datetime.now(timezone.utc).isoformat(),
                "question": "q2",
                "answer": "Respuesta normal.",
                "sources": [],
                "latency_ms": 50,
                "warnings": [],
                "llm_model": "llama3",
                "embedding_model": "bge-m3",
            },
        ]
        monkeypatch.setattr(dashboard.logger_db, "all_queries", lambda: rows)
        df = dashboard._load_dataframe()
        assert list(df["no_match"]) == [True, False]
        assert list(df["warnings_count"]) == [1, 0]
        assert list(df["sources_count"]) == [1, 0]
        assert "date" in df.columns
