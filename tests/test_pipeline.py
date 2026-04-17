"""Tests de integracion para `src/pipeline.py`.

Mockeamos retriever, generator, decomposer, reranker y logger_db para
ejercitar la orquestacion sin Ollama ni ChromaDB reales. El foco son:
propagacion de subqueries, estructura del payload de `ask`, streaming
de `ask_stream`, y robustez de `warmup` frente a fallos del LLM.
"""
import pytest

from src import pipeline


@pytest.fixture
def mock_stack(monkeypatch):
    """Mocks de todas las dependencias externas del pipeline.

    Devuelve un dict para que cada test pueda ajustar lo que necesita.
    """
    state: dict = {
        "subqueries": ["q"],
        "pool": [
            {
                "text": "contexto relevante",
                "meta": {"source": "a.pdf", "page": 2, "section_hint": "Articulo 3"},
                "distance": 0.12,
                "rrf_score": 0.08,
            }
        ],
        "answer_text": "Respuesta.\n\nFuentes: [a.pdf, pagina 2, seccion Articulo 3]",
        "warnings": [],
        "logged": [],
    }

    def fake_decompose(q):
        return state["subqueries"]

    def fake_hybrid_multi(queries, top_k=None):
        return state["pool"]

    def fake_rerank(q, hits, top_k=None):
        return hits[: top_k or 5]

    def fake_answer(q, hits):
        return state["answer_text"]

    def fake_answer_stream(q, hits):
        for tok in state["answer_text"].split():
            yield tok + " "

    def fake_check(ans, ctx):
        return state["warnings"]

    def fake_log(q, a, srcs, lat, warns=None):
        state["logged"].append({"q": q, "latency": lat, "warnings": warns})

    monkeypatch.setattr(pipeline.query_decomposer, "decompose", fake_decompose)
    monkeypatch.setattr(pipeline.retriever, "hybrid_query_multi", fake_hybrid_multi)
    monkeypatch.setattr(pipeline.reranker, "rerank", fake_rerank)
    monkeypatch.setattr(pipeline.generator, "answer", fake_answer)
    monkeypatch.setattr(pipeline.generator, "answer_stream", fake_answer_stream)
    monkeypatch.setattr(pipeline.faithfulness, "check", fake_check)
    monkeypatch.setattr(pipeline.logger_db, "log_query", fake_log)
    monkeypatch.setattr(pipeline.logger_db, "purge_old", lambda: 0)
    return state


class TestAsk:
    def test_payload_shape(self, mock_stack):
        out = pipeline.ask("¿cual es el limite?")
        assert set(out.keys()) == {
            "answer",
            "sources",
            "latency_ms",
            "context",
            "subqueries",
            "faithfulness_warnings",
        }

    def test_sources_rounded_distance(self, mock_stack):
        mock_stack["pool"][0]["distance"] = 0.1234567
        out = pipeline.ask("q")
        assert out["sources"][0]["distance"] == 0.1235

    def test_single_query_does_not_include_original_in_fusion(self, mock_stack, monkeypatch):
        captured: dict = {}
        mock_stack["subqueries"] = ["pregunta simple"]

        def capture_multi(queries, top_k=None):
            captured["queries"] = list(queries)
            return mock_stack["pool"]

        monkeypatch.setattr(pipeline.retriever, "hybrid_query_multi", capture_multi)
        pipeline.ask("pregunta simple")
        # Una sola subquery -> queries = [subquery], sin duplicar la original.
        assert captured["queries"] == ["pregunta simple"]

    def test_multi_subquery_prepends_original(self, mock_stack, monkeypatch):
        captured: dict = {}
        mock_stack["subqueries"] = ["sub1?", "sub2?"]

        def capture_multi(queries, top_k=None):
            captured["queries"] = list(queries)
            return mock_stack["pool"]

        monkeypatch.setattr(pipeline.retriever, "hybrid_query_multi", capture_multi)
        pipeline.ask("compuesta original?")
        # Cuando hay descomposicion: original + todas las subqueries.
        assert captured["queries"] == ["compuesta original?", "sub1?", "sub2?"]

    def test_query_is_logged(self, mock_stack):
        pipeline.ask("q para historico")
        assert len(mock_stack["logged"]) == 1
        assert mock_stack["logged"][0]["q"] == "q para historico"

    def test_warnings_propagated(self, mock_stack):
        mock_stack["warnings"] = [{"kind": "numero", "claim": "99%"}]
        out = pipeline.ask("q")
        assert out["faithfulness_warnings"] == [{"kind": "numero", "claim": "99%"}]


class TestAskStream:
    def test_tokens_yielded_before_meta_filled(self, mock_stack):
        gen, meta = pipeline.ask_stream("q")
        # Hasta que no se consuma el generador, meta esta vacio.
        assert meta["answer"] == ""
        tokens = list(gen)
        assert len(tokens) > 0

    def test_meta_populated_after_consumption(self, mock_stack):
        gen, meta = pipeline.ask_stream("q")
        list(gen)
        assert meta["answer"].strip() != ""
        assert meta["latency_ms"] >= 0
        assert meta["sources"]
        assert isinstance(meta["faithfulness_warnings"], list)

    def test_stream_logs_query(self, mock_stack):
        gen, _ = pipeline.ask_stream("q-stream")
        list(gen)
        assert any(log["q"] == "q-stream" for log in mock_stack["logged"])


class TestWarmup:
    def test_warmup_catches_exceptions(self, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("ollama down")

        monkeypatch.setattr(pipeline.embeddings, "embed_one", boom)
        out = pipeline.warmup()
        assert out["ok"] is False
        assert "error" in out

    def test_warmup_success_path(self, monkeypatch):
        monkeypatch.setattr(pipeline.embeddings, "embed_one", lambda t: [0.0])

        class StubClient:
            def chat(self, **kw):
                return {"message": {"content": "ok"}}

        monkeypatch.setattr(pipeline.generator, "_client", StubClient())
        monkeypatch.setattr(
            pipeline.settings.__class__,
            "reranker_enabled",
            property(lambda self: False),
        )
        out = pipeline.warmup()
        assert out["ok"] is True
        assert "elapsed_s" in out
