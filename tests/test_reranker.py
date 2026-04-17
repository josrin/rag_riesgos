"""Tests para `src/reranker.py`.

No se carga el cross-encoder real (pesa 2 GB): ejercitamos las ramas
deterministas (`_minmax`, bypass cuando esta desactivado, pool corto)
y mockeamos el modelo cuando hace falta verificar el blending.
"""
import dataclasses

import numpy as np
import pytest

from src import reranker


def _hit(text: str, rrf: float) -> dict:
    return {"text": text, "meta": {"source": "a.pdf", "page": 1}, "rrf_score": rrf}


class TestMinmax:
    def test_normalizes_to_unit_interval(self):
        out = reranker._minmax([1.0, 2.0, 3.0])
        assert out[0] == 0.0
        assert out[-1] == 1.0
        assert 0.0 < out[1] < 1.0

    def test_all_equal_returns_midpoint(self):
        # Fallback para evitar division por cero; 0.5 deja a todos los
        # elementos empatados en el blended score final.
        assert reranker._minmax([2.5, 2.5, 2.5]) == [0.5, 0.5, 0.5]

    def test_single_element_returns_midpoint(self):
        assert reranker._minmax([7.0]) == [0.5]


class TestRerankDisabled:
    @pytest.fixture
    def disabled(self, monkeypatch):
        patched = dataclasses.replace(reranker.settings, reranker_enabled=False, top_k=3)
        monkeypatch.setattr(reranker, "settings", patched)

    def test_returns_empty_when_no_hits(self, disabled):
        assert reranker.rerank("q", []) == []

    def test_preserves_input_order_and_truncates(self, disabled):
        hits = [_hit(f"t{i}", 1.0 - i * 0.1) for i in range(5)]
        out = reranker.rerank("q", hits)
        assert len(out) == 3
        assert [h["text"] for h in out] == ["t0", "t1", "t2"]


class TestRerankEnabled:
    @pytest.fixture
    def enabled(self, monkeypatch):
        patched = dataclasses.replace(
            reranker.settings,
            reranker_enabled=True,
            reranker_alpha=1.0,
            top_k=2,
        )
        monkeypatch.setattr(reranker, "settings", patched)

    def test_pool_smaller_than_k_is_not_rerun(self, enabled):
        # Si len(hits) <= k, no tiene sentido llamar al modelo: devolvemos
        # el pool tal cual y nos ahorramos la inferencia.
        hits = [_hit("solo", 0.5)]
        out = reranker.rerank("q", hits)
        assert out == hits

    def test_alpha_1_uses_cross_encoder_order(self, enabled, monkeypatch):
        class StubModel:
            def predict(self, pairs):
                # Invertimos el orden: el primer par obtiene el score mas alto.
                # Devolvemos numpy array porque el codigo llama `.tolist()`.
                return np.array([0.1 * (i + 1) for i in range(len(pairs))][::-1])

        monkeypatch.setattr(reranker, "_get_model", lambda: StubModel())
        hits = [_hit("a", 0.9), _hit("b", 0.5), _hit("c", 0.1)]
        out = reranker.rerank("q", hits)
        # Alpha=1, el reranker manda. Primer score (0.3) va al hit "a",
        # segundo (0.2) al "b", tercero (0.1) al "c". Orden: a, b, c.
        assert [h["text"] for h in out] == ["a", "b"]

    def test_alpha_0_preserves_rrf_order(self, enabled, monkeypatch):
        class StubModel:
            # Orden inverso al RRF, pero con alpha=0 no debe influir.
            def predict(self, pairs):
                return np.array([float(i) for i in range(len(pairs))])

        patched = dataclasses.replace(reranker.settings, reranker_alpha=0.0)
        monkeypatch.setattr(reranker, "settings", patched)
        monkeypatch.setattr(reranker, "_get_model", lambda: StubModel())

        hits = [_hit("mejor-rrf", 0.9), _hit("medio", 0.5), _hit("peor-rrf", 0.1)]
        out = reranker.rerank("q", hits)
        assert [h["text"] for h in out] == ["mejor-rrf", "medio"]

    def test_scores_attached_to_results(self, enabled, monkeypatch):
        class StubModel:
            def predict(self, pairs):
                return np.array([0.5] * len(pairs))

        monkeypatch.setattr(reranker, "_get_model", lambda: StubModel())
        hits = [_hit("a", 0.9), _hit("b", 0.5), _hit("c", 0.1)]
        out = reranker.rerank("q", hits)
        for h in out:
            assert "rerank_score" in h
            assert "blended_score" in h
