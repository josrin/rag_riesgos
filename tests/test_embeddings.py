"""Tests para `src/embeddings.py`.

Ollama se mockea: ejercitamos cache LRU, semantica del retry y el
wrapper de `embed_many`. No se pide embeddings reales — el objetivo es
validar que el cache reduce llamadas y expone stats coherentes.
"""
import pytest

from src import embeddings


@pytest.fixture(autouse=True)
def clean_cache():
    embeddings.clear_cache()
    yield
    embeddings.clear_cache()


class TestEmbedOne:
    def test_returns_list_of_floats(self, monkeypatch):
        calls = {"n": 0}

        def fake_fetch(text, model):
            calls["n"] += 1
            return (0.1, 0.2, 0.3)

        monkeypatch.setattr(embeddings, "_fetch_embedding", fake_fetch)
        embeddings._cached_embed.cache_clear()
        out = embeddings.embed_one("hola")
        assert out == [0.1, 0.2, 0.3]
        assert calls["n"] == 1

    def test_second_call_same_text_uses_cache(self, monkeypatch):
        calls = {"n": 0}

        def fake_fetch(text, model):
            calls["n"] += 1
            return (1.0, 0.0)

        monkeypatch.setattr(embeddings, "_fetch_embedding", fake_fetch)
        embeddings._cached_embed.cache_clear()

        embeddings.embed_one("misma pregunta")
        embeddings.embed_one("misma pregunta")
        assert calls["n"] == 1

    def test_different_text_bypasses_cache(self, monkeypatch):
        calls = {"n": 0}

        def fake_fetch(text, model):
            calls["n"] += 1
            return (0.0, 0.0)

        monkeypatch.setattr(embeddings, "_fetch_embedding", fake_fetch)
        embeddings._cached_embed.cache_clear()

        embeddings.embed_one("a")
        embeddings.embed_one("b")
        assert calls["n"] == 2


class TestEmbedMany:
    def test_preserves_order(self, monkeypatch):
        # El texto determina el vector; cache es transparente.
        def fake_fetch(text, model):
            return (float(len(text)),)

        monkeypatch.setattr(embeddings, "_fetch_embedding", fake_fetch)
        embeddings._cached_embed.cache_clear()

        texts = ["a", "bb", "ccc"]
        out = embeddings.embed_many(texts)
        assert out == [[1.0], [2.0], [3.0]]

    def test_cache_dedups_within_batch(self, monkeypatch):
        calls = {"n": 0}

        def fake_fetch(text, model):
            calls["n"] += 1
            return (0.0,)

        monkeypatch.setattr(embeddings, "_fetch_embedding", fake_fetch)
        embeddings._cached_embed.cache_clear()

        embeddings.embed_many(["x", "y", "x", "y", "z"])
        # x, y, z -> 3 fetches. Los duplicados se sirven del cache.
        assert calls["n"] == 3


class TestCacheStats:
    def test_hit_rate_zero_on_empty(self):
        stats = embeddings.cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["maxsize"] == 2048

    def test_hit_rate_after_cache_hit(self, monkeypatch):
        monkeypatch.setattr(embeddings, "_fetch_embedding", lambda t, m: (1.0,))
        embeddings._cached_embed.cache_clear()

        embeddings.embed_one("q")
        embeddings.embed_one("q")
        stats = embeddings.cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestClearCache:
    def test_after_clear_next_call_refetches(self, monkeypatch):
        calls = {"n": 0}

        def fake_fetch(text, model):
            calls["n"] += 1
            return (0.0,)

        monkeypatch.setattr(embeddings, "_fetch_embedding", fake_fetch)
        embeddings._cached_embed.cache_clear()

        embeddings.embed_one("q")
        embeddings.clear_cache()
        embeddings.embed_one("q")
        assert calls["n"] == 2
