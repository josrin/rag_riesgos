"""Tests para `src/vectorstore.py` con ChromaDB real en `tmp_path`.

ChromaDB corre en proceso (sin servicio externo), asi que podemos
ejercitar upsert/query extremo-a-extremo. Aislamos cada test con un
directorio temporal para no contaminar `data/chroma` del usuario.
"""
import dataclasses

import pytest

from src import vectorstore


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    new_settings = dataclasses.replace(
        vectorstore.settings,
        chroma_dir=tmp_path / "chroma_test",
        collection_name="test_collection",
    )
    monkeypatch.setattr(vectorstore, "settings", new_settings)
    return new_settings


def _chunk(source: str, page: int, idx: int, text: str, section: str = "") -> dict:
    return {
        "source": source,
        "page": page,
        "chunk_index": idx,
        "text": text,
        "extraction": "native",
        "section_hint": section,
    }


class TestGetCollection:
    def test_creates_directory(self, isolated_store):
        # La primera llamada debe materializar el directorio persistente.
        assert not isolated_store.chroma_dir.exists()
        coll = vectorstore.get_collection()
        assert coll is not None
        assert isolated_store.chroma_dir.exists()

    def test_reset_wipes_existing_collection(self, isolated_store):
        emb = [[0.1, 0.2, 0.3]]
        vectorstore.upsert_chunks([_chunk("a.pdf", 1, 0, "hola")], emb)
        assert vectorstore.count() == 1
        vectorstore.get_collection(reset=True)
        assert vectorstore.count() == 0


class TestUpsert:
    def test_upsert_and_count(self, isolated_store):
        vectorstore.get_collection(reset=True)
        chunks = [
            _chunk("a.pdf", 1, 0, "primer chunk"),
            _chunk("a.pdf", 1, 1, "segundo chunk"),
        ]
        vectorstore.upsert_chunks(chunks, [[0.0, 1.0], [1.0, 0.0]])
        assert vectorstore.count() == 2

    def test_upsert_is_idempotent_on_same_id(self, isolated_store):
        vectorstore.get_collection(reset=True)
        chunk = _chunk("a.pdf", 1, 0, "original")
        vectorstore.upsert_chunks([chunk], [[0.0, 1.0]])
        # El mismo (source, page, chunk_index) debe sobreescribir, no
        # duplicar: el id se construye con esos tres campos.
        updated = _chunk("a.pdf", 1, 0, "actualizado")
        vectorstore.upsert_chunks([updated], [[0.1, 0.9]])
        assert vectorstore.count() == 1

    def test_is_table_metadata_persisted(self, isolated_store):
        vectorstore.get_collection(reset=True)
        chunk = _chunk("t.pdf", 2, 0, "| a | b |")
        chunk["is_table"] = True
        vectorstore.upsert_chunks([chunk], [[0.5, 0.5]])
        hits = vectorstore.query([0.5, 0.5], top_k=1)
        assert hits[0]["meta"]["is_table"] is True


class TestQuery:
    def test_closest_vector_ranked_first(self, isolated_store):
        vectorstore.get_collection(reset=True)
        chunks = [
            _chunk("a.pdf", 1, 0, "lejos"),
            _chunk("a.pdf", 1, 1, "cerca"),
        ]
        # Vectores ortogonales: la query (1,0) debe coincidir con el 1er chunk.
        vectorstore.upsert_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])
        hits = vectorstore.query([1.0, 0.0], top_k=2)
        assert hits[0]["text"] == "lejos"

    def test_returns_distance(self, isolated_store):
        vectorstore.get_collection(reset=True)
        vectorstore.upsert_chunks(
            [_chunk("a.pdf", 1, 0, "unico")], [[1.0, 0.0]]
        )
        hits = vectorstore.query([1.0, 0.0], top_k=1)
        assert "distance" in hits[0]
        assert isinstance(hits[0]["distance"], float)

    def test_query_respects_top_k(self, isolated_store):
        vectorstore.get_collection(reset=True)
        chunks = [_chunk("a.pdf", 1, i, f"t{i}") for i in range(5)]
        embs = [[float(i), 1.0] for i in range(5)]
        vectorstore.upsert_chunks(chunks, embs)
        hits = vectorstore.query([0.0, 1.0], top_k=2)
        assert len(hits) == 2
