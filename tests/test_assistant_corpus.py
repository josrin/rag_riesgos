import dataclasses

import pytest

from src.assistant import corpus_utils


@pytest.fixture
def isolated_docs(tmp_path, monkeypatch):
    """Aisla settings.docs_dir en tmp_path para listar solo archivos de prueba."""
    new_settings = dataclasses.replace(corpus_utils.settings, docs_dir=tmp_path)
    monkeypatch.setattr(corpus_utils, "settings", new_settings)
    return tmp_path


class TestListDocs:
    def test_empty_dir(self, isolated_docs):
        assert corpus_utils.list_docs() == []

    def test_pdf_txt_md_accepted(self, isolated_docs):
        for name in ("a.pdf", "b.txt", "c.md"):
            (isolated_docs / name).write_text("x", encoding="utf-8")
        assert set(corpus_utils.list_docs()) == {"a.pdf", "b.txt", "c.md"}

    def test_other_extensions_ignored(self, isolated_docs):
        (isolated_docs / "doc.txt").write_text("x", encoding="utf-8")
        (isolated_docs / "doc.docx").write_text("x", encoding="utf-8")
        (isolated_docs / "doc.xlsx").write_text("x", encoding="utf-8")
        assert corpus_utils.list_docs() == ["doc.txt"]

    def test_sorted_alphabetically(self, isolated_docs):
        for name in ("c.txt", "a.txt", "b.txt"):
            (isolated_docs / name).write_text("x", encoding="utf-8")
        assert corpus_utils.list_docs() == ["a.txt", "b.txt", "c.txt"]

    def test_subdirectories_ignored(self, isolated_docs):
        (isolated_docs / "doc.txt").write_text("x", encoding="utf-8")
        sub = isolated_docs / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("x", encoding="utf-8")
        assert corpus_utils.list_docs() == ["doc.txt"]


class TestListActaDocs:
    def test_filters_by_acta_substring(self, isolated_docs):
        for name in (
            "doc_05_acta_comite.txt",
            "doc_04_politica.txt",
            "Acta_2026.pdf",
            "ACTA_MAYUSCULA.md",
            "actas_anuales.txt",
            "otro.txt",
        ):
            (isolated_docs / name).write_text("x", encoding="utf-8")
        result = corpus_utils.list_acta_docs()
        assert "doc_05_acta_comite.txt" in result
        assert "Acta_2026.pdf" in result
        assert "ACTA_MAYUSCULA.md" in result
        assert "actas_anuales.txt" in result
        assert "doc_04_politica.txt" not in result
        assert "otro.txt" not in result

    def test_case_insensitive(self, isolated_docs):
        # En Windows el filesystem es case-insensitive, asi que usamos
        # nombres con distinto stem pero cada uno con "acta" en un casing.
        (isolated_docs / "ACTA_2026.txt").write_text("x", encoding="utf-8")
        (isolated_docs / "mi_acta_q1.txt").write_text("x", encoding="utf-8")
        (isolated_docs / "Acta_final.pdf").write_text("x", encoding="utf-8")
        result = set(corpus_utils.list_acta_docs())
        assert result == {"ACTA_2026.txt", "mi_acta_q1.txt", "Acta_final.pdf"}

    def test_no_actas_returns_empty(self, isolated_docs):
        (isolated_docs / "doc.txt").write_text("x", encoding="utf-8")
        assert corpus_utils.list_acta_docs() == []
