"""Tests para `src/generator.py`.

No se llama al LLM real: se usan los fast-paths (`hits=[]`) y se
inspeccionan las funciones de formateo de contexto/mensaje, que son
deterministas.
"""
from src import generator


class TestFormatContext:
    def test_single_hit_structure(self):
        hits = [
            {
                "text": "El limite es 5%.",
                "meta": {"source": "a.pdf", "page": 3, "section_hint": "Articulo 2"},
            }
        ]
        out = generator._format_context(hits)
        assert "[Fragmento 1]" in out
        assert "Fuente: a.pdf" in out
        assert "Pagina: 3" in out
        assert "Seccion: Articulo 2" in out
        assert "El limite es 5%." in out

    def test_missing_section_hint_uses_placeholder(self):
        hits = [{"text": "x", "meta": {"source": "a.pdf", "page": 1}}]
        out = generator._format_context(hits)
        assert "(sin titulo)" in out

    def test_empty_section_hint_uses_placeholder(self):
        hits = [{"text": "x", "meta": {"source": "a.pdf", "page": 1, "section_hint": ""}}]
        out = generator._format_context(hits)
        assert "(sin titulo)" in out

    def test_multiple_hits_separated(self):
        hits = [
            {"text": "t1", "meta": {"source": "a", "page": 1, "section_hint": "s1"}},
            {"text": "t2", "meta": {"source": "b", "page": 2, "section_hint": "s2"}},
        ]
        out = generator._format_context(hits)
        assert "[Fragmento 1]" in out
        assert "[Fragmento 2]" in out
        # Separador entre bloques para que el LLM no concatene fragmentos.
        assert "============================" in out

    def test_preserves_hit_order(self):
        hits = [
            {"text": "primero", "meta": {"source": "a", "page": 1, "section_hint": "s"}},
            {"text": "segundo", "meta": {"source": "b", "page": 2, "section_hint": "s"}},
        ]
        out = generator._format_context(hits)
        assert out.index("primero") < out.index("segundo")


class TestBuildMessages:
    def test_has_system_and_user_roles(self):
        msgs = generator._build_messages("q", [])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_prompt_enforces_sources_section(self):
        assert "Fuentes:" in generator.SYSTEM_PROMPT

    def test_user_prompt_includes_question(self):
        msgs = generator._build_messages("¿cual es el VaR?", [])
        assert "¿cual es el VaR?" in msgs[1]["content"]


class TestAnswerFastPaths:
    def test_answer_no_hits_returns_no_match_message(self):
        out = generator.answer("q", [])
        assert "No encuentro" in out

    def test_answer_stream_no_hits_yields_no_match_message(self):
        out = list(generator.answer_stream("q", []))
        assert out == ["No encuentro esta informacion en los documentos indexados."]
