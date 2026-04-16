from src import query_decomposer
from src.query_decomposer import _looks_like_question, decompose, is_compound


class TestIsCompound:
    def test_simple_question_is_not_compound(self):
        assert not is_compound("¿Que es el VaR?")

    def test_conjunction_plus_wh_is_compound(self):
        assert is_compound("¿Que es X y cual es Y?")

    def test_multiple_question_marks_is_compound(self):
        assert is_compound("¿Primero? ¿Segundo?")

    def test_ademas_is_compound(self):
        assert is_compound("¿Cuanto es X? Ademas, ¿que dice la norma?")

    def test_ademas_with_accent_is_compound(self):
        assert is_compound("¿Cuanto es X? Además, ¿que dice la norma?")

    def test_conjunction_without_wh_not_compound(self):
        # "y el limite" no incluye un wh-word despues de "y ".
        assert not is_compound("¿Cual es el limite y el horario?")


class TestLooksLikeQuestion:
    def test_trailing_question_mark(self):
        assert _looks_like_question("¿Cuando ocurrio el incidente?")

    def test_leading_spanish_open(self):
        assert _looks_like_question("¿Cuando")

    def test_short_string_rejected(self):
        assert not _looks_like_question("¿?")

    def test_empty_string_rejected(self):
        assert not _looks_like_question("")

    def test_statement_rejected(self):
        # El LLM a veces devuelve filler como afirmaciones; se descartan.
        assert not _looks_like_question("No hay un limite definido.")


class TestDecomposeWithMockedLLM:
    """Mockea el cliente Ollama para aislar la logica de validacion."""

    def _fake_client(self, raw_content: str):
        class FakeClient:
            def chat(self, *args, **kwargs):
                return {"message": {"content": raw_content}}

        return FakeClient()

    def test_simple_question_returns_original(self, monkeypatch):
        # is_compound=False → no se invoca al LLM, se devuelve tal cual.
        monkeypatch.setattr(query_decomposer, "_client", self._fake_client("never called"))
        out = decompose("¿Que es el VaR?")
        assert out == ["¿Que es el VaR?"]

    def test_compound_decomposed_to_subqueries(self, monkeypatch):
        raw = '["¿Que es X?", "¿Cual es Y?"]'
        monkeypatch.setattr(query_decomposer, "_client", self._fake_client(raw))
        out = decompose("¿Que es X y cual es Y?")
        assert out == ["¿Que es X?", "¿Cual es Y?"]

    def test_llm_wrapper_with_json_extracted(self, monkeypatch):
        raw = 'Aqui tienes: ["¿Pregunta 1?", "¿Pregunta 2?"]. Espero ayude.'
        monkeypatch.setattr(query_decomposer, "_client", self._fake_client(raw))
        out = decompose("¿Que es X y cual es Y?")
        assert len(out) == 2

    def test_statement_fillers_discarded(self, monkeypatch):
        raw = '["¿Cual es X?", "No hay limite definido.", "¿Y que es Y?"]'
        monkeypatch.setattr(query_decomposer, "_client", self._fake_client(raw))
        out = decompose("¿Que es X y cual es Y?")
        assert out == ["¿Cual es X?", "¿Y que es Y?"]

    def test_capped_at_3_subqueries(self, monkeypatch):
        # Cada subquery debe tener >=6 chars para pasar _looks_like_question.
        raw = '["¿Pregunta A?", "¿Pregunta B?", "¿Pregunta C?", "¿Pregunta D?", "¿Pregunta E?"]'
        monkeypatch.setattr(query_decomposer, "_client", self._fake_client(raw))
        out = decompose("¿Cuando A y cuando B y cuando C y cuando D?")
        assert len(out) == 3

    def test_malformed_json_falls_back_to_original(self, monkeypatch):
        monkeypatch.setattr(query_decomposer, "_client", self._fake_client("not json"))
        original = "¿Que es X y cual es Y?"
        assert decompose(original) == [original]

    def test_all_invalid_falls_back_to_original(self, monkeypatch):
        # Todas las "subqueries" son afirmaciones, no se acepta ninguna.
        raw = '["Esto es afirmacion.", "Tambien afirmacion."]'
        monkeypatch.setattr(query_decomposer, "_client", self._fake_client(raw))
        original = "¿Que es X y cual es Y?"
        assert decompose(original) == [original]

    def test_llm_exception_falls_back(self, monkeypatch):
        class BadClient:
            def chat(self, *args, **kwargs):
                raise RuntimeError("ollama down")

        monkeypatch.setattr(query_decomposer, "_client", BadClient())
        original = "¿Que es X y cual es Y?"
        assert decompose(original) == [original]
