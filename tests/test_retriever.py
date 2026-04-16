from src.retriever import _strip_accents, _tokenize


class TestStripAccents:
    def test_removes_tilde(self):
        assert _strip_accents("metodología") == "metodologia"

    def test_removes_uppercase_accents(self):
        assert _strip_accents("MÉTODO") == "METODO"

    def test_leaves_enye_without_tilde_unchanged(self):
        # La ene con tilde es letra base + combining mark; NFD la separa.
        assert _strip_accents("año") == "ano"

    def test_noop_on_ascii(self):
        assert _strip_accents("VaR") == "VaR"

    def test_empty_string(self):
        assert _strip_accents("") == ""


class TestTokenize:
    def test_lowercases(self):
        toks = _tokenize("GARCH Simulacion")
        assert "garch" in toks
        assert "simulacion" in toks

    def test_removes_stopwords(self):
        # "el", "es", "la" son stopwords clasicas del espanol.
        toks = _tokenize("el var es la metrica")
        assert "el" not in toks
        assert "es" not in toks
        assert "la" not in toks
        assert "var" in toks
        assert "metrica" in toks

    def test_stopword_with_accent_normalized(self):
        # "qué" -> accentless "que" esta en la lista de stopwords.
        toks = _tokenize("¿Qué es el VaR?")
        assert "que" not in toks
        assert "qué" not in toks

    def test_punctuation_stripped(self):
        toks = _tokenize("¿Cual? ¡El VaR!")
        assert all(not any(c in t for c in "¿?¡!") for t in toks)

    def test_single_char_tokens_dropped(self):
        # len > 1 en el filtro
        toks = _tokenize("a b c metodologia")
        assert "a" not in toks
        assert "b" not in toks
        assert "metodologia" in toks

    def test_numbers_short_are_filtered(self):
        # "1,1" se tokeniza como ["1", "1"]; cada uno len=1 → descartado.
        toks = _tokenize("GARCH 1,1")
        assert toks == ["garch"]

    def test_multi_digit_numbers_kept(self):
        toks = _tokenize("COP 12350 millones")
        assert "12350" in toks

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_only_stopwords_returns_empty(self):
        assert _tokenize("el la de que para") == []

    def test_preserves_order(self):
        toks = _tokenize("VaR GARCH FHS volatilidad")
        assert toks == ["var", "garch", "fhs", "volatilidad"]
