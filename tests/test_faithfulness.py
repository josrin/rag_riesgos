from src.faithfulness import _claim_in_context, _norm, check


class TestNorm:
    def test_lowercase(self):
        assert _norm("HOLA") == "hola"

    def test_removes_accents(self):
        assert _norm("metodología") == "metodologia"

    def test_collapses_whitespace(self):
        assert _norm("a   b\tc") == "a b c"

    def test_multiple_transformations(self):
        # _norm colapsa whitespace interno pero NO strip-ea los extremos;
        # eso lo hace _claim_in_context puntualmente.
        assert _norm("  METÁFORA  Á") == " metafora a"


class TestClaimInContext:
    def test_direct_substring_match(self):
        assert _claim_in_context("12.350", _norm("Reporto 12.350 millones"))

    def test_accent_insensitive(self):
        assert _claim_in_context("metodología", _norm("La metodologia es FHS"))

    def test_absent_claim_returns_false(self):
        assert not _claim_in_context("99%", _norm("No se menciona ningun porcentaje"))

    def test_whitespace_around_percent_tolerated(self):
        # La respuesta escribe "99 %" y el contexto "99%".
        assert _claim_in_context("99 %", _norm("El umbral es del 99% anual"))


class TestCheckNumbers:
    def test_present_number_no_warning(self):
        warnings = check("La perdida fue 12.350", "El informe cita 12.350 millones")
        assert warnings == []

    def test_absent_number_produces_warning(self):
        warnings = check("La perdida fue 99.9%", "El informe cita 12.350 millones")
        assert len(warnings) == 1
        assert warnings[0]["kind"] == "numero"
        assert "99" in warnings[0]["claim"]

    def test_trivial_single_digit_not_flagged(self):
        # Numeros como "3" o "5" son demasiado genericos para ser utiles.
        warnings = check("Hay 5 pasos", "El proceso tiene etapas definidas")
        assert warnings == []

    def test_percentage_number_captured(self):
        # El `\b` final del regex no se satisface contra `%`, asi que el
        # claim se captura como "28" (sin el signo). Lo importante es
        # que el numero no cuele al contexto cuando no aparece.
        warnings = check("El limite es 28%", "No se menciona el limite")
        assert any(w["kind"] == "numero" and "28" in w["claim"] for w in warnings)

    def test_duplicate_claim_deduplicated(self):
        warnings = check(
            "Primero 99.9% y despues otra vez 99.9% y finalmente 99.9%.",
            "Sin porcentajes en el contexto",
        )
        assert len(warnings) == 1


class TestCheckArticles:
    def test_present_article_no_warning(self):
        warnings = check("Segun el Articulo 5", "El Articulo 5 establece...")
        assert warnings == []

    def test_absent_article_produces_warning(self):
        warnings = check("Segun el Articulo 99", "El Articulo 5 establece...")
        kinds = {w["kind"] for w in warnings}
        assert "articulo" in kinds

    def test_article_case_insensitive(self):
        warnings = check("ARTICULO 7", "El articulo 7 regula el tema")
        assert warnings == []


class TestCheckDates:
    def test_present_date_no_warning(self):
        warnings = check("El 15 de marzo de 2024", "Aprobado el 15 de marzo de 2024")
        assert warnings == []

    def test_absent_date_produces_warning(self):
        warnings = check("El 15 de marzo de 2024", "El informe es reciente")
        assert any(w["kind"] == "fecha" for w in warnings)


class TestCheckOverall:
    def test_empty_answer_no_warnings(self):
        assert check("", "algun contexto") == []

    def test_clean_answer_no_warnings(self):
        # Sin cifras, articulos ni fechas especificos.
        warnings = check(
            "La entidad aplica gobierno corporativo robusto.",
            "Contexto arbitrario",
        )
        assert warnings == []

    def test_mixed_claims_multiple_kinds(self):
        warnings = check(
            "El Articulo 42 del 15 de marzo de 2024 fija el 99.5%.",
            "Sin referencias en el contexto",
        )
        kinds = {w["kind"] for w in warnings}
        assert {"articulo", "fecha", "numero"} <= kinds
