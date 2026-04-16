from src.assistant.extractor import _empty_result, _merge_partials, _sanitize
from src.assistant.prompts import EXTRACT_CATEGORIES


class TestEmptyResult:
    def test_has_all_four_categories(self):
        result = _empty_result()
        assert set(result.keys()) == set(EXTRACT_CATEGORIES)
        assert all(v == [] for v in result.values())


class TestSanitize:
    def test_full_dict_passes_through(self):
        raw = {
            "limites_regulatorios": ["10% patrimonio"],
            "indicadores_riesgo": ["VaR 99%"],
            "decisiones_comite": ["Aprobo el aumento"],
            "fechas_criticas": ["2026-04-15"],
        }
        out = _sanitize(raw)
        assert out == raw

    def test_missing_category_becomes_empty_list(self):
        raw = {"limites_regulatorios": ["10%"]}
        out = _sanitize(raw)
        assert out["limites_regulatorios"] == ["10%"]
        assert out["indicadores_riesgo"] == []
        assert out["decisiones_comite"] == []
        assert out["fechas_criticas"] == []

    def test_non_dict_returns_empty(self):
        assert _sanitize([]) == _empty_result()
        assert _sanitize(None) == _empty_result()
        assert _sanitize("texto") == _empty_result()

    def test_non_list_value_becomes_empty(self):
        raw = {"limites_regulatorios": "no es lista"}
        assert _sanitize(raw)["limites_regulatorios"] == []

    def test_deduplication_case_insensitive(self):
        raw = {"limites_regulatorios": ["10% patrimonio", "10% PATRIMONIO", "10% patrimonio"]}
        out = _sanitize(raw)
        assert len(out["limites_regulatorios"]) == 1

    def test_empty_and_whitespace_strings_dropped(self):
        raw = {"limites_regulatorios": ["", "   ", "valido"]}
        out = _sanitize(raw)
        assert out["limites_regulatorios"] == ["valido"]

    def test_dict_item_flattened(self):
        # El LLM a veces devuelve items como objetos; los serializamos.
        raw = {"decisiones_comite": [{"accion": "aprobo", "detalle": "el limite"}]}
        out = _sanitize(raw)
        assert len(out["decisiones_comite"]) == 1
        assert "aprobo" in out["decisiones_comite"][0]
        assert "el limite" in out["decisiones_comite"][0]

    def test_non_string_items_coerced(self):
        raw = {"fechas_criticas": [2026, 2025.5]}
        out = _sanitize(raw)
        assert len(out["fechas_criticas"]) == 2


class TestMergePartials:
    def test_merges_from_multiple_chunks(self):
        partials = [
            {"limites_regulatorios": ["10%"]},
            {"limites_regulatorios": ["25%"]},
            {"indicadores_riesgo": ["VaR"]},
        ]
        merged = _merge_partials(partials)
        assert set(merged["limites_regulatorios"]) == {"10%", "25%"}
        assert merged["indicadores_riesgo"] == ["VaR"]
        assert merged["decisiones_comite"] == []

    def test_dedup_across_chunks(self):
        partials = [
            {"limites_regulatorios": ["10% patrimonio"]},
            {"limites_regulatorios": ["10% patrimonio"]},
        ]
        merged = _merge_partials(partials)
        assert merged["limites_regulatorios"] == ["10% patrimonio"]

    def test_dedup_case_insensitive_across_chunks(self):
        partials = [
            {"decisiones_comite": ["Aprobo el aumento"]},
            {"decisiones_comite": ["APROBO EL AUMENTO"]},
        ]
        merged = _merge_partials(partials)
        assert len(merged["decisiones_comite"]) == 1

    def test_empty_partials_gives_empty_result(self):
        assert _merge_partials([]) == _empty_result()

    def test_preserves_all_four_keys_always(self):
        merged = _merge_partials([{"limites_regulatorios": ["X"]}])
        assert set(merged.keys()) == set(EXTRACT_CATEGORIES)
