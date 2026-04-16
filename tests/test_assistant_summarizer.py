from src.assistant.prompts import SUMMARY_FIELDS
from src.assistant.summarizer import _empty_result, _merge_partials, _sanitize


class TestEmptyResult:
    def test_has_three_fields(self):
        result = _empty_result()
        assert set(result.keys()) == set(SUMMARY_FIELDS)
        assert all(v == [] for v in result.values())


class TestSanitize:
    def test_full_dict(self):
        raw = {
            "decisiones": ["Aprobo X"],
            "riesgos_identificados": ["Ciberseguridad"],
            "acciones_pendientes": ["Entregar informe"],
        }
        assert _sanitize(raw) == raw

    def test_missing_field_becomes_empty(self):
        raw = {"decisiones": ["Aprobo X"]}
        out = _sanitize(raw)
        assert out["decisiones"] == ["Aprobo X"]
        assert out["riesgos_identificados"] == []
        assert out["acciones_pendientes"] == []

    def test_dedup_case_insensitive(self):
        raw = {"decisiones": ["Aprobo X", "APROBO X"]}
        assert len(_sanitize(raw)["decisiones"]) == 1

    def test_non_dict_returns_empty(self):
        assert _sanitize([1, 2, 3]) == _empty_result()
        assert _sanitize(None) == _empty_result()

    def test_empty_strings_dropped(self):
        raw = {"decisiones": ["", "  ", "valido"]}
        assert _sanitize(raw)["decisiones"] == ["valido"]


class TestMergePartials:
    def test_merge_three_fields(self):
        partials = [
            {"decisiones": ["Aprobo VaR"]},
            {"riesgos_identificados": ["Ciberseguridad"]},
            {"acciones_pendientes": ["Entregar informe"]},
        ]
        merged = _merge_partials(partials)
        assert merged["decisiones"] == ["Aprobo VaR"]
        assert merged["riesgos_identificados"] == ["Ciberseguridad"]
        assert merged["acciones_pendientes"] == ["Entregar informe"]

    def test_dedup_across_chunks(self):
        partials = [
            {"decisiones": ["Aprobo VaR"]},
            {"decisiones": ["Aprobo VaR"]},
            {"decisiones": ["Aprobo VaR"]},
        ]
        assert _merge_partials(partials)["decisiones"] == ["Aprobo VaR"]

    def test_empty_input_returns_empty(self):
        assert _merge_partials([]) == _empty_result()

    def test_preserves_three_keys_always(self):
        merged = _merge_partials([{"decisiones": ["X"]}])
        assert set(merged.keys()) == set(SUMMARY_FIELDS)
