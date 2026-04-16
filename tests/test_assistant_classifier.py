from src.assistant.classifier import _extract_list, _sanitize


class TestExtractList:
    def test_direct_list(self):
        assert _extract_list([{"label": "x"}]) == [{"label": "x"}]

    def test_wrapped_in_dict(self):
        wrapped = {"classification": [{"label": "a"}, {"label": "b"}]}
        assert _extract_list(wrapped) == [{"label": "a"}, {"label": "b"}]

    def test_dict_without_list_returns_empty(self):
        assert _extract_list({"msg": "no list"}) == []

    def test_dict_with_mixed_list_ignored(self):
        # Lista de ints, no de dicts -> no se acepta como clasificacion.
        wrapped = {"result": [1, 2, 3]}
        assert _extract_list(wrapped) == []

    def test_scalar_returns_empty(self):
        assert _extract_list("just a string") == []
        assert _extract_list(42) == []
        assert _extract_list(None) == []


class TestSanitize:
    def test_valid_items_preserved(self):
        items = [
            {"label": "riesgo_credito", "weight": 0.6},
            {"label": "riesgo_mercado", "weight": 0.3},
        ]
        out = _sanitize(items)
        assert len(out) == 2
        assert out[0]["label"] == "riesgo_credito"
        assert out[0]["weight"] == 0.6

    def test_unknown_label_filtered(self):
        items = [
            {"label": "riesgo_credito", "weight": 0.5},
            {"label": "riesgo_inventado", "weight": 0.5},
        ]
        out = _sanitize(items)
        assert len(out) == 1
        assert out[0]["label"] == "riesgo_credito"

    def test_duplicate_label_first_wins(self):
        items = [
            {"label": "riesgo_mercado", "weight": 0.7},
            {"label": "riesgo_mercado", "weight": 0.2},
        ]
        out = _sanitize(items)
        assert len(out) == 1
        assert out[0]["weight"] == 0.7

    def test_zero_or_negative_weight_dropped(self):
        items = [
            {"label": "riesgo_credito", "weight": 0},
            {"label": "riesgo_mercado", "weight": -0.3},
            {"label": "riesgo_operacional", "weight": 0.5},
        ]
        out = _sanitize(items)
        assert len(out) == 1
        assert out[0]["label"] == "riesgo_operacional"

    def test_weight_above_one_clamped(self):
        items = [{"label": "riesgo_credito", "weight": 1.5}]
        out = _sanitize(items)
        assert out[0]["weight"] <= 1.0

    def test_sum_greater_than_one_normalized(self):
        items = [
            {"label": "riesgo_credito", "weight": 0.8},
            {"label": "riesgo_mercado", "weight": 0.8},
        ]
        out = _sanitize(items)
        total = sum(x["weight"] for x in out)
        assert total <= 1.0 + 1e-9

    def test_sorted_descending_by_weight(self):
        items = [
            {"label": "riesgo_credito", "weight": 0.2},
            {"label": "riesgo_mercado", "weight": 0.5},
            {"label": "riesgo_operacional", "weight": 0.3},
        ]
        out = _sanitize(items)
        weights = [x["weight"] for x in out]
        assert weights == sorted(weights, reverse=True)

    def test_non_dict_items_skipped(self):
        items = ["riesgo_credito", 42, {"label": "riesgo_mercado", "weight": 0.5}]
        out = _sanitize(items)
        assert len(out) == 1
        assert out[0]["label"] == "riesgo_mercado"

    def test_label_case_insensitive(self):
        items = [{"label": "RIESGO_CREDITO", "weight": 0.5}]
        out = _sanitize(items)
        assert len(out) == 1
        assert out[0]["label"] == "riesgo_credito"

    def test_non_numeric_weight_skipped(self):
        items = [{"label": "riesgo_credito", "weight": "alto"}]
        assert _sanitize(items) == []

    def test_empty_input(self):
        assert _sanitize([]) == []
        assert _sanitize(None) == []
