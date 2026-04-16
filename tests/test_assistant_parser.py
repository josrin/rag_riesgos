from src.assistant.llm_utils import _balanced_extract, parse_json_response


class TestBalancedExtract:
    def test_simple_braces(self):
        assert _balanced_extract('xx {"a": 1} yy', "{", "}") == '{"a": 1}'

    def test_nested_braces(self):
        text = 'prefix {"outer": {"inner": 1}, "b": 2} suffix'
        assert _balanced_extract(text, "{", "}") == '{"outer": {"inner": 1}, "b": 2}'

    def test_brackets_with_nested_objects(self):
        text = 'aa [{"a": 1}, {"b": 2}] bb'
        assert _balanced_extract(text, "[", "]") == '[{"a": 1}, {"b": 2}]'

    def test_braces_inside_string_ignored(self):
        # Las llaves dentro de una string JSON no cuentan para el balance.
        text = '{"label": "{nope}"}'
        assert _balanced_extract(text, "{", "}") == '{"label": "{nope}"}'

    def test_escaped_quote_inside_string(self):
        text = r'{"label": "con \"comilla\" adentro"}'
        assert _balanced_extract(text, "{", "}") == text

    def test_missing_open_returns_none(self):
        assert _balanced_extract("no braces here", "{", "}") is None

    def test_unbalanced_returns_none(self):
        assert _balanced_extract('{"a": 1', "{", "}") is None


class TestParseJsonResponse:
    def test_direct_object(self):
        assert parse_json_response('{"a": 1}', default=None) == {"a": 1}

    def test_direct_array(self):
        assert parse_json_response("[1, 2, 3]", default=None) == [1, 2, 3]

    def test_object_wrapped_in_prose(self):
        raw = 'Aqui va: {"x": 42} espero que sirva.'
        assert parse_json_response(raw, default=None) == {"x": 42}

    def test_array_wrapped_in_prose(self):
        raw = 'Respuesta: [{"label": "a"}, {"label": "b"}]. Fin.'
        result = parse_json_response(raw, default=None)
        assert result == [{"label": "a"}, {"label": "b"}]

    def test_array_inside_markdown_code_block(self):
        raw = '```json\n[{"label": "riesgo_mercado", "weight": 0.8}]\n```'
        result = parse_json_response(raw, default=None)
        assert result == [{"label": "riesgo_mercado", "weight": 0.8}]

    def test_array_before_object_returns_array(self):
        # Regresion: el array aparece primero, no debe perderse por el {} del primer elemento.
        raw = 'PASO 4: [{"label": "a", "weight": 0.5}, {"label": "b", "weight": 0.5}]'
        result = parse_json_response(raw, default=None)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_trailing_comma_recovered(self):
        raw = '{"a": 1, "b": 2,}'
        assert parse_json_response(raw, default=None) == {"a": 1, "b": 2}

    def test_invalid_falls_back_to_default(self):
        assert parse_json_response("no json at all", default=[]) == []
        assert parse_json_response("", default={"k": "v"}) == {"k": "v"}

    def test_none_raw_returns_default(self):
        assert parse_json_response("", default="x") == "x"
