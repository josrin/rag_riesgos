from src.ingestion import _clean, _table_to_markdown


class TestClean:
    def test_strips_null_bytes(self):
        assert "\x00" not in _clean("hola\x00mundo")

    def test_collapses_multiple_spaces(self):
        assert _clean("a    b   c") == "a b c"

    def test_preserves_single_newline(self):
        assert _clean("linea1\nlinea2") == "linea1\nlinea2"

    def test_collapses_three_or_more_newlines_to_two(self):
        assert _clean("a\n\n\n\nb") == "a\n\nb"

    def test_strips_leading_trailing_whitespace(self):
        assert _clean("  \n  texto  \n  ") == "texto"

    def test_tabs_collapsed_with_spaces(self):
        assert _clean("a\t\t\tb") == "a b"


class TestTableToMarkdown:
    def test_empty_returns_empty_string(self):
        assert _table_to_markdown([]) == ""

    def test_header_only_empty_row(self):
        assert _table_to_markdown([[]]) == ""

    def test_basic_2x2(self):
        md = _table_to_markdown([["a", "b"], ["1", "2"]])
        expected = "| a | b |\n| --- | --- |\n| 1 | 2 |\n"
        assert md == expected

    def test_trailing_newline_present(self):
        md = _table_to_markdown([["h1"], ["v1"]])
        assert md.endswith("\n")

    def test_none_cell_becomes_empty_string(self):
        md = _table_to_markdown([["a", "b"], [None, "x"]])
        assert "|  | x |" in md

    def test_pipe_inside_cell_escaped(self):
        md = _table_to_markdown([["a", "b"], ["x|y", "z"]])
        assert "x\\|y" in md
        assert "| x\\|y | z |" in md

    def test_newline_inside_cell_normalized(self):
        md = _table_to_markdown([["a", "b"], ["line1\nline2", "x"]])
        assert "line1 line2" in md
        assert "line1\nline2" not in md.replace("\n|", "|")

    def test_short_row_padded(self):
        # Fila con menos columnas que el header debe rellenarse con "".
        md = _table_to_markdown([["a", "b", "c"], ["1"]])
        lines = md.strip().split("\n")
        # header + separator + data
        assert len(lines) == 3
        assert lines[2] == "| 1 |  |  |"

    def test_whitespace_trimmed(self):
        md = _table_to_markdown([["  a  ", "\tb\t"], ["  1  ", "2"]])
        assert "| a | b |" in md
        assert "| 1 | 2 |" in md

    def test_separator_matches_header_width(self):
        md = _table_to_markdown([["c1", "c2", "c3", "c4"], ["a", "b", "c", "d"]])
        lines = md.split("\n")
        # separator line (second) debe tener 4 --- como el header
        assert lines[1] == "| --- | --- | --- | --- |"
