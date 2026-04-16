from src.chunking import (
    _HEADING_RE,
    _TABLE_RE,
    _all_headings,
    _section_hint,
    _split_tables,
    chunk_pages,
)


# El regex _HEADING_RE tiene flag (?i), asi que el patron "ALL-CAPS"
# matchea tambien frases normales (por diseno, para cubrir maquetacion
# atipica). Estos tests documentan el comportamiento REAL, no el ideal.


class TestSectionHint:
    def test_empty_text_returns_empty(self):
        assert _section_hint("") == ""

    def test_one_short_word_returns_empty(self):
        # Palabra de 1-2 chars no activa ningun patron del regex.
        assert _section_hint("la") == ""

    def test_article_heading_at_start(self):
        text = "ARTICULO 5 Objeto del presente reglamento\n\nEl presente articulo..."
        assert "ARTICULO 5" in _section_hint(text)

    def test_chapter_heading_at_start(self):
        text = "CAPITULO II De las obligaciones\n\nLas entidades deben..."
        assert "CAPITULO II" in _section_hint(text)

    def test_paso_heading(self):
        text = "PASO 3 Calcular la volatilidad\nSe aplica la formula..."
        assert "PASO 3" in _section_hint(text)

    def test_numbered_hierarchical_heading(self):
        text = "2.1 Indicadores clave de riesgo\nLos KRIs medidos son..."
        assert "Indicadores clave" in _section_hint(text)

    def test_heading_out_of_first_third_returns_empty(self):
        # Usamos tokens de 1-2 chars para no activar el regex permisivo.
        prose_safe = "la a b. " * 200
        text = prose_safe + "\n\nARTICULO 99 Algo"
        assert _section_hint(text) == ""


class TestAllHeadings:
    def test_multiple_article_headings_in_order(self):
        # Intercalamos tokens cortos para que solo los ARTICULOs matcheen.
        text = "ARTICULO 1 Objeto\n\nla a b\n\nARTICULO 2 Ambito\n\nla a b"
        hs = _all_headings(text)
        articulos = [h for _, h in hs if "ARTICULO" in h]
        assert len(articulos) == 2
        assert "ARTICULO 1" in articulos[0]
        assert "ARTICULO 2" in articulos[1]

    def test_positions_are_ascending(self):
        text = "ARTICULO 1 Uno\n\nARTICULO 2 Dos\n\nARTICULO 3 Tres"
        hs = _all_headings(text)
        positions = [p for p, _ in hs]
        assert positions == sorted(positions)


class TestTableRegex:
    def test_two_pipe_lines_matched(self):
        text = "| a | b |\n| c | d |\n"
        assert _TABLE_RE.search(text) is not None

    def test_one_pipe_line_not_matched(self):
        text = "| a | b |\nprosa normal\n"
        assert _TABLE_RE.search(text) is None

    def test_last_line_without_newline_matched(self):
        text = "| a | b |\n| c | d |"
        assert _TABLE_RE.search(text) is not None

    def test_regex_captures_all_rows(self):
        text = "| h1 | h2 |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        m = _TABLE_RE.search(text)
        assert m is not None
        assert "| 3 | 4 |" in m.group(0)


class TestSplitTables:
    def test_no_table_returns_single_prose(self):
        segs = _split_tables("Solo prosa sin tabla alguna.")
        assert segs == [("Solo prosa sin tabla alguna.", False)]

    def test_prose_then_table_then_prose(self):
        text = "Intro\n\n| a | b |\n| c | d |\n\nCierre"
        segs = _split_tables(text)
        kinds = [s[1] for s in segs]
        assert kinds == [False, True, False]

    def test_leading_table(self):
        text = "| a | b |\n| c | d |\n\nTexto despues"
        segs = _split_tables(text)
        assert segs[0][1] is True
        assert segs[-1][1] is False

    def test_only_table(self):
        text = "| a | b |\n| c | d |\n"
        segs = _split_tables(text)
        assert len(segs) == 1
        assert segs[0][1] is True


class TestChunkPages:
    def _page(self, text: str, source: str = "doc.pdf", page: int = 1) -> dict:
        return {"source": source, "page": page, "text": text, "extraction": "native"}

    def test_plain_prose_single_chunk(self):
        chunks = chunk_pages([self._page("Parrafo corto sin heading.")])
        assert len(chunks) == 1
        assert chunks[0]["is_table"] is False
        assert chunks[0]["source"] == "doc.pdf"
        assert chunks[0]["page"] == 1
        assert chunks[0]["chunk_index"] == 0

    def test_table_emitted_as_dedicated_chunk(self):
        text = "Intro\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\nCierre"
        chunks = chunk_pages([self._page(text)])
        tables = [c for c in chunks if c["is_table"]]
        prose = [c for c in chunks if not c["is_table"]]
        assert len(tables) == 1
        assert "| 1 | 2 |" in tables[0]["text"]
        assert all("| 1 | 2 |" not in p["text"] for p in prose)

    def test_section_hint_inherited_across_chunks(self):
        # Usamos tokens de 1-2 chars en el cuerpo para no disparar el
        # regex permisivo; asi los chunks sin heading propio heredan el
        # ARTICULO 7 del primero.
        body = "la a b c. " * 400
        text = "ARTICULO 7 Limites\n\n" + body
        chunks = chunk_pages([self._page(text)])
        assert len(chunks) >= 2
        assert all("ARTICULO 7" in c["section_hint"] for c in chunks)

    def test_metadata_preserved(self):
        chunks = chunk_pages([self._page("Texto", source="foo.md", page=3)])
        assert chunks[0]["source"] == "foo.md"
        assert chunks[0]["page"] == 3
        assert chunks[0]["extraction"] == "native"

    def test_chunk_index_monotonic(self):
        text = ("Parrafo largo. " * 300)
        chunks = chunk_pages([self._page(text)])
        indices = [c["chunk_index"] for c in chunks]
        assert indices == sorted(indices)
        assert indices[0] == 0
