# Pruebas unitarias — RAG Riesgos

Fecha: 2026-04-16 (suite original + modulo asistente)
Resultado global: **171/171 passing** en 7.4 s.

## Proposito

Complementar el eval harness extremo-a-extremo (`scripts/eval.py`, 12/12
preguntas) con tests aislados sobre los componentes criticos del
pipeline. Objetivo: detectar regresiones al nivel de funcion / regex
antes de que impacten las metricas de retrieval o la calidad de la
respuesta.

## Estructura

```
rag-riesgos/
  pytest.ini                  # configuracion: testpaths=tests, -q, filterwarnings
  requirements-dev.txt        # pytest==8.3.4 (dev-only, no entra en requirements.txt)
  tests/
    __init__.py
    test_chunking.py
    test_ingestion.py
    test_faithfulness.py
    test_query_decomposer.py
    test_retriever.py
    test_logger_db.py
    test_assistant_parser.py
    test_assistant_classifier.py
    test_assistant_extractor.py
    test_assistant_summarizer.py
    test_assistant_corpus.py
```

## Como correr

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt    # solo la primera vez
python -m pytest                       # suite completa
python -m pytest tests/test_chunking.py -v    # un modulo con verbose
python -m pytest -k faithfulness       # filtrar por nombre
```

## Cobertura por modulo

### `test_chunking.py` — 22 tests

Cubre la fragmentacion y la deteccion de headings / tablas.

| Clase | Tests | Que valida |
|---|---|---|
| `TestSectionHint` | 7 | Texto vacio, token corto, ARTICULO/CAPITULO/PASO al inicio, numerado jerarquico (`2.1 ...`), heading fuera del primer tercio → herencia |
| `TestAllHeadings` | 2 | Orden de multiples ARTICULOs detectados, posiciones ascendentes |
| `TestTableRegex` | 4 | Dos lineas pipe → match, una sola → no match, **ultima fila sin `\n` final → match** (fix `\Z`), captura todas las filas |
| `TestSplitTables` | 4 | Sin tabla, prosa-tabla-prosa, tabla al inicio, solo tabla |
| `TestChunkPages` | 5 | Prosa plana → 1 chunk, tabla → chunk dedicado sin duplicar, herencia de section_hint entre chunks, metadata preservada, chunk_index monotonico |

Nota: los tests de `_section_hint` documentan que el regex `_HEADING_RE`
tiene flag `(?i)`, por lo que el patron ALL-CAPS tambien matchea frases
normales. No es un bug a corregir — es el comportamiento validado en
produccion (100 % section_hint cobertura, eval 12/12).

### `test_ingestion.py` — 16 tests

| Clase | Tests | Que valida |
|---|---|---|
| `TestClean` | 6 | Null bytes eliminados, whitespace colapsado, newline simple preservado, 3+ newlines → 2, strip extremos, tabs como espacios |
| `TestTableToMarkdown` | 10 | Input vacio, header solo, 2x2 basico, `\n` final garantizado, `None` → "", pipes escapados, newlines internos normalizados a espacio, filas cortas con padding, whitespace en celdas trimmed, separador con ancho del header |

`extract_pdf()` no se testea unitariamente (requiere Tesseract + Poppler
+ PDFs reales); queda cubierto por la validacion end-to-end documentada
en PLAN.md §Iteracion #3.

### `test_faithfulness.py` — 21 tests

| Clase | Tests | Que valida |
|---|---|---|
| `TestNorm` | 4 | Lowercase, remocion de acentos (NFD), colapso de whitespace, transformaciones encadenadas |
| `TestClaimInContext` | 4 | Substring directo, insensible a acentos, claim ausente → False, tolerancia al espacio alrededor de `%` |
| `TestCheckNumbers` | 5 | Numero presente → sin warning, ausente → warning, triviales (`1`-`10`, `100`) ignorados, porcentaje capturado, duplicados deduplicados |
| `TestCheckArticles` | 3 | `Articulo N` presente/ausente, case-insensitive |
| `TestCheckDates` | 2 | `DD de MMMM de AAAA` presente / ausente |
| `TestCheckOverall` | 3 | Respuesta vacia → [], sin claims concretos → [], mezcla de tipos → warnings con los 3 `kind`s |

### `test_query_decomposer.py` — 19 tests

| Clase | Tests | Que valida |
|---|---|---|
| `TestIsCompound` | 6 | Pregunta simple, conjuncion + wh-word, multiples `?`, "ademas" con/sin acento, conjuncion sin wh-word → no compuesta |
| `TestLooksLikeQuestion` | 5 | Termina con `?`, empieza con `¿`, string corto (`¿?`), vacio, afirmacion filler |
| `TestDecomposeWithMockedLLM` | 8 | Simple → original sin llamar LLM, compuesta → subqueries, wrapper JSON, filler descartado, cap a 3, JSON malformado → fallback, todas invalidas → fallback, excepcion del cliente → fallback |

El cliente Ollama se mockea con `monkeypatch.setattr(query_decomposer,
"_client", FakeClient(...))`. Los tests no hacen llamadas reales.

### `test_retriever.py` — 15 tests

| Clase | Tests | Que valida |
|---|---|---|
| `TestStripAccents` | 5 | Tildes minuscula/mayuscula, `ñ` → `n` (consecuencia de NFD), ASCII noop, vacio |
| `TestTokenize` | 10 | Lowercase, stopwords ES (con y sin acento), puntuacion stripped, tokens de 1 char descartados, numeros cortos descartados pero multi-digito preservados, vacio, solo stopwords → [], orden preservado |

### `test_logger_db.py` — 14 tests

Fixture `isolated_db` aisla SQLite en `tmp_path` via
`dataclasses.replace(settings, queries_db=...)` + `monkeypatch.setattr`
al modulo. No toca el `data/queries.db` real.

| Clase | Tests | Que valida |
|---|---|---|
| `TestSchema` | 2 | DB creada al primer acceso, migracion idempotente de `warnings_json` en DB legacy |
| `TestLogQuery` | 6 | Insert basico, UTF-8 con acentos/ñ preservado, sources serializadas como JSON, warnings default `[]`, warnings persistidos, metadata de modelo (llm + embedding) |
| `TestRecent` | 3 | Orden DESC por `ts`, `limit` respetado, DB vacia → [] |
| `TestPurgeOld` | 3 | Borra >30 dias / conserva recientes, sin candidatos → 0, boundary 30d 1min → borrado |

### `test_assistant_parser.py` — 16 tests

Valida el parser JSON tolerante usado por las 3 tareas del asistente.

| Clase | Tests | Que valida |
|---|---|---|
| `TestBalancedExtract` | 7 | Llaves simples y anidadas, corchetes con objetos, llaves dentro de strings JSON ignoradas, escape de comilla, input sin open → None, desbalance → None |
| `TestParseJsonResponse` | 9 | Objeto/array directos, envuelto en prosa, code block markdown, **array antes del primer `{` no se pierde** (regresion), trailing comma recuperada, raw vacio → default |

### `test_assistant_classifier.py` — 17 tests

| Clase | Tests | Que valida |
|---|---|---|
| `TestExtractList` | 5 | Lista directa, dict con lista de dicts dentro, dict sin listas → `[]`, lista de no-dicts ignorada, escalares → `[]` |
| `TestSanitize` | 12 | Items validos preservados, etiquetas desconocidas filtradas, duplicados (primero gana), peso ≤ 0 dropped, clamp a 1.0, normalizacion cuando suma > 1, orden DESC por peso, no-dict items saltados, label case-insensitive, weight no-numerico descartado, input vacio |

### `test_assistant_extractor.py` — 13 tests

| Clase | Tests | Que valida |
|---|---|---|
| `TestEmptyResult` | 1 | Resultado vacio tiene las 4 categorias |
| `TestSanitize` | 7 | Dict completo passthrough, categoria faltante → `[]`, no-dict → empty, valor no-lista → `[]`, dedup case-insensitive, strings vacios dropped, dict items aplanados en string, non-string coercionado |
| `TestMergePartials` | 5 | Merge cross-chunk, dedup exacto y case-insensitive, input vacio, siempre preserva las 4 keys |

### `test_assistant_summarizer.py` — 9 tests

| Clase | Tests | Que valida |
|---|---|---|
| `TestEmptyResult` | 1 | 3 campos fijos (decisiones, riesgos_identificados, acciones_pendientes) |
| `TestSanitize` | 5 | Dict completo, dedup case-insensitive, no-dict → empty, campos faltantes → `[]`, strings vacios |
| `TestMergePartials` | 4 | Merge cross-chunk sobre los 3 campos, dedup, input vacio, keys preservadas |

### `test_assistant_corpus.py` — 9 tests

Fixture `isolated_docs` aisla `settings.docs_dir` en `tmp_path`.

| Clase | Tests | Que valida |
|---|---|---|
| `TestListDocs` | 5 | Dir vacio, extensiones aceptadas (pdf/txt/md), otras ignoradas, orden alfabetico, subdirectorios ignorados |
| `TestListActaDocs` | 3 | Filtro por substring "acta" case-insensitive sobre varios nombres, 3 casings distintos, sin actas → `[]` |

## Politica de uso

1. Correr `python -m pytest` antes de commitear cambios que toquen
   `src/chunking.py`, `src/ingestion.py`, `src/faithfulness.py`,
   `src/query_decomposer.py`, `src/retriever.py`, `src/logger_db.py`
   o cualquier archivo bajo `src/assistant/`.
2. Si un test falla tras un cambio intencional, actualizar el test
   (documentando el cambio), no desactivarlo.
3. Para cambios en retriever, prompt o modelos correr ademas el
   harness extremo-a-extremo (`python -m scripts.eval`) — los tests
   unitarios no cubren calidad semantica.

## Cobertura no incluida

- `src/embeddings.py` y `src/vectorstore.py`: adaptadores sobre Ollama
  y ChromaDB; testear requiere infraestructura real y el eval harness
  ya los ejercita.
- `src/pipeline.py`: orquestador; validado por el eval harness.
- `src/generator.py`: hace llamadas al LLM, no es deterministico.
- `src/reranker.py`: requiere descarga de `bge-reranker-v2-m3` (~2 GB);
  su infra ya esta validada via sweep y el flag default es `false`.
- `src/corpus_sync.py` y watcher: probados manualmente en la pestaña
  de Streamlit.
- `extract_pdf()` y OCR: validados end-to-end con PDFs sinteticos
  (ver PLAN.md §Iteracion #3 y Fase E).

## Resultado final

```
$ python -m pytest
........................................................................ [ 42%]
........................................................................ [ 84%]
...........................                                              [100%]
171 passed in 7.38s
```
