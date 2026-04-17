# Pruebas unitarias — RAG Riesgos

Fecha: 2026-04-16 (iteración #6 — cobertura extendida)
Resultado global: **234/234 passing** en ~5.6 s.

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
  requirements-dev.txt        # pytest==8.3.4 + flake8 (dev-only, no entra en requirements.txt)
  tests/
    __init__.py
    test_chunking.py
    test_ingestion.py
    test_faithfulness.py
    test_query_decomposer.py
    test_retriever.py
    test_logger_db.py
    test_config.py
    test_embeddings.py
    test_generator.py
    test_vectorstore.py
    test_reranker.py
    test_pipeline.py
    test_dashboard.py
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

### `test_config.py` — 6 tests

| Test | Que valida |
|---|---|
| `test_settings_is_frozen_dataclass` | `Settings` es frozen; asignar atributos lanza `FrozenInstanceError` |
| `test_paths_are_absolute` | `docs_dir`, `chroma_dir`, `queries_db` son absolutos independientemente del cwd |
| `test_numeric_defaults_are_positive` | `chunk_size > 0`, `chunk_overlap < chunk_size`, `top_k > 0`, `pool_size ≥ top_k`, `0 ≤ alpha ≤ 1` |
| `test_collection_name_stable` | `"riesgos_corpus"` hardcodeado — un rename rompe persistencia |
| `test_env_override` | `LLM_MODEL`, `TOP_K`, `RERANKER_ENABLED` se leen de env al recargar el modulo |
| `test_reranker_enabled_parses_falsy_strings` | Cualquier valor distinto de `"true"/"1"/"yes"` (case-insensitive) es `False` |

### `test_embeddings.py` — 11 tests

| Clase | Tests | Que valida |
|---|---|---|
| `TestEmbedOne` | 3 | Devuelve lista de floats, segunda llamada con mismo texto no pega Ollama (cache), textos distintos → llamadas distintas |
| `TestEmbedMany` | 2 | Preserva orden, dedup dentro del batch (cache transparente) |
| `TestCacheStats` | 2 | `hit_rate=0.0` sobre cache vacio, `hit_rate=0.5` con 1 hit / 1 miss |
| `TestClearCache` | 1 | Tras `clear_cache()` la siguiente llamada vuelve a pegar Ollama |

`_fetch_embedding` se mockea; no se pega Ollama en los tests.

### `test_generator.py` — 11 tests

| Clase | Tests | Que valida |
|---|---|---|
| `TestFormatContext` | 5 | Estructura `[Fragmento N]`, placeholder `(sin titulo)` cuando falta `section_hint`, separador entre bloques, orden preservado |
| `TestBuildMessages` | 3 | 2 mensajes (system + user), prompt de sistema exige seccion `Fuentes:`, pregunta embebida en user |
| `TestAnswerFastPaths` | 2 | `answer([])` y `answer_stream([])` devuelven el mensaje de "no encuentro" sin llamar al LLM |

### `test_vectorstore.py` — 9 tests

ChromaDB corre en proceso; los tests usan un `tmp_path` aislado con
`dataclasses.replace(settings, chroma_dir=..., collection_name=...)`.

| Clase | Tests | Que valida |
|---|---|---|
| `TestGetCollection` | 2 | Directorio persistente se crea al primer acceso, `reset=True` vacia la coleccion |
| `TestUpsert` | 3 | Count tras upsert, upsert idempotente en mismo id (source/page/chunk_index), metadata `is_table` persistida |
| `TestQuery` | 3 | Vector mas cercano primero, `distance` presente, `top_k` respetado |

### `test_reranker.py` — 10 tests

No se carga el CrossEncoder real (2 GB); se mockea con `numpy`.

| Clase | Tests | Que valida |
|---|---|---|
| `TestMinmax` | 3 | `[1,2,3]` → `[0, 0.5, 1]`, todos iguales → `[0.5, ...]`, single element → `[0.5]` |
| `TestRerankDisabled` | 2 | `hits=[]` → `[]`, preserva orden y trunca a `top_k` |
| `TestRerankEnabled` | 5 | Pool ≤ `top_k` no invoca al modelo, `alpha=1` usa orden del cross-encoder, `alpha=0` preserva orden RRF, scores `rerank_score`/`blended_score` adjuntados |

### `test_pipeline.py` — 10 tests

Todas las dependencias del pipeline se mockean (retriever, generator,
decomposer, reranker, logger_db, faithfulness). El foco es la
orquestación, no la calidad semántica.

| Clase | Tests | Que valida |
|---|---|---|
| `TestAsk` | 6 | Shape del payload, `distance` redondeada, single-query no incluye original en fusion, multi-subquery prepende original, log ejecutado, warnings propagados |
| `TestAskStream` | 3 | Meta vacia hasta consumir el generador, meta poblada tras consumo, stream loguea |
| `TestWarmup` | 2 | Captura excepciones y devuelve `ok=False`, success path devuelve `ok=True` |

### `test_dashboard.py` — 12 tests

Los tests cubren las funciones puras de agregación. El render de
Streamlit no se testea (requiere runtime de la app).

| Clase | Tests | Que valida |
|---|---|---|
| `TestNormalize` | 3 | Quita acentos, lowercase, detecta el marcador `no encuentro` en varias variantes |
| `TestSourceFrequencies` | 3 | DataFrame vacio → cols correctas, cuenta citas cross-rows, orden DESC |
| `TestTopQuestions` | 3 | Vacio → cols correctas, cuenta + mediana, `n` respetado |
| `TestLoadDataframe` | 2 | Sin rows → vacio, rows → `date`/`no_match`/`warnings_count`/`sources_count` derivadas |

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

1. Correr `python -m pytest` antes de commitear cualquier cambio en
   `src/`. La suite tarda ~5.6 s.
2. Correr `python -m flake8 src/ scripts/ tests/ app.py
   --max-line-length=120` como check adicional. Target: 0 issues.
3. Si un test falla tras un cambio intencional, actualizar el test
   (documentando el cambio), no desactivarlo.
4. Para cambios en retriever, prompt o modelos correr ademas el
   harness extremo-a-extremo (`python -m scripts.eval`) — los tests
   unitarios no cubren calidad semantica.

## Cobertura no incluida

- `extract_pdf()` y OCR: requieren Tesseract + Poppler + PDFs reales;
  validados end-to-end (ver PLAN.md §Iteracion #3 y Fase E).
- `src/reranker._get_model()`: cargar el CrossEncoder real descarga
  ~2 GB; los tests mockean el modelo y cubren el resto del flujo
  (min-max, blending, alpha bypass).
- `src/corpus_sync.py` y watcher (`scripts/watch_docs.py`): probados
  manualmente desde la UI de Streamlit y validados por el flujo
  end-to-end.
- Render de Streamlit (`app.py`, `src/dashboard.render`, los
  `src/assistant/ui*.py::render`): requieren runtime de Streamlit;
  las funciones puras subyacentes sí están cubiertas.

## Resultado final

```
$ python -m pytest
........................................................................ [ 30%]
........................................................................ [ 61%]
........................................................................ [ 92%]
..................                                                       [100%]
234 passed in 5.56s
```
