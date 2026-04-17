# Plan de ejecución — RAG Riesgos

> Este archivo es un **documento vivo**. Cada iteración debe
> actualizar el estado de los pasos, añadir los nuevos y dejar
> comentarios bajo "Notas de iteración".

---

## Leyenda de estado
- `[ ]` pendiente
- `[~]` en progreso
- `[x]` completado
- `[!]` bloqueado (indicar motivo)

---

## Iteración actual: **#1 — Construcción del pipeline base**
Fecha inicio: 2026-04-15

### Fase A — Scaffolding y entorno
- [x] Crear estructura del repositorio en `C:\Data Projects\rag-riesgos`.
- [x] `docs/` como carpeta de entrada de documentos del usuario.
- [x] `requirements.txt` con versiones exactas y prerequisitos documentados.
- [x] `.env.example` con todas las variables configurables.
- [x] `.gitignore` excluyendo `data/`, `.venv/`, documentos fuente.
- [x] Crear venv `.venv` con Python 3.13.12.
- [x] `pip install -r requirements.txt` — todas las dependencias OK.

### Fase B — Código del pipeline (`src/`)
- [x] `config.py`: carga de `.env` con rutas, modelos y parámetros.
- [x] `ingestion.py`: extracción híbrida (PDF nativo + OCR fallback + .txt/.md).
- [x] `chunking.py`: `RecursiveCharacterTextSplitter` 900/150 + `section_hint`.
- [x] `embeddings.py`: wrapper Ollama `nomic-embed-text` con reintentos.
- [x] `vectorstore.py`: ChromaDB persistente, colección única, coseno.
- [x] `generator.py`: prompt sistema + usuario con citas obligatorias.
- [x] `logger_db.py`: SQLite con purga a 30 días.
- [x] `pipeline.py`: orquestación `index_corpus()` y `ask()`.

### Fase C — Herramientas
- [x] `scripts/index_documents.py`: reindexar corpus completo.
- [x] `scripts/setup_ollama.ps1`: instalar Ollama + pull de modelos.
- [x] `app.py`: UI Streamlit con consulta e historial.

### Fase D — Documentación
- [x] `DESIGN.md` con justificación de decisiones.
- [x] `PLAN.md` (este archivo).
- [x] `README.md` con pasos de arranque, prerequisitos y troubleshooting.

### Fase E — Puesta en marcha
- [x] Instalar **Ollama 0.20.7** via `winget`.
- [x] Descargar modelo `nomic-embed-text` (274 MB) — reemplazado por `bge-m3`.
- [x] Descargar modelo `llama3` (4.7 GB).
- [x] Descargar modelo `bge-m3` (1.2 GB) — embeddings multilingues.
- [x] Cargar 5 documentos `.txt` en `docs/`.
- [x] Indexar corpus: **5 documentos → 36 chunks** en ChromaDB (17 seg con bge-m3).
- [x] Probar consulta de VaR en Streamlit, detectar fallo de recuperacion.
- [x] Instalar **Tesseract-OCR** 5.4.0 y **Poppler** 25.07.0 (winget,
      2026-04-16). Idioma `spa` copiado a `data/tessdata/` local al
      proyecto (`TESSDATA_PREFIX`) para no tocar `Program Files`.
      Validado end-to-end con PDF escaneado: `extract_pdf()` detecta
      texto escaso, dispara OCR y recupera el contenido en español.

### Fase F — Mejoras de recuperacion (iteracion #1.5)
- [x] Migrar embeddings a `bge-m3` (resuelve baja calidad en español).
- [x] Implementar retriever hibrido BM25 + vectorial con RRF
      (`src/retriever.py`) adelantando el item de la iteracion #3.
- [x] Tokenizer BM25 con stopwords en español + normalizacion de acentos.
- [x] Quitar panel "Fuentes recuperadas" de la pestaña Consulta
      (duplicaba la seccion "Fuentes:" de la respuesta).
- [x] Validar que la pregunta del VaR ahora responde correctamente
      citando "Simulacion Historica Filtrada (FHS)" y GARCH.
- [x] Evaluacion con 12 preguntas cubriendo los 5 documentos y 5
      niveles de complejidad. Resultado: 10/12 (83 %) → ver
      `EVALUACION.md`.
- [x] Hardening del prompt del sistema: sin preambulos, formato de
      citas estricto, manejo explicito de preguntas compuestas, no
      concatenar encabezados de fragmentos.
- [x] Query decomposition (`src/query_decomposer.py`): detecta
      preguntas compuestas y las descompone en subqueries antes de
      recuperar. Fusion con `retriever.hybrid_query_multi()` usando un
      segundo RRF sobre los rankings por subquery.
- [x] Re-ejecucion de las dos preguntas que fallaron → 12/12 (100 %).
- [x] Validado el historial en la pestaña "Historial (30 días)"
      (2026-04-16): `ask_stream()` persiste cada consulta en
      `queries.db` al cerrar el generador (UTF-8 con acentos íntegros,
      latencia, fuentes y warnings). `logger_db.recent(limit=200)`
      devuelve dicts con `sources`/`warnings` ya parseados —
      compatible con el renderizado de `app.py`. `purge_old()` respeta
      `LOG_RETENTION_DAYS=30`: inyectando registros a 40 y 1 día se
      borra solo el primero.

---

## Cómo retomar la sesión

```powershell
# 1. Abrir terminal en la carpeta del proyecto
cd "C:\Data Projects\rag-riesgos"

# 2. Activar el entorno virtual
.\.venv\Scripts\Activate.ps1

# 3. Verificar que Ollama está corriendo y con los modelos
ollama list
# Debe mostrar llama3:latest y bge-m3:latest
# Si Ollama no responde, ejecutar en otra terminal: ollama serve

# 4. (Opcional) Sincronizar incrementalmente si cambiaron los docs/
python -m scripts.sync_docs
# o dejar el watcher corriendo en segundo plano:
#   python -m scripts.watch_docs

# 5. Lanzar la UI
streamlit run app.py
# Abrir http://localhost:8501
```

---

## Versiones del entorno (validadas 2026-04-16)

| Componente | Versión |
|---|---|
| Windows | 11 Pro 10.0.26200 |
| Python | 3.13.12 |
| Ollama | 0.20.7 |
| chromadb | 1.5.7 |
| ollama (Python) | 0.6.1 |
| streamlit | 1.56.0 |
| pdfplumber | 0.11.9 |
| langchain-text-splitters | 1.1.1 |
| rank-bm25 | 0.2.2 |
| sentence-transformers | 5.4.1 |
| torch | 2.11.0 (CPU) |
| watchdog | 4.0.2 |
| pandas | 3.0.2 |

**Modelos Ollama activos:** `llama3:latest` (4.7 GB), `bge-m3:latest`
(1.2 GB), `llama3.1:8b` (4.9 GB, solo para comparaciones).
**Modelo HuggingFace (opcional):** `BAAI/bge-reranker-v2-m3` (~2 GB
en `~/.cache/huggingface/`, se descarga en el primer uso si
`RERANKER_ENABLED=true`).

---

## Iteración #2 — Evaluación y mejora

> Ya existe una evaluacion manual con 12 preguntas documentada en
> `EVALUACION.md`. Esta iteracion formaliza el harness automatizado.

- [x] Crear `eval/ground_truth.jsonl` con las 12 preguntas, sus
      `expected_sources` y lista de `must_contain` (literales).
- [x] Runner `scripts/eval.py` con métricas automáticas:
      `source_recall@5`, `answer_accuracy`, `faithfulness`, latencia
      p50/p95. Exporta `eval/report.json`.
- [x] Guardrail de faithfulness en runtime (`src/faithfulness.py`):
      detecta cifras, articulos y fechas que la respuesta cita pero no
      aparecen en el contexto. Se expone como `faithfulness_warnings`
      en la respuesta de `ask()` y se muestra como aviso plegable en
      Streamlit.
- [x] Warmup del LLM y embeddings al arrancar Streamlit
      (`pipeline.warmup()` cacheado con `@st.cache_resource`).
- [x] Script `scripts/eval_compare.py` que corre el harness contra
      varios LLMs y emite tabla comparativa
      (`eval/compare_report.json`). Uso:
      `python -m scripts.eval_compare llama3 llama3.1:8b mistral`.
- [x] Comparacion `llama3` vs `llama3.1:8b`. Ganador: **llama3**
      (100 % acc, p50 1537 ms) sobre `llama3.1:8b` (97 % acc, p50
      1933 ms). El 3 % perdido fue Q10 (sintesis cross-doc): 3.1
      tenia el contexto pero no extrajo las cifras especificas. Se
      mantiene `LLM_MODEL=llama3`.

---

## Iteración #3 — Calidad de recuperación

- [x] Integrar cross-encoder reranker (`BAAI/bge-reranker-v2-m3` via
      `sentence-transformers`). Codigo en `src/reranker.py`, configurable
      por `RERANKER_ENABLED` en `.env`. **Deshabilitado por defecto**:
      el sweep de `RERANKER_ALPHA` en el corpus actual (37 chunks) no
      encontro ningun alpha > 0 que superara al RRF puro; ademas el
      reranker anade ~5-9 s de latencia en CPU. Evaluar reactivacion
      cuando el corpus crezca a cientos/miles de chunks.
- [x] ~~Probar búsqueda híbrida (BM25 + vectorial)~~ — **adelantado a
      iteración #1.5** (ver `src/retriever.py`).
- [x] Enriquecer `section_hint`: detección ampliada de headings (Artículo,
      Capítulo, Sección, Parte, Paso N:, jerarquicos 1.2.3, ALL-CAPS),
      umbral de posición (solo se considera heading propio si aparece
      en el primer tercio del chunk), y herencia del último heading
      visto. Resultado: 100 % de chunks con section_hint (antes ~60 %).
- [x] Detección de tablas markdown (`| ... |`) con chunk dedicado y
      flag `is_table=true` en metadata. Evita romper tablas al chunkear.
- [x] Detección de tablas en PDFs via `pdfplumber.find_tables()` +
      `page.filter(...)` (2026-04-16). Las tablas se convierten a
      markdown pipe-table con `_table_to_markdown()`, se anexan al
      texto de la página y el chunker existente las separa como
      chunks dedicados (`is_table=true`). Se filtran los objetos del
      `extract_text()` dentro del bbox de cada tabla para evitar que
      la prosa duplique el contenido tabular. Regex `_TABLE_RE` del
      chunker ahora admite `\Z` para no perder la última fila cuando
      un `strip()` aguas arriba elimina el `\n` final. Test con PDF
      real de reportlab: 1 chunk prosa + 1 chunk tabla completa, sin
      duplicación. Eval harness sigue 12/12 (p50 1517 ms, sin
      regresión).
- [x] **Blended reranking**: `reranker.rerank()` ahora combina
      `rerank_score` y `rrf_score` con normalizacion min-max en el
      pool: `final = alpha * rerank_norm + (1-alpha) * rrf_norm`.
      Configurable via `RERANKER_ALPHA`. Sweep con
      `scripts/eval_alpha_sweep.py` en corpus de 37 chunks: ningun
      alpha > 0 supera al RRF puro (alpha=0). Decision: mantener
      `RERANKER_ENABLED=false` por defecto; la infraestructura queda
      lista para corpus de cientos/miles de chunks donde si deberia
      aportar.

---

## Iteración #3.5 — UX y latencia

- [x] Streaming de tokens en Streamlit (`pipeline.ask_stream()` +
      `generator.answer_stream()` + `st.write_stream`). La respuesta
      aparece incremental en lugar de esperar al cierre del LLM.
- [x] Warmup de modelos al arrancar la app (integrado en
      `pipeline.warmup()`, cacheado con `@st.cache_resource`).
- [x] Cache LRU de embeddings (`@lru_cache(maxsize=2048)` sobre
      `embed_one`, clave `(texto, modelo)`). Benchmark: segunda
      corrida del eval batch -22 % wall time (22.6 s -> 17.7 s),
      p50 1549 -> 1276 ms. Stats expuestas en la pestaña
      **Dashboard** (expander "Cache de embeddings").

## Iteración #4 — Operación

- [x] Dashboard interno con pandas sobre `queries.db` (pestaña
      **Dashboard** en Streamlit, modulo `src/dashboard.py`). Muestra
      6 KPIs (total, últimos 7 días, p50/p95, tasa "no encuentro",
      tasa con warnings), serie temporal de consultas por día, ranking
      de documentos más citados, latencia por modelo LLM, top de
      preguntas frecuentes y tabla de warnings faithfulness
      registrados.
- [x] Sincronizacion incremental de `docs/` → ChromaDB
      (`src/corpus_sync.py` + `scripts/sync_docs.py`). Manifest JSON
      con hashes SHA-256; solo reindexa la delta (nuevos / modificados
      / eliminados). Streamlit detecta cambios al arrancar y ofrece un
      boton de sincronizacion.
- [x] Watcher continuo (`watchdog==4.0.2`) que llama a `sync()`
      automaticamente al detectar cambios en `docs/`. Script
      `scripts/watch_docs.py` con debouncing de 2 s para absorber
      rafagas de eventos (copia de varios archivos al tiempo).
      Se corre `python -m scripts.watch_docs` en una terminal aparte
      de Streamlit y queda activo hasta `Ctrl+C`.

---

## Iteración #5 — Asistente de tareas especializadas

Nueva pestaña **Asistente** con tres sub-secciones que aplican prompts
especializados sobre texto o documentos del corpus. Cada sub-seccion
compara dos tecnicas — **zero-shot** y **chain-of-thought** — lado a
lado (`st.columns(2)`). Ejecuciones **efimeras**: no se persisten en
`queries.db` ni aparecen en Historial/Dashboard.

Tareas:
1. **Clasificación por tipo de riesgo** (multi-etiqueta con peso) sobre
   texto pegado por el usuario. Categorías: crédito, mercado,
   operacional, liquidez, ciberseguridad.
2. **Extracción de información clave** (4 categorías: límites
   regulatorios, indicadores de riesgo, decisiones del comité, fechas
   críticas) sobre uno o más documentos de `docs/` vía **map-reduce**
   usando los chunks ya indexados en ChromaDB.
3. **Resumen ejecutivo** estructurado (decisiones / riesgos
   identificados / acciones pendientes) sobre documentos de `docs/`
   cuyo nombre (lowercase) contenga `acta`. Map-reduce idéntico al
   ejercicio 2.

### Fase A — Modulo `src/assistant/`
- [x] `prompts.py`: plantillas zero-shot y CoT para las 3 tareas
      (6 pares system+user). Todas exigen JSON estricto al cierre.
- [x] `llm_utils.py`: cliente Ollama + `parse_json_response` tolerante
      (acepta objeto directo, array directo, envuelto en prosa, code
      block markdown). El parser prioriza el delimitador que aparece
      primero en el texto para no capturar solo el primer `{}` de un
      array que empieza antes.
- [x] `classifier.py`: `classify(text, technique)` con `_sanitize`
      (etiquetas válidas, pesos ≤ 1, normaliza suma > 1) + `_extract_list`
      (desempaqueta si el LLM envuelve en dict).
- [x] `extractor.py`: `extract(doc_names, technique, progress_cb)`.
      Map-reduce sobre chunks de ChromaDB con fallback a merge local
      si el reduce LLM vuelve vacío.
- [x] `summarizer.py`: `summarize(doc_name, technique, progress_cb)`.
      Filtro de actas via `list_acta_docs()` (substring "acta" en
      lowercase).
- [x] `corpus_utils.py`: listado de docs soportados y chunks desde
      ChromaDB con fallback a disco.

### Fase B — UI Streamlit (`app.py`)
- [x] Nueva pestaña **Asistente** entre Dashboard e Historial.
- [x] `src/assistant/ui.py` con 3 sub-secciones via `st.radio`
      horizontal.
- [x] **Clasificación**: textarea + botón → 2 columnas con
      `st.dataframe` + `st.bar_chart` por categoría.
- [x] **Info clave**: `st.multiselect` sobre `docs/` + botón → 2
      columnas con las 4 categorías como expanders.
- [x] **Resumen ejecutivo**: `st.multiselect` sobre `acta_docs()` →
      2 columnas con los 3 campos en markdown.
- [x] `st.progress` con texto `[doc i/N] chunk j/M`, ejecución
      secuencial (zero-shot → CoT).
- [x] Latencia al pie de cada columna + `st.download_button`
      "Exportar comparación (markdown)".
- [x] Smoke-test con `streamlit run app.py --headless`: arranca sin
      errores y responde `200 OK` en `/_stcore/health`.

### Fase C — Evaluación comparativa
- [x] `eval/assistant_ground_truth.json`: 12 casos de clasificación
      (L1-L5 por riesgo, incluye compuestas multi-etiqueta), 2 casos
      de extracción (doc_02 + doc_05) y 1 de resumen (doc_05_acta).
- [x] `scripts/eval_assistant.py`: corre las 3 tareas x 2 técnicas,
      calcula F1 micro-promedio (clasificación) y cobertura de
      keywords (extracción/resumen), latencia total y por caso.
      Reporte en `eval/assistant_report.json`.
- [x] Resultados (2026-04-16):

  | Tarea | Zero-shot | CoT | Ganador |
  |---|---|---|---|
  | Clasificación | F1=**0.737** (P=0.609, R=0.933) | F1=0.722 (P=0.619, R=0.867) | Zero-shot |
  | Extracción | cov=0.625, lat=**83 s** | cov=0.625, lat=196 s | Zero-shot (por latencia) |
  | Resumen | cov=**0.833**, lat=**55 s** | cov=0.500, lat=119 s | Zero-shot |

  - **Zero-shot** gana las 3 tareas. Tiende a sobre-etiquetar
    (agrega "riesgo_credito"/"riesgo_operacional" de más) pero nunca
    devuelve vacío. Su recall dominante (0.933) es valioso en
    clasificación multi-etiqueta.
  - **CoT** es más preciso en aciertos exactos (5/12 vs 3/12 F1=1.0),
    pero **frágil al parseo**: 2/12 casos de clasificación devolvieron
    `[]` porque el razonamiento envolvió el JSON en dict no parseable.
    En extracción iguala cobertura pero a 2.4x la latencia. En
    resumen pierde 33 % de cobertura.
  - **Default en la UI**: ambas técnicas corren lado a lado (el
    usuario elige). Para automatización batch, usar zero-shot.

### Fase D — Tests unitarios
- [x] `tests/test_assistant_parser.py` (16 tests): `_balanced_extract`
      + `parse_json_response`.
- [x] `tests/test_assistant_classifier.py` (17 tests):
      `_extract_list` + `_sanitize`.
- [x] `tests/test_assistant_extractor.py` (13 tests): `_sanitize`
      + `_merge_partials`.
- [x] `tests/test_assistant_summarizer.py` (9 tests): sanitize +
      merge sobre los 3 campos.
- [x] `tests/test_assistant_corpus.py` (9 tests): listado de docs y
      filtro "acta" con `docs_dir` aislado.
- [x] Suite completa: **171/171 passing** en 7.4 s.

### Fase E — Documentación
- [x] `test_unitarios.md` actualizado con los 64 nuevos tests.
- [x] Resultados comparativos documentados en Fase C arriba.

---

## Iteración #6 — Calidad y limpieza de código (2026-04-16)

Barrido sistemático para dejar el proyecto con PEP 8 al 100 %,
docstrings en todas las funciones, modularidad sin módulos gigantes
y tests que cubren también los módulos que antes quedaban fuera.

### Fase A — PEP 8 al 100 %
- [x] Instalar `flake8==7.3.0` y correr sobre `src/`, `scripts/`,
      `tests/`, `app.py` con `--max-line-length=120`. Baseline inicial:
      13 issues (`E402`, `F401`, `E231`, `E203`).
- [x] Corregir los 13 issues: imports no usados removidos, `# noqa: E402`
      en los scripts que reasignan `sys.stdout` a UTF-8 antes de
      importar, espacios normalizados.
- [x] flake8 final: **0 issues** en las 4 carpetas.

### Fase B — Comentarios inline en lógica compleja
- [x] `retriever._hybrid_rank`: bloque RRF documentado paso a paso
      (por qué `k=60`, por qué sumar y no promediar, qué aporta cada
      ranking).
- [x] `ingestion._extract_page_tables_and_text`: geometría de bboxes
      de pdfplumber + contención estricta explicada inline.
- [x] `llm_utils.parse_json_response`: trailing commas y elección de
      delimitador primario documentados inline.

### Fase C — Modularidad (partición de `ui.py`)
- [x] `src/assistant/ui.py` (371 líneas, 3 responsabilidades) se
      dividió en 5 módulos:
  - `ui.py` (dispatcher, 30 L): selector de tarea + delegación.
  - `ui_common.py`: constantes compartidas (TECHNIQUE_LABELS + dicts
    pretty).
  - `ui_classification.py`: render + markdown de clasificación.
  - `ui_extraction.py`: render + markdown de extracción.
  - `ui_summary.py`: render + markdown de resumen ejecutivo.
- [x] `app.py` no requiere cambios (sigue llamando a
      `assistant_ui.render()`).

### Fase D — Cobertura de tests ampliada
- [x] Nuevos tests unitarios para los 7 módulos sin cobertura:
  - `test_config.py` (6 tests): Settings frozen, paths absolutos,
    overrides por env var.
  - `test_generator.py` (11 tests): `_format_context`, `_build_messages`,
    fast-paths sin LLM.
  - `test_vectorstore.py` (9 tests): upsert/query contra ChromaDB
    real aislado en `tmp_path`.
  - `test_reranker.py` (10 tests): `_minmax`, bypass disabled, blending
    con CrossEncoder mockeado (numpy stub).
  - `test_embeddings.py` (11 tests): cache LRU, dedup en batch,
    stats coherentes.
  - `test_pipeline.py` (10 tests): `ask`, `ask_stream`, `warmup` con
    todas las dependencias mockeadas.
  - `test_dashboard.py` (12 tests): `_normalize`, `_source_frequencies`,
    `_top_questions`, `_load_dataframe`.
- [x] Suite completa: **234/234 passing** en ~5.6 s.

### Fase E — Docstrings al 100 %
- [x] Audit con walker AST: 92 funciones sin docstring detectadas
      (públicas y privadas en `src/`, `scripts/` y `app.py`).
- [x] Agregadas las 92 docstrings de una línea, explicando el *qué*
      cuando no es obvio desde la firma.
- [x] Audit post-cambio: **0 funciones sin docstring**.

### Fase F — Documentación
- [x] `README.md` — conteo de tests 234 y estructura de assistant
      actualizada con los 4 nuevos `ui_*.py`.
- [x] `DESIGN.md` — nueva sección §10 "Calidad de código" con las
      tres invariantes (PEP 8, docstrings, tests) y subsección en §8
      explicando la partición del asistente.
- [x] `PLAN.md` — esta iteración.
- [x] `test_unitarios.md` — 7 módulos nuevos documentados + update
      de la sección "Cobertura no incluida".

---

## Deuda técnica conocida
- OCR dependiente de binarios externos (Tesseract, Poppler). Documentar
  bien su instalación y validar en máquina limpia.
- El `section_hint` por regex captura los patrones comunes en español
  regulatorio; documentos con maquetación atípica pueden caer en la
  herencia. Sustituirlo eventualmente por un parser estructural de
  encabezados (por ejemplo, aprovechando la jerarquia de headings que
  pdfplumber expone).
- ~~No hay tests unitarios (pytest)~~ — resuelto 2026-04-16. Suite de
  **234 tests** en `tests/` cubriendo chunking, ingestion,
  faithfulness, query_decomposer, retriever, logger_db, los 5 módulos
  del asistente, y en la iteración #6 se añadió cobertura de config,
  embeddings, generator, vectorstore, reranker, pipeline y dashboard.
  pytest es dev-only: `pip install -r requirements-dev.txt`. Correr
  con `python -m pytest`. Runtime ~5.6 s.
- Tablas en PDFs: actualmente solo se detectan tablas markdown
  (`| ... |`). Para PDFs reales con tablas nativas habrá que integrar
  `pdfplumber.extract_tables()`.
- Ollama no está en PATH del shell Git Bash; funciona desde PowerShell
  y la API Python conecta bien por HTTP a `localhost:11434`.

---

## Notas de iteración

### #1 (2026-04-15)
- Se eligió stack 100 % local (Ollama + Chroma) por sensibilidad del
  corpus regulatorio. Registrado en `DESIGN.md` §1.
- Ollama instalado via `winget`, modelos descargados (`llama3` 4.7 GB,
  `nomic-embed-text` 274 MB). Descarga tomó ~50 min.
- Los 5 documentos de prueba resultaron ser `.txt` (no PDF). Se amplió
  `ingestion.py` para soportar `.txt` y `.md` además de `.pdf`.
- Indexación exitosa: 5 docs → 36 chunks en ChromaDB en 10 seg.
- `chromadb==0.5.23` no compilaba en Python 3.13 (falta MSVC para
  `chroma-hnswlib`). Solucionado subiendo a `chromadb==1.5.7`.
- **Pendiente para mañana**: probar Streamlit con preguntas reales,
  evaluar calidad de respuestas/citas, ajustar parámetros si hace falta.

### #1.5 (2026-04-16) — Fix de recuperación
- Al probar la pregunta *"¿Qué metodología se usa para calcular el
  VaR?"* el sistema respondió "no encuentro la información" a pesar
  de que el documento `doc_04_metodologia_var.txt` la contiene
  textualmente. Diagnóstico: `nomic-embed-text` rankeaba el chunk
  con la respuesta literal en posición 23 de 36.
- **Fix 1**: migración a `bge-m3` (multilingüe, 1024 dims). El chunk
  subió a posición 12 — insuficiente con `TOP_K=5`.
- **Fix 2**: implementación de retriever híbrido BM25 + vectorial con
  Reciprocal Rank Fusion en `src/retriever.py`. Stopwords españolas +
  filtro de acentos en el tokenizer para evitar falsos positivos con
  palabras genéricas ("metodología", "para"). El chunk llegó a
  posición 4 del top 5.
- Resultado: la respuesta ahora cita correctamente "Simulación
  Histórica Filtrada (FHS)" y menciona el modelo GARCH(1,1).
- Se quitó el panel redundante "Fuentes recuperadas" de la pestaña
  Consulta; las citas del LLM (sección **Fuentes:** de la respuesta)
  son suficientes. En Historial se conservan por valor de auditoría.
- `.env.example` y `requirements.txt` actualizados con `bge-m3` y
  `rank-bm25==0.2.2`. Instrucciones de arranque de Streamlit añadidas
  a `README.md` §3.3.

### #1.10 (2026-04-16) — Comparacion LLMs (cierra #6)
- `llama3` vs `llama3.1:8b` con el mismo eval harness (retriever,
  prompt, contexto, todo identico salvo el modelo LLM).
- Resultado: `llama3` mantiene 100 % de accuracy; `llama3.1:8b` cae
  a 97 % (Q10 sintesis cross-doc pierde las cifras "COP 12,350 M" y
  "2.1 %" a pesar de tenerlas en contexto). Ademas llama3.1 es ~26 %
  mas lento en p50.
- Decision: mantener `LLM_MODEL=llama3` como default. El script
  `scripts/eval_compare.py` queda para futuras comparaciones cuando
  se pruebe mistral u otros modelos.

### #1.9 (2026-04-16) — Chunking robusto + tablas + top_k 7
- `src/chunking.py` rediseñado:
  - Detección de tablas markdown antes del splitter; cada tabla va a
    un chunk dedicado con `is_table=true` en metadata.
  - Regex de headings ampliada: además de `Artículo`, `Capítulo`,
    `Sección`, ahora captura `Parte`, `Paso N:`, jerárquicos 1.2.3,
    y headings ALL-CAPS con ≥2 tokens.
  - Umbral de posición: `section_hint` solo toma heading "propio" si
    aparece en el primer tercio del chunk. En caso contrario, hereda
    el último heading visto en el documento.
  - Al propagar herencia entre chunks se usa el *último* heading del
    chunk actual, no el primero — así un chunk que contiene dos
    headings ("1.7 ..." y "2. INDICADORES ...") transmite "2." hacia
    adelante y alinea correctamente la siguiente sección/tabla.
- Resultado: **100 % de chunks con section_hint** (antes ~60 %). La
  tabla de KRIs del informe operacional ahora es un único chunk con
  `section_hint="2. INDICADORES CLAVE DE RIESGO (KRIs)"`.
- `src/pipeline.py`: al descomponer una pregunta compuesta se incluye
  la pregunta original junto con las subqueries en la fusión multi-RRF.
  Las subqueries a veces pierden contexto ("¿Qué decidió el comité al
  respecto?"); la pregunta original ancla ese contexto.
- `TOP_K` subido de 5 a 7. Con el chunking nuevo el chunk con "DECISIÓN
  ...28%..." de Q11 caía a posición 6 en el ranking fusionado. Dos
  chunks de margen cierran este tipo de casos en el límite sin impacto
  significativo de latencia (+0.2 s p50).
- Re-ejecución del harness: 12/12 100 %/100 %/100 %, p50=1.5 s, p95=3.8 s.

### #1.8 (2026-04-16) — Reranker integrado pero off por defecto
- Se integro `BAAI/bge-reranker-v2-m3` via `sentence-transformers`
  (arrastra `torch 2.11` CPU ~500 MB + `transformers`). Codigo en
  `src/reranker.py`, orquestado por `pipeline.ask()` cuando
  `RERANKER_ENABLED=true`.
- Re-ejecucion del harness con reranker activo: `source_recall@5`
  100 %, `answer_accuracy` **91.7 %** (Q4 regression — el chunk de
  "METODOLOGIA: SIMULACION HISTORICA FILTRADA" fue demotado a favor de
  los "Paso 5" y "Paso 7" que tienen prosa mas fluida). Latencia p50
  salto de 1.2 s a 10.5 s.
- Diagnostico: con solo 36 chunks el hibrido ya satura la calidad
  alcanzable; el reranker agrega ruido. Se espera inversion en corpus
  de cientos/miles de chunks donde el hibrido tendra mas falsos
  positivos que el reranker puede filtrar.
- Decision: dejar el codigo listo pero `RERANKER_ENABLED=false` por
  defecto. Reactivar cuando el corpus crezca. Pendiente: probar
  blended scoring (`alpha * rerank + (1-alpha) * rrf`) antes de
  reactivar.

### #1.7 (2026-04-16) — Eval harness + guardrails
- `eval/ground_truth.jsonl` con las 12 preguntas de `EVALUACION.md`
  convertidas a casos de prueba (question, expected_sources, must_contain).
- `scripts/eval.py` ejecuta el batch, calcula `source_recall@5`,
  `answer_accuracy`, `faithfulness`, latencia p50/p95 y persiste
  `eval/report.json`. **Baseline actual: 12/12 en las 3 metricas**,
  p50=1.2 s, p95=3.3 s.
- `src/faithfulness.py` inspecciona la respuesta post-generacion y
  marca como `faithfulness_warnings` las cifras, articulos o fechas
  que no aparecen textualmente en el contexto recuperado. La UI los
  muestra en un expander plegable cuando existen. En el baseline
  actual: 0 warnings (0 falsos positivos despues de afinar el regex).
- `pipeline.warmup()` pre-carga los modelos en Ollama. La app Streamlit
  lo invoca al arrancar via `@st.cache_resource`; la primera consulta
  ya no paga los ~10 s de cold start.

### #1.6 (2026-04-16) — Evaluación sistemática y fix de fallos
- Se diseñó una batería de **12 preguntas** cubriendo los 5 documentos
  en 5 niveles de complejidad (L1 lookup → L5 inferencia). Script en
  `scripts/eval_questions.py`. Documento completo: `EVALUACION.md`.
- Resultado inicial: **10/12 (83 %)**. Dos fallos:
  - Q10 (L4): artifact textual al concatenar encabezados de fragmentos.
  - Q11 (L4): pregunta compuesta → respondía solo la primera parte y
    declaraba "no encuentro" la segunda, a pesar de tener los chunks.
- **Fix 1 — Prompt hardening** (`src/generator.py`): 5 reglas
  explícitas (sin preámbulos, formato de citas estricto, respuesta
  obligatoria a todas las partes de preguntas compuestas, no
  concatenar encabezados, no inventar cifras). Separadores de
  fragmentos más visibles en `_format_context`.
- **Fix 2 — Query decomposition** (`src/query_decomposer.py`, nuevo):
  detección de preguntas compuestas por heurística (conjunción + wh-word,
  múltiples `?`, "además") + descomposición vía LLM cuando aplica.
  Validación posterior: solo se aceptan subqueries que terminen con
  `?` o empiecen con `¿`, descartando alucinaciones como
  *"No hay un límite..."* que el LLM añadía como filler.
- **Fix 3 — Retriever multi-query** (`retriever.hybrid_query_multi`):
  fusiona los rankings de cada subquery con un segundo RRF. Garantiza
  representación en el top-k para cada parte de la pregunta.
- Re-ejecución de Q10 y Q11: ambas correctas. **Precisión final:
  12/12 (100 %)**.
