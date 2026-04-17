# Decisiones de diseño — Sistema RAG para documentos de riesgo

Este documento justifica cada componente del pipeline. Está pensado
para ser leído por otro analista o ingeniero que quiera entender o
auditar las decisiones tomadas.

---

## 0. Contexto y objetivo

El equipo de riesgos consulta manualmente un corpus de regulaciones,
políticas internas, informes y actas. La revisión manual es lenta y
propensa a omisiones. El objetivo es un sistema **RAG (Retrieval
Augmented Generation)** que:

1. Indexe todos los documentos a la vez.
2. Responda preguntas en lenguaje natural citando **documento +
   página + sección** exacta.
3. Registre el histórico de consultas (30 días) para auditoría y
   mejora.

Restricción implícita del dominio: la información es **sensible**
(regulación, políticas). Por eso todo el pipeline corre **100 % local**.

---

## 1. Stack tecnológico

| Componente      | Elegido                        | Motivo                                                                 |
|-----------------|-------------------------------|------------------------------------------------------------------------|
| LLM             | `llama3` vía **Ollama**        | Local, sin coste, sin enviar documentos a terceros.                    |
| Embeddings      | `bge-m3` vía Ollama            | Multilingüe de alta calidad en español, 1024 dims.                     |
| Recuperación    | BM25 (`rank-bm25`) + vectorial | Híbrido con Reciprocal Rank Fusion: precisión léxica + semántica.      |
| Vector DB       | **ChromaDB** persistente       | Cero infra, un solo archivo, metadata rica para citar.                 |
| Parser PDF      | `pdfplumber` + `pytesseract`   | Nativo + fallback OCR para PDFs escaneados.                            |
| Chunker         | `RecursiveCharacterTextSplitter` (LangChain) | Respeta párrafos/oraciones antes de romper.                 |
| UI              | **Streamlit**                  | Rápida para analistas no técnicos, historial visible.                  |
| Log de consultas| **SQLite**                     | Auditable, single-file, soporta TTL de 30 días.                        |
| Reranker (opcional)| `BAAI/bge-reranker-v2-m3` via `sentence-transformers` | Cross-encoder blended con RRF. Off por defecto: útil solo en corpus grandes. |
| File watcher    | `watchdog`                     | Reindex incremental automático al cambiar `docs/`.                     |

Se descartaron alternativas que habrían sido razonables en otro contexto:
- **OpenAI / Azure OpenAI**: mejor calidad de razonamiento pero implica
  enviar fragmentos regulatorios a un proveedor externo. Descartado por
  el perfil de confidencialidad del corpus.
- **FAISS**: más rápido pero no persiste metadata estructurada.
- **CLI / FastAPI**: menos fricción para un analista no técnico que
  Streamlit, que es el usuario objetivo.

---

## 2. Ingesta y extracción de texto (`src/ingestion.py`)

### Problema
Los PDFs regulatorios suelen mezclar páginas **digitales** (texto
seleccionable) con páginas **escaneadas** (imágenes). Un solo parser no
funciona.

### Decisión — extracción híbrida con fallback
1. Intentar siempre primero con `pdfplumber` (rápido y limpio).
2. Si una página devuelve menos de **40 caracteres** de texto útil, se
   trata como escaneada y se renderiza a imagen con `pdf2image` +
   Poppler, para luego pasarla por **Tesseract** (`lang=spa+eng`).
3. Se guarda un campo `extraction = native | ocr` en los metadatos para
   poder auditar después la calidad del texto fuente.

### Por qué 40 caracteres
Es suficiente para descartar páginas vacías o con solo pie de página,
sin disparar OCR innecesario en separadores de capítulo. Es un umbral
ajustable en `ingestion.MIN_CHARS`.

### Metadatos generados por página
`{ source (nombre archivo), page (nº), text, extraction }` — es la
unidad mínima para poder citar.

---

## 3. Fragmentación (`src/chunking.py`)

### Decisión
`RecursiveCharacterTextSplitter` con:
- `chunk_size = 900` caracteres
- `chunk_overlap = 150` (≈17 %)
- separadores `["\n\n", "\n", ". ", " ", ""]`

Antes de llamar al splitter se extraen **tablas markdown** (cualquier
bloque de ≥2 líneas consecutivas con `|`) y se emiten como chunks
dedicados con `is_table=true`. Evita que la tabla quede partida entre
varios chunks y permite al LLM ver todas las filas juntas (caso típico:
la tabla de KRIs del informe trimestral).

### Por qué estos tamaños
- **900** deja espacio cómodo para 4–6 chunks en el contexto de
  `llama3` (8k tokens) sin saturar, dejando margen al prompt y la
  respuesta.
- **150 de solape** evita que un concepto que vive a caballo entre dos
  chunks pierda contexto (p. ej. la definición de un límite en un
  párrafo y su valor numérico en el siguiente).
- Los **separadores en orden** hacen que se rompa primero por párrafo,
  luego por oración, y solo al final por palabra — así los chunks
  tienden a ser semánticamente coherentes.

### Section hint — pieza clave para citar
De cada chunk se extrae con regex el primer encabezado detectado
(`Artículo N`, `Capítulo X`, `Sección Y`, `Parte Z`, `Paso N:`,
numerado jerárquico `1.2.3 Título`, o cualquier heading ALL-CAPS con
≥2 palabras). Se guarda en `section_hint`, permitiendo que la respuesta
cite *"Política de Crédito, página 12, Artículo 5"* en vez de solo
*"página 12"*.

**Reglas finas para evitar etiquetas equivocadas:**

1. **Umbral de posición**: el heading propio del chunk solo cuenta si
   aparece dentro del primer tercio del texto. Si está cerca del final,
   indica que el chunk termina entrando en una sección nueva pero su
   contenido mayoritario es de la anterior — se prefiere heredar.
2. **Herencia = último heading visto**: cuando un chunk no tiene
   heading propio (o éste es tardío), hereda el *último* heading que
   apareció en los chunks previos del mismo documento. Si un chunk
   contiene dos headings ("1.7 …" y "2. INDICADORES …"), el que se
   propaga hacia adelante es el último, no el primero. Así los chunks
   siguientes se alinean con la sección correcta.

---

## 4. Embeddings (`src/embeddings.py`)

### Decisión — `bge-m3` local
- Modelo multilingüe de BAAI entrenado con foco explícito en lenguas
  no inglesas (incluido español).
- 1024 dims → ligeramente mayor coste de almacenamiento que un modelo
  de 768, compensado por una calidad semántica sustancialmente mejor.
- Se ejecuta en la misma instancia de Ollama que el LLM, simplificando
  el despliegue.

### Por qué no `nomic-embed-text`
Se probó en la iteración #1. Con preguntas en español regulatorio
rankeaba chunks semánticamente distantes por encima del chunk que
contenía literalmente la respuesta (por ejemplo: "3. METODOLOGÍA:
SIMULACIÓN HISTÓRICA FILTRADA" quedaba en posición 23 de 36 para la
pregunta *"¿Qué metodología se usa para calcular el VaR?"*). La
migración a `bge-m3` subió ese chunk a posición 12; la recuperación
híbrida (§6) lo llevó al top 5.

### Robustez
`embed_one` está decorado con `tenacity` (3 reintentos, backoff
exponencial) porque Ollama puede devolver errores transitorios al
cargar el modelo la primera vez.

### Cache LRU
Sobre `embed_one` se envuelve un `functools.lru_cache(maxsize=2048)`
con clave `(texto, modelo)`. Las preguntas repetidas del usuario y los
subqueries que se repiten entre consultas no vuelven a pegar a Ollama.
Al cambiar `EMBEDDING_MODEL` las entradas con el modelo anterior ya no
generan hits (clave incluye el modelo) y se desalojan conforme llenan
nuevas entradas. En eval batch de 12 preguntas: segunda corrida -22 %
wall time respecto a la primera. Stats (`hits`, `misses`, `hit_rate`,
`size`) expuestas en el dashboard.

> **Nota para futuras iteraciones**: si el volumen de chunks crece
> mucho, vale la pena paralelizar los embeddings con un pool
> asíncrono o usar batching nativo.

---

## 5. Vector store (`src/vectorstore.py`)

### Decisión — una sola colección, distancia coseno
- Una única colección `riesgos_corpus` contiene los chunks de **todos**
  los documentos. Esto satisface directamente el requisito del usuario
  de *"consultar todos los documentos a la vez"*.
- Distancia **coseno** (`hnsw:space=cosine`) porque los embeddings de
  `bge-m3` no están garantizados como unitarios y el coseno es robusto
  a diferencias de norma.
- `id` del chunk = `"{archivo}::p{página}::c{idx}"` → permite upsert
  idempotente: reindexar no duplica, sobrescribe.
- Metadata indexable: `source`, `page`, `chunk_index`, `extraction`,
  `section_hint`, `is_table`. El flag `is_table` permite filtros futuros
  ("dame solo las tablas de doc_03") y distinguir en el dashboard los
  chunks que vienen de tablas markdown.

---

## 6. Recuperación y generación (`src/retriever.py`, `query_decomposer.py`, `generator.py`, `pipeline.py`)

### 6.1 Recuperación híbrida (BM25 + vectorial con RRF)
El retriever combina dos señales independientes:

1. **BM25** (léxico) sobre los tokens de cada chunk. Aporta precisión
   cuando la pregunta contiene siglas o términos técnicos exactos
   (*VaR*, *GARCH*, *Circular 034*, *SFC*).
2. **Búsqueda vectorial** con `bge-m3` en ChromaDB. Aporta coincidencia
   semántica cuando el usuario formula la pregunta con sinónimos o
   paráfrasis.

Los dos rankings se fusionan con **Reciprocal Rank Fusion** (Cormack,
Clarke, Büttcher 2009):

```
score(d) = Σ_i  1 / (k + rank_i(d))
```

con `k=60`. RRF es insensible a las escalas absolutas de los scores —
no hay que normalizar — y funciona muy bien cuando los dos retrievers
discrepan sobre documentos mid-rank.

### Tokenizer para BM25
Lowercase + regex `\w+` + eliminación de stopwords en español + filtro
de tokens de un carácter. Sin eliminación de stopwords palabras como
"qué", "se", "de", "para" dominaban el score y ensuciaban el ranking.

### Top-k
Top-k por defecto **7** (configurable en `TOP_K`). Se subió desde 5 tras
detectar en evaluación que algunas preguntas compuestas colocaban el
chunk relevante justo en posición 6 del ranking fusionado (p. ej. la
decisión del comité sobre renta variable internacional). Dos chunks
extra dan un margen razonable sin inflar el contexto de forma
significativa — el LLM recibe ~6 KB adicionales en el peor caso.

### 6.2 Descomposición de preguntas compuestas
Las preguntas como *"¿Cuál es el límite X y qué decidió el comité al
respecto?"* combinan dos preguntas independientes. Si se recuperan
chunks únicamente con los mejores scores para la pregunta original, es
frecuente que el contexto quede dominado por una sola de las partes,
llevando al LLM a responder solo esa y declarar "no encuentro" la otra
(regresión observada en `EVALUACION.md` §4.1).

`src/query_decomposer.py` implementa un pipeline en dos pasos:

1. **Heurística barata**: regex que detecta *conjunción + wh-word*
   (`y qué`, `y cómo`, `y cuánto`, etc.), signos `?` múltiples, y
   marcadores como "además". Las preguntas que no casan pasan directo
   al retriever sin overhead.
2. **Descomposición vía LLM**: solo cuando la heurística marca la
   pregunta como compuesta. El LLM devuelve un array JSON con las
   subpreguntas. Se filtra manteniendo solo items que parezcan
   preguntas (`?` final o `¿` inicial) para evitar que el LLM añada
   afirmaciones espurias.

Las subqueries se pasan a `retriever.hybrid_query_multi`, que corre el
híbrido para cada una y fusiona los rankings con un **segundo RRF**
sobre los rankings por subquery. Así cada parte de la pregunta tiene
garantía de representación en el top-k final.

### 6.3 Prompt del sistema (`src/generator.py`)
El system prompt impone cinco reglas numeradas:

1. **Sin preámbulos** — no repetir la pregunta, no prefijar con "La
   respuesta es" ni "Según los fragmentos".
2. **Preguntas compuestas** — responder todas las partes; declarar "no
   encuentro información" solo si *ninguna* parte está en el contexto.
3. **Formato de citas estricto** — `[archivo.ext, pagina X, seccion Y]`
   al final en una sección que comienza con `Fuentes:`.
4. **No inventar cifras ni artículos** — todo dato numérico o
   referencia a artículo debe aparecer textualmente en el contexto.
5. **No concatenar encabezados** — tratar cada `[Fragmento i]` como una
   unidad aislada.

`temperature=0.1` para respuestas deterministas y auditables.

Los bloques de contexto se separan con un delimitador visual
(`============================`) entre fragmentos para reforzar la
regla 5 a nivel de tokens, no solo de instrucción.

### Por qué citar es obligatorio
En un dominio regulatorio, una respuesta sin fuente no es accionable.
Además mitiga alucinaciones: si el modelo no tiene de dónde citar, el
propio prompt lo empuja a declarar la ausencia.

### 6.4 Faithfulness guardrail (`src/faithfulness.py`)
Tras cada generación, `faithfulness.check(answer, context)` busca
cifras, artículos y fechas citados en la respuesta y verifica que
aparezcan literalmente (normalizando acentos + espaciado) en los
fragmentos recuperados. Los claims que no aparecen se devuelven como
`faithfulness_warnings` y se muestran en la UI dentro de un expander
de advertencia. El prompt puede mitigar las alucinaciones pero no
eliminarlas — el guardrail es la última línea de defensa antes de
que el usuario tome una decisión regulatoria a partir de la respuesta.

Regex cubre: números (con `%`, `.`, `,` como separadores),
`Artículo N`, fechas en formato "dd de mes de yyyy". Se filtran
números triviales (1–10, 100). Warnings = 0 en el baseline actual
(12/12 preguntas).

---

## 7. Log de consultas (`src/logger_db.py`)

### Decisión — SQLite con TTL de 30 días
Tabla `queries` con:

```
id, ts (UTC ISO), question, answer,
sources_json, latency_ms, llm_model, embedding_model
```

- `purge_old()` borra registros con más de `LOG_RETENTION_DAYS` días.
  Se invoca automáticamente en cada `ask()` y también manualmente
  desde el sidebar.
- Se guarda el modelo usado para poder comparar respuestas entre
  versiones del LLM/embeddings.

### Por qué SQLite
Single-file, transaccional, auditable, exportable a pandas en una
línea. Alternativa JSON-lines era más simple pero no permite filtrar
por fecha eficientemente.

### Usos previstos del log
- Detectar preguntas frecuentes → candidatas a FAQ.
- Medir latencia por modelo.
- Detectar respuestas *"No encuentro esta información"* → pistas de
  gaps en el corpus.
- Auditar warnings de faithfulness (`warnings_json`) para encontrar
  patrones de alucinación por modelo o por tipo de consulta.

### Dashboard (`src/dashboard.py`)
La pestaña **Dashboard** de Streamlit consume `logger_db.all_queries()`
con pandas y renderiza 6 KPIs (total, últimos 7 días, p50/p95, tasa de
*"no encuentro"*, tasa de respuestas con warnings), una serie temporal
de volumen diario, el ranking de documentos más citados, la latencia
desagregada por modelo LLM, el top de preguntas frecuentes, y una
tabla desplegable con los warnings faithfulness acumulados. No depende
de bibliotecas de gráficos externas — usa `st.bar_chart` y
`st.dataframe` nativos.

---

## 8. Interfaz (`app.py`)

Streamlit con tres bloques:
1. **Sidebar**: métricas del corpus + botones de reindexar y purgar.
2. **Tab Consulta**: área de pregunta y respuesta. La respuesta se
   renderiza con **streaming** token-a-token (`st.write_stream` sobre
   `pipeline.ask_stream()`), lo que reduce el tiempo percibido a la
   latencia del primer token (~500–1500 ms) aunque la respuesta total
   siga tardando varios segundos. La sección **Fuentes:** la produce
   el LLM dentro del mismo stream — se eliminó la tabla separada que
   había antes (duplicaba información).
3. **Tab Historial (30 días)**: listado expandible de las últimas 200
   consultas. Aquí sí se conserva la tabla de fuentes con la distancia
   coseno, útil para auditoría retrospectiva.

### Estructura del Asistente (`src/assistant/ui*.py`)
La pestaña **Asistente** se divide en cuatro módulos para evitar que
`ui.py` concentre tres responsabilidades distintas: `ui.py` queda
como dispatcher (<35 líneas) que elige la sub-sección, y
`ui_classification.py`, `ui_extraction.py`, `ui_summary.py` implementan
cada tarea de forma independiente. Las constantes compartidas
(`TECHNIQUE_LABELS`, diccionarios *pretty*) viven en `ui_common.py` —
se importan desde los tres sub-módulos sin duplicación. Esta partición
permite tocar, por ejemplo, la clasificación sin recargar el código de
resumen o extracción.

### Warmup de modelos
`pipeline.warmup()` se invoca al arrancar la app via
`@st.cache_resource`: precarga `llama3`, `bge-m3` (y el reranker si
está habilitado) con una llamada trivial. Evita que la primera consulta
del usuario pague ~10 s de cold start.

---

## 9. Sincronización del corpus (`src/corpus_sync.py`)

### Problema
Un reindex completo cada vez que el corpus cambia es caro: con N
documentos de M páginas, se rehacen todos los embeddings. El usuario
normalmente solo agrega/quita un archivo a la vez.

### Decisión — manifest + delta
Se mantiene `data/manifest.json` con el SHA-256 de cada archivo ya
indexado. `scan_state()` compara contra el estado actual de `docs/`
y clasifica archivos en `unchanged | new | modified | deleted`.
`sync()` aplica solo la delta: borra los chunks del archivo
modificado/eliminado (via `coll.delete(where={"source": fname})`) y
reindexa los nuevos/modificados. Idempotente — una segunda llamada
sin cambios no hace nada.

**Bootstrap**: si el manifest no existe pero la colección tiene
chunks (escenario de migración), se genera el manifest a partir del
estado actual de `docs/` sin reindexar — evita duplicar chunks.

### Watcher continuo (`scripts/watch_docs.py`)
Usa `watchdog` para observar eventos de filesystem en `docs/`. Al
detectar un evento relevante (archivo `.pdf/.txt/.md` creado,
modificado, movido o borrado) programa una sync con debounce de 2 s:
si entran N eventos en ráfaga (copiar una carpeta entera), espera a
que cese la ráfaga y ejecuta **una** sync con todos los cambios.
Se corre en una terminal separada de Streamlit y queda activo hasta
`Ctrl+C`.

---

## 10. Calidad de código

La base de código se mantiene con tres invariantes auditables en CI o
pre-commit:

- **PEP 8 via flake8** (`--max-line-length=120`) — 0 issues en `src/`,
  `scripts/`, `tests/` y `app.py`. Las únicas excepciones intencionales
  son `# noqa: E402` en scripts de evaluación, donde `sys.stdout` se
  reasigna a UTF-8 antes de importar dependencias (necesario en Windows
  para caracteres acentuados en consola).
- **Docstrings en toda función** — auditado con un walker AST:
  0 funciones sin docstring en código productivo. Las docstrings son
  de una línea cuando la firma + nombre ya son descriptivos, y
  extendidas cuando hay invariantes o decisiones no obvias (RRF,
  bbox de tablas, parser JSON tolerante).
- **Tests unitarios** — 234 tests en `~5.6 s`; cualquier cambio en
  `src/` debe mantener la suite verde. Ver `test_unitarios.md`.

Comentarios **inline** se reservan para explicar el *por qué* de
lógica no evidente (fórmula RRF en `retriever._hybrid_rank`, geometría
de bboxes de tabla en `ingestion._extract_page_tables_and_text`,
recuperación de trailing commas en `llm_utils.parse_json_response`).
El *qué* lo explican los nombres; los comentarios que solo repiten la
firma se omiten deliberadamente.

---

## 11. Limitaciones conocidas

- Se comparó `llama3` vs `llama3.1:8b` con el harness de evaluación
  (`scripts/eval_compare.py`): `llama3` gana (100 % acc, p50 1.5 s vs
  `llama3.1:8b` con 97 % acc, p50 1.9 s — falla la síntesis cross-doc
  de Q10). Queda pendiente probar `mistral` u otros modelos no-Meta.
- OCR con Tesseract depende de la calidad del escaneo; documentos
  sellados o con manuscritos pueden fallar.
- El re-ranker cross-encoder (`BAAI/bge-reranker-v2-m3`) está integrado
  (`src/reranker.py`) pero **deshabilitado por defecto**. El reranker
  hace **blended scoring** cuando está activo: normaliza min-max ambas
  señales en el pool y combina `alpha * rerank_norm +
  (1-alpha) * rrf_norm`. Aun así, un sweep sobre este corpus de 37
  chunks con alpha en {0.0, 0.3, 0.5, 0.7, 1.0} mostró que **ningún
  valor de alpha > 0 supera al RRF puro**: el reranker introduce
  ruido incluso mezclado. La configuración queda lista para corpus
  con cientos/miles de chunks donde el RRF sí empieza a tener falsos
  positivos que el cross-encoder pueda filtrar. Se activa con
  `RERANKER_ENABLED=true` y se calibra con
  `scripts/eval_alpha_sweep.py`.
- El `section_hint` es regex y captura formatos comunes en español
  regulatorio; documentos con maquetación atípica pueden no tener
  hint.

Estas limitaciones están recogidas como siguientes pasos en `PLAN.md`.
