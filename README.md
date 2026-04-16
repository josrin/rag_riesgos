# RAG Riesgos — Consulta inteligente de documentos regulatorios

Sistema RAG **100 % local** para que el equipo de riesgos consulte en
lenguaje natural un corpus de regulaciones, políticas internas,
informes y actas, obteniendo respuestas con **cita exacta de documento,
página y sección**.

Stack: Python 3.13 · Ollama (`llama3` + `bge-m3`) · ChromaDB · BM25
(`rank-bm25`) · Streamlit · SQLite · cross-encoder opcional
(`bge-reranker-v2-m3`).

La recuperación es **híbrida**: combina búsqueda léxica (BM25) y
semántica (embeddings) fusionando los rankings con Reciprocal Rank
Fusion, lo que mejora la precisión sobre siglas y jerga regulatoria
(VaR, GARCH, SFC, etc.). Capas adicionales:

- **Descomposición de preguntas compuestas** — ruta los subqueries por
  separado y fusiona los rankings, evitando falsos "no encuentro".
- **Faithfulness guardrail** — detecta cifras/artículos/fechas que la
  respuesta cite y no aparezcan en los fragmentos recuperados.
- **Streaming de tokens** en la UI, **cache LRU** de embeddings,
  **warmup** automático de modelos al arrancar.
- **Sincronización incremental** de `docs/` (manual, one-shot CLI o
  watcher continuo) con manifest SHA-256.
- **Dashboard de observabilidad** sobre el log de consultas.
- **Eval harness** con 12 preguntas de ground truth y script de
  comparación entre LLMs.

- Decisiones de diseño → [`DESIGN.md`](DESIGN.md)
- Plan de ejecución y roadmap → [`PLAN.md`](PLAN.md)
- Evaluación con 12 preguntas y fallos detectados → [`EVALUACION.md`](EVALUACION.md)
- Suite de tests unitarios (171 tests, 7.4 s) → [`test_unitarios.md`](test_unitarios.md)

---

## 1. Requisitos previos (Windows 11)

| Software | Versión probada | Para qué | Enlace |
|---|---|---|---|
| **Python** | 3.13.12 | Runtime del pipeline | https://www.python.org/downloads/ |
| **Ollama** | 0.20.7 | Servidor local de LLM y embeddings | https://ollama.com/download/windows |
| **Tesseract-OCR** | 5.x (UB-Mannheim) | OCR de PDFs escaneados | https://github.com/UB-Mannheim/tesseract/wiki |
| **Poppler** | 24.08+ | Renderizar páginas PDF a imagen para OCR | https://github.com/oschwartz10612/poppler-windows/releases |

> **Nota:** Tesseract y Poppler solo son necesarios si tienes PDFs
> **escaneados** (imágenes). Si todos tus documentos son texto
> (`.txt`, `.md`) o PDFs digitales (texto seleccionable), puedes
> omitir ambos.

### 1.1 Instalar Ollama y descargar modelos

```powershell
# Opcion A: via winget (requiere admin)
winget install --id Ollama.Ollama -e

# Opcion B: descargar instalador desde https://ollama.com/download/windows

# Una vez instalado, descargar los modelos (~6 GB en total):
ollama pull llama3
ollama pull bge-m3

# Verificar:
ollama list
# Debe mostrar:
#   llama3:latest    ...  4.7 GB
#   bge-m3:latest    ...  1.2 GB
```

### 1.2 Instalar Tesseract-OCR (solo si hay PDFs escaneados)

1. Descargar el instalador `.exe` de
   [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
2. Durante la instalación, marcar **idioma Spanish** en "Additional
   language data".
3. Ruta por defecto: `C:\Program Files\Tesseract-OCR\tesseract.exe`.

### 1.3 Instalar Poppler (solo si hay PDFs escaneados)

1. Descargar el `.zip` de
   [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases).
2. Descomprimir en `C:\Program Files\poppler-<version>\`.
3. La ruta a anotar es la carpeta `bin` dentro, ej:
   `C:\Program Files\poppler-24.08.0\Library\bin`.

---

## 2. Instalación del proyecto

```powershell
cd "C:\Data Projects\rag-riesgos"

# Crear entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalar dependencias Python
pip install -r requirements.txt

# Crear archivo de configuracion
copy .env.example .env
```

### 2.1 Configurar `.env`

Edita `.env` y ajusta las rutas si difieren de las por defecto:

```ini
# Solo ajustar si las rutas de instalacion son distintas
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
POPPLER_PATH=C:\Program Files\poppler-24.08.0\Library\bin

# Parametros que puedes ajustar (los valores por defecto funcionan bien):
CHUNK_SIZE=900
CHUNK_OVERLAP=150
TOP_K=7
LOG_RETENTION_DAYS=30

# Cross-encoder reranker opcional (off por defecto). Solo aporta en
# corpus grandes; en corpus pequeños el hibrido RRF es mejor.
RERANKER_ENABLED=false
RERANKER_POOL_SIZE=20
RERANKER_ALPHA=0.5
```

---

## 3. Uso

### 3.1 Preparar documentos

Copia los documentos a indexar en la carpeta **`docs/`**.

**Formatos soportados:**
- `.pdf` — digitales (texto seleccionable) o escaneados (OCR automático)
- `.txt` / `.md` — texto plano

### 3.2 Indexar el corpus

**Primera indexación completa:**

```powershell
python -m scripts.index_documents
```

**Actualizaciones incrementales** (recomendado a partir de la 2ª vez):

```powershell
python -m scripts.sync_docs          # aplica la delta (+ nuevos / ~ modificados / - eliminados)
python -m scripts.sync_docs --dry-run # solo muestra que haria
```

El sincronizador compara `docs/` con `data/manifest.json` (hashes
SHA-256) y solo reindexa los archivos que cambiaron. Es idempotente.

**Watcher automático** (opcional, en otra terminal):

```powershell
python -m scripts.watch_docs
```

Mantiene la colección sincronizada en tiempo real: cualquier archivo
nuevo, modificado o borrado en `docs/` se aplica a ChromaDB tras ~2 s
de debounce. `Ctrl+C` para detener.

**Desde Streamlit:** al abrir la app, si detecta cambios pendientes en
`docs/` muestra un aviso en el sidebar con el botón **Sincronizar
(incremental)**. En el desplegable *Reindex completo* está el botón
para borrar y reconstruir la colección desde cero (solo usar si hay
corrupción).

Salida esperada:
```
Procesando doc_01_politica_riesgo_credito.txt
Procesando doc_02_circular_sfc_riesgo_mercado.txt
...
Indexacion completada:
  pages: 5
  chunks: 37
  elapsed_s: 4.0
```

### 3.3 Lanzar la aplicación Streamlit

**Flujo completo para cualquier persona con el repositorio:**

```powershell
# 1. Entrar al proyecto y activar el venv
cd "C:\Data Projects\rag-riesgos"
.\.venv\Scripts\Activate.ps1

# 2. (Si no lo has hecho) crear .env desde la plantilla
copy .env.example .env

# 3. Verificar que Ollama esta corriendo y con los modelos
ollama list
# Debe aparecer llama3:latest y bge-m3:latest.
# Si Ollama no responde, abrir otra terminal y ejecutar: ollama serve

# 4. (Solo la primera vez o tras agregar documentos a docs/)
python -m scripts.index_documents

# 5. Lanzar la UI
streamlit run app.py
```

Se abrirá automáticamente http://localhost:8501 en tu navegador. Para
detener la app pulsa `Ctrl+C` en la terminal.

> **Tip:** si arrancas Streamlit por primera vez te pedirá un correo;
> puedes dejarlo en blanco y pulsar Enter, o ejecutar
> `streamlit run app.py --server.headless=true` para saltar el prompt.

### 3.4 Consultar

Con la UI abierta haz preguntas en lenguaje natural. Ejemplos:

- *"¿Cuál es el límite de exposición por emisor individual?"*
- *"¿Qué metodología se usa para calcular el VaR?"*
- *"¿Qué decisiones tomó el comité de riesgos?"*

La respuesta incluye una sección **Fuentes:** citando archivo, página y
sección.

### 3.5 Pestañas de la UI

La UI tiene cuatro pestañas:

- **Consulta** — pregunta + respuesta con streaming y aviso de
  inconsistencias faithfulness cuando las haya.
- **Dashboard** — KPIs agregados sobre `queries.db` (total, últimos
  7 días, p50/p95 latencia, tasa "no encuentro", tasa de warnings),
  serie temporal de consultas por día, ranking de documentos más
  citados, latencia por modelo LLM, preguntas más frecuentes y tabla
  de warnings faithfulness detectados. Incluye expander con métricas
  del cache LRU de embeddings (hits, misses, hit rate).
- **Asistente** — tres tareas especializadas que corren **Zero-shot
  vs Chain-of-Thought lado a lado** (dos columnas por tarea, latencia
  por técnica, botón "Exportar comparación markdown"). Las
  ejecuciones son **efímeras**: no se registran en Historial ni
  Dashboard.
  - *Clasificación de riesgo* — pega un texto y se clasifica
    multi-etiqueta (crédito, mercado, operacional, liquidez,
    ciberseguridad) con peso por categoría (suma ≤ 1.0) y barra.
  - *Información clave* — multiselect de documentos de `docs/`;
    extrae límites regulatorios, indicadores de riesgo, decisiones
    del comité y fechas críticas vía **map-reduce** sobre los chunks
    indexados, con barra de progreso por chunk.
  - *Resumen ejecutivo* — multiselect restringido a actas
    (documentos cuyo nombre en lowercase contenga `acta`); produce
    resumen estructurado en tres campos: decisiones, riesgos
    identificados, acciones pendientes.
- **Historial (30 días)** — listado expandible de las últimas 200
  consultas con sus fuentes originales. Los registros con más de
  `LOG_RETENTION_DAYS` días se purgan automáticamente.

---

### 3.6 Evaluación automatizada

Las 12 preguntas de `eval/ground_truth.jsonl` miden el pipeline RAG
en 3 dimensiones:

```powershell
# Corrida única del pipeline RAG → eval/report.json
python -m scripts.eval

# Comparar varios LLMs sobre el mismo ground truth → eval/compare_report.json
python -m scripts.eval_compare llama3 llama3.1:8b mistral

# Sweep de alpha del reranker blended (solo con RERANKER_ENABLED=true)
python -m scripts.eval_alpha_sweep

# Eval comparativa del Asistente: Zero-shot vs Chain-of-Thought
# sobre las 3 tareas (12 casos de clasificación + 2 de extracción
# + 1 de resumen). → eval/assistant_report.json
python -m scripts.eval_assistant
```

Métricas del pipeline RAG: `source_recall@5` (el top-k contiene algún
chunk del/los documento(s) esperado(s)), `answer_accuracy` (fracción
de literales `must_contain` presentes en la respuesta), `faithfulness`
(los literales que aparecen en la respuesta también aparecen en el
contexto recuperado), latencia p50/p95.

Métricas del Asistente: **F1 micro-promedio** por técnica en
clasificación (precision, recall, F1), **cobertura de keywords** en
extracción y resumen (fracción de keywords esperadas que aparecen en
los items generados), latencia total y por caso. El reporte JSON
permite auditar cada caso individual.

### 3.7 Tests unitarios

Suite de 171 tests en `tests/` (runtime 7.4 s) que cubre chunking,
ingestion, faithfulness, retriever BM25, query decomposer, logger y
los 5 módulos del asistente. Detalle en
[`test_unitarios.md`](test_unitarios.md).

```powershell
# Instalar dependencias de dev (pytest) solo la primera vez
pip install -r requirements-dev.txt

# Correr la suite completa
python -m pytest
```

---

## 4. Estructura del repositorio

```
rag-riesgos/
├── docs/                     # <- coloca aqui los documentos (.pdf, .txt, .md)
├── data/
│   ├── chroma/               # vector store persistente (se genera al indexar)
│   ├── manifest.json         # hashes SHA-256 de docs/ para sync incremental
│   └── queries.db            # log SQLite de consultas (se genera con el primer ask)
├── src/
│   ├── config.py             # configuracion central desde .env
│   ├── ingestion.py          # extraccion PDF + OCR fallback + .txt (+ tablas via pdfplumber)
│   ├── chunking.py           # fragmentacion + section_hint para citas
│   ├── embeddings.py         # bge-m3 via Ollama
│   ├── vectorstore.py        # ChromaDB persistente
│   ├── retriever.py          # hibrido BM25 + vectorial con RRF
│   ├── query_decomposer.py   # descomposicion de preguntas compuestas
│   ├── reranker.py           # cross-encoder bge-reranker-v2-m3 (opcional)
│   ├── generator.py          # prompt + llama3 via Ollama (stream)
│   ├── faithfulness.py       # guardrail: cifras/articulos/fechas inventados
│   ├── dashboard.py          # observabilidad sobre queries.db (pandas + streamlit)
│   ├── corpus_sync.py        # sincronizacion incremental docs/ <-> ChromaDB
│   ├── logger_db.py          # log SQLite con retencion 30 dias
│   ├── pipeline.py           # orquestacion index_corpus() y ask()/ask_stream()
│   └── assistant/            # asistente de tareas especializadas
│       ├── prompts.py        # templates zero-shot y CoT para las 3 tareas
│       ├── llm_utils.py      # cliente Ollama + parser JSON tolerante
│       ├── classifier.py     # clasificacion multi-etiqueta con peso
│       ├── extractor.py      # extraccion 4 categorias via map-reduce
│       ├── summarizer.py     # resumen de actas via map-reduce
│       ├── corpus_utils.py   # listado de docs + filtro "acta"
│       └── ui.py             # render de la pestaña Asistente
├── scripts/
│   ├── index_documents.py    # reindexar corpus completo desde consola
│   ├── sync_docs.py          # sincronizacion incremental (one-shot)
│   ├── watch_docs.py         # watcher continuo: sync automatica al cambiar docs/
│   ├── eval.py               # harness de evaluacion (recall, accuracy, faith.)
│   ├── eval_compare.py       # comparar el mismo harness contra varios LLMs
│   ├── eval_alpha_sweep.py   # sweep de RERANKER_ALPHA para calibrar blend
│   ├── eval_assistant.py     # eval comparativa Zero-shot vs CoT del asistente
│   └── setup_ollama.ps1      # instalar Ollama + modelos (PowerShell)
├── eval/
│   ├── ground_truth.jsonl             # 12 preguntas con expected_sources y must_contain
│   ├── assistant_ground_truth.json    # casos de clasificacion, extraccion y resumen
│   ├── report.json                    # ultimo reporte de eval (pipeline RAG)
│   ├── compare_report.json            # ultima comparacion entre LLMs
│   ├── alpha_sweep.json               # ultimo sweep del reranker blended
│   └── assistant_report.json          # ultimo reporte del eval del asistente
├── tests/                    # 171 tests pytest (chunking, ingestion, retriever,
│                             # faithfulness, logger, asistente) — ver test_unitarios.md
├── app.py                    # UI Streamlit (4 pestañas: Consulta, Dashboard, Asistente, Historial)
├── requirements.txt          # dependencias Python (con prerequisitos documentados)
├── requirements-dev.txt      # dependencias de dev (pytest)
├── pytest.ini                # configuracion de pytest
├── .env.example              # plantilla de configuracion
├── .gitignore
├── DESIGN.md                 # justificacion de decisiones de diseño
├── PLAN.md                   # plan de ejecucion y roadmap
├── EVALUACION.md             # evaluacion con 12 preguntas (congelado)
├── test_unitarios.md         # inventario de la suite de tests unitarios
└── README.md                 # este archivo
```

---

## 5. Troubleshooting

| Problema | Solución |
|---|---|
| `ollama: command not found` | Abre una terminal nueva (PATH se actualiza al instalar) o usa la ruta completa: `%LOCALAPPDATA%\Programs\Ollama\ollama.exe` |
| `ConnectionError` al consultar | Verifica que Ollama esté corriendo: `ollama list`. Si no, ejecuta `ollama serve` en otra terminal. |
| OCR devuelve texto basura | Verifica que Tesseract tiene el idioma español: `tesseract --list-langs` debe incluir `spa`. |
| `chroma-hnswlib` no compila | Usa Python 3.11+ y asegúrate de tener chromadb >= 0.6.0 (las versiones recientes traen wheel precompilado). |
| `ModuleNotFoundError` | Verifica que el venv está activado: `which python` debe apuntar a `.venv/Scripts/python.exe`. |
