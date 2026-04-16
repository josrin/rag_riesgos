# Evaluación del sistema RAG — Iteración #1.5

> **Archivo congelado.** Este documento captura el estado del sistema
> tras la implementación del retriever híbrido y el cambio a embeddings
> `bge-m3`. Registra las pruebas realizadas, los fallos observados y el
> plan de mejoras para llevar el sistema a producción. No debe
> modificarse posteriormente: las acciones que se deriven de él se
> ejecutan en el código y se reflejan en `PLAN.md`.

Fecha de evaluación: 2026-04-16
Corpus: 5 documentos `.txt` → 36 chunks indexados.
Stack: `llama3` · `bge-m3` · ChromaDB · retriever híbrido BM25+vector con RRF.

---

## 1. Metodología

Se diseñaron **12 preguntas** cubriendo los 5 documentos con distintos
niveles de complejidad:

- **L1 — Lookup simple**: un dato puntual en un solo documento.
- **L2 — Lookup con contexto**: requiere entender el párrafo.
- **L3 — Síntesis intra-documento**: combinar varias partes del mismo doc.
- **L4 — Síntesis cross-document**: combinar información de dos o más docs.
- **L5 — Detalle fino / inferencia**: datos específicos que requieren
  lectura atenta.

Cada pregunta se ejecutó contra `pipeline.ask()` (retriever híbrido +
`llama3` con `temperature=0.1`). El resultado se evaluó manualmente
contra el texto literal de los documentos fuente.

---

## 2. Preguntas ejecutadas

Las preguntas se lanzaron literalmente como se listan a continuación:

| # | Nivel | Pregunta |
|---|-------|----------|
| 01 | L1 | ¿Cuál es el límite de exposición por emisor individual en la política de crédito? |
| 02 | L1 | ¿Cuándo entra en vigencia la Circular 034 de 2025 de la SFC? |
| 03 | L1 | ¿Cuánto fueron las pérdidas operacionales totales del Q4 2025? |
| 04 | L2 | ¿Qué metodología se usa para calcular el VaR? |
| 05 | L2 | ¿Qué medidas se aprobaron tras el incidente de phishing del 5 de diciembre? |
| 06 | L2 | ¿Qué tests estadísticos se usan en el backtesting del modelo VaR? |
| 07 | L3 | ¿Qué haircuts se aplican a las distintas garantías? |
| 08 | L3 | ¿Qué KRIs estuvieron en estado rojo en el Q4 2025? |
| 09 | L3 | ¿Cuáles son todos los límites de inversión establecidos por la SFC en la Circular 034? |
| 10 | L4 | ¿El VaR actual del portafolio cumple con los requisitos de la SFC? |
| 11 | L4 | ¿Cuál es el límite de renta variable internacional y qué decidió el comité al respecto? |
| 12 | L5 | ¿Cuánto tiempo pasó entre el compromiso de credenciales y la detección del ataque de phishing? |

---

## 3. Resultados

| # | Nivel | Resultado | Latencia | Observación |
|---|-------|-----------|----------|-------------|
| 01 | L1 | ✅ Correcto | 11.4 s* | Cold start de Ollama |
| 02 | L1 | ✅ Correcto | 0.9 s | |
| 03 | L1 | ✅ Correcto | 0.8 s | |
| 04 | L2 | ✅ Correcto | 2.1 s | Nombra FHS + GARCH + fórmula √10 |
| 05 | L2 | ✅ Correcto | 1.3 s | Lista las 4 medidas (MFA, simulaciones, dual-control, lista blanca) |
| 06 | L2 | ✅ Correcto | 1.2 s | Preámbulo ruidoso: "La pregunta es:..." |
| 07 | L3 | ✅ Correcto | 1.1 s | |
| 08 | L3 | ✅ Correcto | 1.2 s | Preámbulo ruidoso: "La respuesta es:..." |
| 09 | L3 | ✅ Correcto | 1.7 s | Cita Artículo 4 cuando el contenido es del Artículo 5 |
| 10 | L4 | ⚠️ Parcial | 2.4 s | Contenido correcto, pero artifact textual |
| 11 | L4 | ❌ Fallo | 0.7 s | Responde solo la primera parte y dice "no encuentro" la segunda |
| 12 | L5 | ✅ Correcto | 0.9 s | Formato de citas distinto al esperado |

*Cold start. Latencia media tras el warmup: **1.3 s**.

**Precisión global:** 10/12 = **83 %**.

---

## 4. Análisis de fallos

### 4.1 Fallo grave — Pregunta 11 (pregunta compuesta)

> *¿Cuál es el límite de renta variable internacional y qué decidió el
> comité al respecto?*

**Respuesta del sistema:**
> "El límite de renta variable internacional aprobado es del 30% del
> valor del fondo, según lo establecido en la Circular 034 de 2025.
> No encuentro esta información en los documentos indexados."

**Respuesta esperada:**
> La Circular 034 fija el tope en **30 %**, pero el Comité de Riesgos
> aprobó un incremento del 25 % al **28 %** (rechazando el 30 %
> solicitado), con la condición de mantener cobertura cambiaria mínima
> del 60 % y un período de transición de 3 meses.

**Por qué falla:**
- El retriever híbrido sí recuperó 4 chunks del acta del comité con la
  información correcta.
- El LLM respondió únicamente la primera subpregunta y, al no encontrar
  literalmente "30%" en los chunks de la decisión del comité, aplicó la
  instrucción del prompt de decir *"no encuentro esta información"*.
- El prompt actual no distingue entre "falta toda la información" y
  "falta parte de la información", generando un falso negativo cuando
  la pregunta es compuesta.

### 4.2 Fallo menor — Pregunta 10 (artifact textual)

**Respuesta del sistema:**
> "La Circular E DOCUMENTO METODOLÓGICO: CÁLCULO DEL VALUE AT RISK
> (VaR) indica que el nivel de confianza..."

El contenido conceptual es correcto, pero el LLM concatenó el final del
encabezado `[Fragmento i]` con el inicio del siguiente chunk. Es un
problema de formato del bloque de contexto entregado al LLM.

### 4.3 Patrones transversales menores

- **Preámbulos redundantes**: "La pregunta es:...", "La respuesta es:..."
  aparecen en al menos 2 respuestas. El prompt no los prohíbe.
- **Formato de citas inconsistente**: algunas respuestas usan
  `[archivo.pdf, página X, sección Y]` como pide el system prompt, otras
  usan `[Fragmento N] Fuente: ...` (formato del bloque de contexto).
- **Citas imprecisas en número de artículo**: en la pregunta 9 cita
  "Artículo 4" cuando el contenido corresponde al "Artículo 5".

### 4.4 Evaluación de retrieval (híbrido BM25 + vectorial)

El retriever funciona correctamente: en todas las preguntas el top-5
contuvo al menos un chunk del documento fuente correcto. El fallo de
la pregunta 11 es de **generación**, no de recuperación.

---

## 5. Plan de trabajo (producción)

Ordenado por relación impacto/costo.

### P0 — Habilitadores y requisitos no negociables

**#1 — Evaluación automatizada** (*bloqueante para iterar sin regresiones*)
- `eval/ground_truth.jsonl` con `question`, `expected_answer`,
  `expected_chunk_ids`, `must_contain` (literales).
- Runner `scripts/eval.py` con métricas: `recall@5`, `precision@5`,
  `faithfulness`, latencia p50/p95.
- Ejecutar al tocar prompt, chunker, retriever o embeddings.

**#2 — Hardening del prompt del sistema**
- Prohibir preámbulos ("no repitas la pregunta, no digas 'La respuesta
  es:'").
- Formato de citas estricto con ejemplo literal.
- Instrucción específica para preguntas compuestas: *"si la pregunta
  tiene varias partes, respóndelas todas en orden. Solo declara 'no
  encuentro información' si NINGUNA parte aparece en el contexto."*
- Instrucción anti-concatenación de encabezados de fragmentos.

**#5 — Control de acceso y auditoría** (*requisito regulatorio*)
- SSO corporativo (OIDC) en lugar de Streamlit abierto.
- Log enriquecido: usuario, versión del prompt, `TOP_K`, scores RRF,
  modelo LLM, modelo embedding, timestamp UTC.
- Control por documento si hay confidencialidad por área.

### P1 — Calidad de respuesta

**#3 — Query decomposition** (*arregla la pregunta 11 directamente*)
- Detectar preguntas compuestas (conjunciones *"y qué..."*,
  *"además..."*, dos signos `?`) y generar subqueries.
- Unir top-k de cada subquery para el LLM.

**#4 — Cross-encoder reranker**
- `bge-reranker-v2-m3` sobre los top-20 del híbrido.
- Sube `precision@5` entre 5–15 pp en español.

**#7 — Faithfulness check post-generación**
- Validar que cifras, fechas y números de artículo citados aparezcan
  literalmente en los chunks recuperados.
- Flag si no. Crítico en dominio regulatorio.

### P2 — Operación sostenida

**#6 — Gestión del corpus**
- Watcher sobre `docs/` + reindex incremental automático.
- Versionado: marcar `version=N` al cambiar un documento.
- Dashboard de salud del corpus.

**#10 — Chunking robusto**
- Detección de tablas (`is_table=true` en metadata).
- Reforzar extracción del `section_hint` (hoy vacío en varios chunks).

**#9 — Warmup y optimización de latencia**
- Warmup de `llama3` y `bge-m3` al arrancar la app (evita cold start de
  11 s).
- Streaming de tokens en Streamlit.
- Cache LRU de embeddings de queries repetidas.

**#12 — Observabilidad**
- Dashboard (pandas/Grafana) con `recall@5` diario, latencia p95, tasa
  de *"no encuentro"*, preguntas más frecuentes.

### P3 — Refinamientos

**#8 — Guardrails regulatorios**
- Redacción de PII.
- Política de retención alineada con exigencias del regulador (posible
  aumento más allá de 30 días).
- Firma/hash del corpus al momento de consulta → respuesta reproducible.

**#11 — HyDE / query expansion**
- Para preguntas cortas o ambiguas, generar respuesta hipotética y
  embeberla en vez de la pregunta literal.

---

## 6. Matriz de priorización

| Prioridad | Iniciativa | Justificación |
|-----------|-----------|---------------|
| P0 | #1 Eval harness | Habilita iterar sin regresiones. |
| P0 | #2 Prompt hardening | Coste bajo, cierra 3 de los problemas observados. |
| P0 | #5 Control de acceso + audit log | Requisito regulatorio no negociable. |
| P1 | #3 Query decomposition | Arregla el fallo L4 (pregunta 11). |
| P1 | #7 Faithfulness check | Mitiga alucinaciones en dominio sensible. |
| P1 | #4 Reranker | Mejora retrieval con bajo esfuerzo. |
| P2 | #6 Gestión corpus + #10 Chunking | Necesario cuando el corpus crezca. |
| P2 | #9 Latencia + #12 Observabilidad | UX y operación. |
| P3 | #8 Guardrails + #11 HyDE | Refinamientos. |

---

## 7. Próximo paso inmediato

Trabajar en los items **#2 (prompt hardening)** y **#3 (query
decomposition)** en esta misma iteración, ya que cierran los fallos
observados (pregunta 10 parcial y pregunta 11 grave) con bajo costo. Al
terminar se volverán a ejecutar únicamente las preguntas 10 y 11 para
verificar la corrección.
