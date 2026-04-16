"""Dashboard de observabilidad sobre `data/queries.db`.

Renderiza KPIs, series temporales y distribuciones sobre el historico
de consultas para detectar patrones operativos:
- Tasa de respuestas "no encuentro informacion" (gap del corpus).
- Documentos mas citados (cobertura real vs. cobertura esperada).
- Latencia p50/p95 por dia y por modelo.
- Volumen de consultas por dia.
- Warnings de faithfulness acumulados.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import embeddings, logger_db
from .config import settings

_NO_MATCH_MARKER = "no encuentro esta informacion"


def _normalize(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()


def _load_dataframe() -> pd.DataFrame:
    rows = logger_db.all_queries()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = df["ts"].dt.date
    df["no_match"] = df["answer"].apply(lambda a: _NO_MATCH_MARKER in _normalize(a or ""))
    df["warnings_count"] = df["warnings"].apply(len)
    df["sources_count"] = df["sources"].apply(len)
    return df


def _source_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["source", "citas"])
    counts: dict[str, int] = {}
    for sources in df["sources"]:
        for s in sources:
            counts[s["source"]] = counts.get(s["source"], 0) + 1
    out = pd.DataFrame(
        [{"source": k, "citas": v} for k, v in counts.items()]
    ).sort_values("citas", ascending=False)
    return out


def _top_questions(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["question", "veces", "no_match"])
    agg = df.groupby("question").agg(
        veces=("id", "count"),
        no_match=("no_match", "sum"),
        p50=("latency_ms", "median"),
    )
    return agg.sort_values("veces", ascending=False).head(n).reset_index()


def render() -> None:
    st.header("Dashboard de consultas")
    df = _load_dataframe()

    if df.empty:
        st.info("Todavía no hay consultas registradas. Usa la pestaña *Consulta* y vuelve aquí.")
        return

    st.caption(
        f"{len(df)} consultas registradas en ventana de "
        f"{settings.log_retention_days} días."
    )

    total = len(df)
    last_7 = (df["ts"] >= (pd.Timestamp.utcnow() - pd.Timedelta(days=7))).sum()
    p50 = int(df["latency_ms"].median())
    p95 = int(df["latency_ms"].quantile(0.95))
    no_match_rate = df["no_match"].mean()
    warn_rate = (df["warnings_count"] > 0).mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total", total)
    c2.metric("Últimos 7 días", int(last_7))
    c3.metric("Latencia p50", f"{p50} ms")
    c4.metric("Latencia p95", f"{p95} ms")
    c5.metric("No encuentro", f"{no_match_rate:.0%}")
    c6.metric("Con warnings", f"{warn_rate:.0%}")

    st.divider()

    st.subheader("Consultas por día")
    daily = df.groupby("date").size().rename("consultas").to_frame()
    st.bar_chart(daily)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Documentos más citados")
        sources = _source_frequencies(df)
        if not sources.empty:
            st.bar_chart(sources.set_index("source")["citas"])
        else:
            st.caption("Sin citas registradas.")

    with col_right:
        st.subheader("Latencia por modelo LLM")
        if df["llm_model"].notna().any():
            per_model = df.groupby("llm_model")["latency_ms"].agg(
                ["count", "median", lambda s: s.quantile(0.95)]
            )
            per_model.columns = ["n", "p50_ms", "p95_ms"]
            st.dataframe(per_model, use_container_width=True)
        else:
            st.caption("Modelo no registrado.")

    st.subheader("Preguntas más frecuentes")
    top = _top_questions(df)
    if not top.empty:
        top["no_match_rate"] = (top["no_match"] / top["veces"]).map(lambda x: f"{x:.0%}")
        top["p50_ms"] = top["p50"].astype(int)
        st.dataframe(
            top[["question", "veces", "p50_ms", "no_match_rate"]],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Cache de embeddings"):
        cs = embeddings.cache_stats()
        total = cs["hits"] + cs["misses"]
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Hits", cs["hits"])
        cB.metric("Misses", cs["misses"])
        cC.metric("Hit rate", f"{cs['hit_rate']:.0%}" if total else "—")
        cD.metric("Tamaño", f"{cs['size']}/{cs['maxsize']}")
        st.caption(
            "Embeddings recientes se cachean en memoria. Al cambiar "
            "`EMBEDDING_MODEL` el cache se invalida automaticamente."
        )

    with st.expander("Faithfulness warnings detectados"):
        rows = []
        for _, row in df[df["warnings_count"] > 0].iterrows():
            for w in row["warnings"]:
                rows.append(
                    {
                        "ts": row["ts"],
                        "question": row["question"][:80],
                        "kind": w.get("kind", ""),
                        "claim": w.get("claim", ""),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Sin warnings registrados en el período.")
