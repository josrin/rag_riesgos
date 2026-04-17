"""Tests para `src/config.py`.

Solo valida contratos estaticos: dataclass frozen, tipos esperados,
paths absolutos y defaults razonables. El override por env var se
verifica recargando el modulo con variables inyectadas.
"""
import dataclasses
import importlib
from pathlib import Path

import pytest


def test_settings_is_frozen_dataclass():
    from src.config import Settings, settings

    assert dataclasses.is_dataclass(Settings)
    # frozen=True -> asignacion directa debe levantar FrozenInstanceError.
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.llm_model = "otro"  # type: ignore[misc]


def test_paths_are_absolute():
    from src.config import settings

    # Los paths derivados de ROOT deben ser absolutos para que funcionen
    # independientemente del cwd desde el que se invoque la app.
    assert Path(settings.docs_dir).is_absolute()
    assert Path(settings.chroma_dir).is_absolute()
    assert Path(settings.queries_db).is_absolute()


def test_numeric_defaults_are_positive():
    from src.config import settings

    assert settings.chunk_size > 0
    assert settings.chunk_overlap >= 0
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.top_k > 0
    assert settings.reranker_pool_size >= settings.top_k
    assert 0.0 <= settings.reranker_alpha <= 1.0
    assert settings.log_retention_days > 0


def test_collection_name_stable():
    # Cambiar este nombre rompe persistencia: los datos indexados con
    # el nombre anterior no se recuperan. Lo fijamos para detectar
    # renombrados accidentales.
    from src.config import settings

    assert settings.collection_name == "riesgos_corpus"


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "mistral")
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    import src.config

    reloaded = importlib.reload(src.config)
    assert reloaded.settings.llm_model == "mistral"
    assert reloaded.settings.top_k == 9
    assert reloaded.settings.reranker_enabled is True


def test_reranker_enabled_parses_falsy_strings(monkeypatch):
    for val in ("false", "0", "no", "random"):
        monkeypatch.setenv("RERANKER_ENABLED", val)
        import src.config

        reloaded = importlib.reload(src.config)
        assert reloaded.settings.reranker_enabled is False, f"fallo con '{val}'"
