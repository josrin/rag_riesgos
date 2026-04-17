"""Sincronizacion incremental desde linea de comandos.

    python -m scripts.sync_docs [--dry-run]

- Sin flags: detecta cambios en `docs/` y aplica la delta sobre
  ChromaDB (reindexa solo nuevos/modificados, borra eliminados).
- Con `--dry-run`: solo imprime que se haria.

Idempotente: una segunda llamada consecutiva sin cambios no hace nada.
"""
from __future__ import annotations

import io
import sys

from src import corpus_sync


def _fmt_list(lst: list[str]) -> str:
    """Formatea una lista de nombres para impresion multilinea o '(ninguno)' si vacia."""
    return "\n  ".join(lst) if lst else "(ninguno)"


def main() -> None:
    """Entrypoint CLI: muestra delta pendiente y sincroniza si no es `--dry-run`."""
    dry = "--dry-run" in sys.argv[1:]
    corpus_sync.bootstrap_if_needed()
    state = corpus_sync.scan_state()

    print(f"Unchanged : {len(state['unchanged'])}")
    print(f"Nuevos    ({len(state['new'])}):\n  {_fmt_list(state['new'])}")
    print(f"Modificados ({len(state['modified'])}):\n  {_fmt_list(state['modified'])}")
    print(f"Eliminados ({len(state['deleted'])}):\n  {_fmt_list(state['deleted'])}")

    if not corpus_sync.has_changes(state):
        print("\nSin cambios pendientes.")
        return

    if dry:
        print("\n[dry-run] No se aplican cambios.")
        return

    result = corpus_sync.sync()
    print()
    print(
        f"Sync completo en {result['elapsed_s']}s: "
        f"+{len(result['added'])} nuevos, "
        f"~{len(result['modified'])} modificados, "
        f"-{len(result['deleted'])} eliminados "
        f"({result['indexed_chunks']} chunks indexados)"
    )


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
