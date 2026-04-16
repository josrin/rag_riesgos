"""Watcher continuo de `docs/` → sync incremental automatico.

Uso:
    python -m scripts.watch_docs

Mira los eventos del sistema de archivos en `docs/` y dispara
`corpus_sync.sync()` cuando hay cambios. Usa debouncing de 2 segundos:
si llegan varios eventos seguidos (p.ej. copiar 5 archivos a la vez),
espera a que cese la ráfaga y ejecuta una unica sync. `Ctrl+C` para
detener.
"""
from __future__ import annotations

import io
import logging
import sys
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src import corpus_sync
from src.config import settings
from src.ingestion import SUPPORTED_EXTS

DEBOUNCE_S = 2.0

logger = logging.getLogger("watch_docs")


class _SyncHandler(FileSystemEventHandler):
    """Colecciona eventos y programa una unica sync tras la ráfaga."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._pending_reason = ""

    def _relevant(self, path: str) -> bool:
        p = Path(path)
        return p.suffix.lower() in SUPPORTED_EXTS

    def _schedule(self, reason: str) -> None:
        with self._lock:
            self._pending_reason = reason
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_S, self._run_sync)
            self._timer.daemon = True
            self._timer.start()

    def _run_sync(self) -> None:
        with self._lock:
            reason = self._pending_reason
            self._pending_reason = ""
            self._timer = None
        try:
            result = corpus_sync.sync()
            if any([result["added"], result["modified"], result["deleted"]]):
                logger.info(
                    "sync tras '%s': +%s ~%s -%s (%s chunks, %ss)",
                    reason,
                    len(result["added"]),
                    len(result["modified"]),
                    len(result["deleted"]),
                    result["indexed_chunks"],
                    result["elapsed_s"],
                )
            else:
                logger.debug("sync tras '%s': sin cambios netos", reason)
        except Exception as e:  # pragma: no cover — diagnostico
            logger.error("Fallo en sync: %s", e, exc_info=True)

    def _on_event(self, event, kind: str) -> None:
        if event.is_directory:
            return
        if not self._relevant(event.src_path):
            return
        self._schedule(f"{kind}:{Path(event.src_path).name}")

    def on_created(self, event) -> None:
        self._on_event(event, "create")

    def on_modified(self, event) -> None:
        self._on_event(event, "modify")

    def on_deleted(self, event) -> None:
        self._on_event(event, "delete")

    def on_moved(self, event) -> None:
        self._on_event(event, "move")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    docs_dir = settings.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    corpus_sync.bootstrap_if_needed()
    initial = corpus_sync.sync()
    if any([initial["added"], initial["modified"], initial["deleted"]]):
        logger.info(
            "sync inicial: +%s ~%s -%s (%s chunks)",
            len(initial["added"]),
            len(initial["modified"]),
            len(initial["deleted"]),
            initial["indexed_chunks"],
        )
    else:
        logger.info("corpus ya sincronizado al arrancar el watcher")

    observer = Observer()
    handler = _SyncHandler()
    observer.schedule(handler, str(docs_dir), recursive=False)
    observer.start()
    logger.info("Watcher activo sobre %s. Ctrl+C para detener.", docs_dir)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Deteniendo watcher...")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
