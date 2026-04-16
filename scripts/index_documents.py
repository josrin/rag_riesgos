"""Re-indexa todos los PDFs de docs/ en ChromaDB.

Uso:
    python -m scripts.index_documents
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import index_corpus  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    stats = index_corpus(reset=True)
    print("\nIndexacion completada:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
