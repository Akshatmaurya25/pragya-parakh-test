"""Interactive command-line scripts. Run from the project root: python -m scripts.<name>

Shared input helpers live here so every script asks the same way.
"""

import logging
from pathlib import Path

from parakh.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS

LOG_FILE = Path(__file__).resolve().parent.parent / "parakh.log"
MODELS = list(EMBEDDING_MODELS)


def setup_logging() -> None:
    """Log to the console and to parakh.log."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")],
    )


def ask_form_id(allow_all: bool) -> int | None:
    """Ask for a positive form id. Blank (or * when allow_all) means every form."""
    hint = "* for all forms" if allow_all else "blank for all forms"
    while True:
        text = input(f"Form ID ({hint}): ").strip()
        if (allow_all and text == "*") or (not allow_all and not text):
            return None
        if text.isdigit() and int(text) > 0:
            return int(text)
        print("Please enter a positive number.")


def ask_model(allow_all: bool = False) -> list[str]:
    """Ask for an embedding model (Enter = default). Returns the chosen model(s)."""
    print("Embedding model:")
    for number, model in enumerate(MODELS, 1):
        default = " (default)" if model == DEFAULT_EMBEDDING_MODEL else ""
        print(f"  {number}. {model}{default}")
    if allow_all:
        print("  a. all models (compare)")
    while True:
        text = input("Choose [Enter = default]: ").strip().lower()
        if not text:
            return [DEFAULT_EMBEDDING_MODEL]
        if allow_all and text == "a":
            return MODELS
        if text.isdigit() and 1 <= int(text) <= len(MODELS):
            return [MODELS[int(text) - 1]]
        print("Invalid choice.")
