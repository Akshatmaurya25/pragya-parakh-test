"""Embed a form's questions and its submitted answers in one go.

Run: python -m scripts.embed
"""

import logging

from parakh.answers import answered_form_ids, embed_answers
from parakh.questions import embed_questions
from scripts import ask_form_id, ask_model, setup_logging

LOGGER = logging.getLogger("scripts.embed")
WHAT = {"1": "questions + answers", "2": "questions only", "3": "answers only"}


def main() -> None:
    setup_logging()
    form_id = ask_form_id(allow_all=True)
    [model] = ask_model()
    print("What to embed:", ", ".join(f"{k}. {v}" for k, v in WHAT.items()))
    choice = input("Choose [Enter = 1]: ").strip() or "1"
    if choice not in WHAT:
        raise SystemExit("Invalid choice.")

    try:
        if choice in {"1", "2"}:
            embed_questions(form_id, model)
        if choice in {"1", "3"}:
            # One form at a time keeps memory bounded when embedding everything.
            for fid in [form_id] if form_id else answered_form_ids():
                embed_answers(fid, model)
    except Exception:
        LOGGER.exception("Embedding failed")
        raise
    LOGGER.info("Done: %s for form %s with %s.", WHAT[choice], form_id or "*", model)


if __name__ == "__main__":
    main()
