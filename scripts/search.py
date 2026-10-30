"""Search form questions with one embedding model, or compare all models side by side.

Run: python -m scripts.search
"""

import pandas as pd

from parakh.questions import search_questions
from scripts import ask_form_id, ask_model

TOP_K = 10

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 50)


def main() -> None:
    query = ""
    while not query:
        query = input("Query (English or Hindi): ").strip()
    form_id = ask_form_id(allow_all=False)
    models = ask_model(allow_all=True)

    results = []
    for model in models:
        rows = pd.DataFrame(search_questions(query, model, form_id=form_id, top_k=TOP_K))
        print(f"\nTop matches: {model}")
        if rows.empty:
            print("(no matches)")
            continue
        print(rows.to_string(index=False))
        results.append(rows.assign(model=model))

    if len(results) > 1:
        combined = pd.concat(results).pivot_table(
            index=["form_id", "ques_id", "question_caption"],
            columns="model",
            values="similarity",
            aggfunc="max",
        )
        print("\nBest similarity per question and model")
        print(combined.round(3).to_string())


if __name__ == "__main__":
    main()
