from answer_embedder import embed_answers
from form_embedder  import embed_form


def main() -> None:
    """Prompt for a form ID and embed its answers."""
    form_id = int(input("Enter form ID: ").strip())
    try:
        print(f"Embedding questions for form_id={form_id}.")
        embed_form(form_id)
        print(f"Embedding answers for form_id={form_id}.")

        embed_answers(form_id)
        print(f"Embedding completed for form_id={form_id}.")
    except Exception as error:
        print(f"Answer embedding failed: {error}")
        print("Answer embedding traceback")

if __name__ == "__main__":
    main()