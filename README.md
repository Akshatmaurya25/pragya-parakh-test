# Parakh Assistant

Ask questions about Madhya Pradesh inspection forms (schools, anganwadis,
hospitals, hostels…) in English, Hindi or Hinglish. The app finds the matching
form questions and submitted answers in PostgreSQL and an LLM explains them in
plain language.

## How it works, in plain words

1. **Embedding** turns each question caption (English and Hindi) and each
   submitted answer into a list of numbers that captures its meaning, and
   stores it in `ques_embed` / `ans_embed`.
2. When you ask something, your text is turned into numbers the same way and
   the closest stored questions/answers are found (pgvector similarity).
3. The **agent** (an LLM, NVIDIA by default) decides which search or summary
   to run, reads the results and writes the reply.

Details: [docs/architecture.md](docs/architecture.md), [docs/flow.md](docs/flow.md).

## Project layout

```
app.py              Streamlit UI
parakh/             the code: config, db, embeddings, questions, answers
parakh/agent/       LLM agent: llm clients, prompt, tool registry, tools, loop
scripts/            interactive scripts: embed, search, chat
tests/              checks that run without DB / models / LLM
sql/                table definitions for ques_embed and ans_embed
data/               reference Excel files
docs/               architecture and flow diagrams
```

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then fill it in
psql -f sql/ques_embed.sql
psql -f sql/ans_embed.sql
```

## Run (from the project root)

| What | Command |
|---|---|
| Embed a form's questions **and** answers | `python -m scripts.embed` |
| Search questions / compare embedding models | `python -m scripts.search` |
| Chat in the terminal | `python -m scripts.chat` |
| Web UI | `streamlit run app.py` |
| Tests | `python -m tests.test_core` |

Defaults: embedding model **krutrim-ai-labs/vyakyarth**, LLM **NVIDIA**
(change with `LLM_PROVIDER` in `.env` or the UI sidebar).
