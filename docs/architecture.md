# Architecture

Parakh assistant: a RAG + tool-calling agent over Madhya Pradesh inspection
forms. Officers ask in English, Hindi or Hinglish; an LLM planner calls tools
that search embedded **questions** and **answers** in PostgreSQL (pgvector)
and replies in plain language.

See [flow.md](./flow.md) for step-by-step flows.

## Layers

Each layer only imports from the layers below it.

```mermaid
flowchart TD
    subgraph ENTRY["Entry points"]
        APP["app.py<br/>Streamlit: Chat + Forms tabs"]
        SE["scripts/embed.py<br/>questions + answers → DB"]
        SS["scripts/search.py<br/>search / compare models"]
        SC["scripts/chat.py<br/>terminal chat"]
    end

    subgraph AGENT["parakh/agent"]
        MA["master_agent.py<br/>planner loop + fallback"]
        LLMM["llm.py<br/>NVIDIA / OpenRouter / Ollama"]
        PR["prompts.py"]
        TL["tools.py<br/>@tool wrappers + RRF"]
        RG["registry.py<br/>@tool, call_tool"]
    end

    subgraph DOMAIN["parakh (domain)"]
        Q["questions.py<br/>embed / search / list"]
        A["answers.py<br/>embed / search / summary"]
    end

    subgraph CORE["parakh (core)"]
        EMB["embeddings.py"]
        DB["db.py"]
        CFG["config.py (.env)"]
    end

    APP --> MA
    APP --> Q
    SC --> MA
    SE --> Q & A
    SS --> Q
    MA --> LLMM & PR & TL & RG
    TL --> RG
    TL --> Q & A
    Q & A --> EMB & DB
    EMB & DB & LLMM --> CFG
```

## Files

| File | Role |
|---|---|
| `app.py` | Streamlit UI. Sidebar picks the LLM and embedding model. **Chat** tab talks to the agent; **Forms** tab lists embedded forms and their questions. |
| `parakh/config.py` | Every setting: DB, LLM keys/models, `EMBEDDING_MODELS`, defaults, `IMAGE_BASE_URL`. Loads `.env` once. |
| `parakh/db.py` | `fetch(query, params) -> list[dict]`, `execute_many(query, rows)`. |
| `parakh/embeddings.py` | `embed()` (cached models), `embed_unique()` (each distinct text once → pgvector text), `to_pgvector()`. |
| `parakh/questions.py` | `embed_questions`, `search_questions`, `get_form_questions`, `list_forms`. |
| `parakh/answers.py` | `answer_text`, `image_urls`, `embed_answers`, `answered_form_ids`, `search_answers`, `get_submission`, `answer_summary`. |
| `parakh/agent/llm.py` | LLM clients, `LLMS` (UI list, NVIDIA first), `default_llm`, `extract_json`. |
| `parakh/agent/prompts.py` | `PLANNER_SYSTEM`. |
| `parakh/agent/registry.py` | `@tool`, `describe_tools`, `call_tool` (validates + coerces LLM arguments). |
| `parakh/agent/tools.py` | The tools below. Thin wrappers over `questions` / `answers`. |
| `parakh/agent/master_agent.py` | `MasterAgent`: each step the LLM returns `call_tool`, `final` or `clarify`; falls back to plain search if the LLM fails. |
| `scripts/` | Interactive scripts (`embed`, `search`, `chat`) + shared input helpers in `__init__.py`. |
| `tests/test_core.py` | Checks with fakes; no DB, model or LLM needed. |
| `sql/*.sql` | DDL for `ques_embed` and `ans_embed`. |

## Agent tools (`parakh/agent/tools.py`)

| Tool | Reads | Calls | Used for |
|---|---|---|---|
| `search_questions` | `ques_embed` | `questions.search_questions` per phrasing, fused by RRF | Find form fields/questions on a topic |
| `get_form_questions` | `questionl_details` | `questions.get_form_questions` | List all questions of one form |
| `list_forms` | `ques_embed` ⟕ `depertment_form_master` | `questions.list_forms` | Which forms are embedded |
| `search_answers` | `ans_embed` ⟕ `ques_embed` | `answers.search_answers` | Semantic search over submitted answers of a form |
| `answer_summary` | `form_respond_details` | `answers.answer_summary` | Count of each distinct answer for one question |
| `get_submission` | `form_respond_details` ⟕ `questionl_details` | `answers.get_submission` | Every answer of one submission |
| `query_submission_data` | — | raises `NotImplementedError` | Placeholder for future NL→SQL |

## Database

```mermaid
erDiagram
    depertment_form_master ||--o{ questionl_details : "form_id"
    questionl_details ||--o{ form_respond_details : "form_id, ques_id"
    questionl_details ||--o{ ques_embed : "embed_questions"
    form_respond_details ||--o| ans_embed : "id = response_id"

    depertment_form_master {
        int form_id
        text form_name
        text form_name_h
    }
    questionl_details {
        int form_id
        bigint ques_id
        text question_captions
        text question_caption_hindi
        int qus_ordering
    }
    form_respond_details {
        bigint id
        varchar ans_submissiom_id
        int form_id
        bigint ques_id
        bigint user_id
        text ans_value
        timestamp insert_dt
    }
    ques_embed {
        bigint id
        int form_id
        bigint ques_id
        text question_caption
        text question_caption_hindi
        vector embedding_en
        vector embedding_hi
        text model
    }
    ans_embed {
        bigint id
        bigint response_id
        varchar ans_submission_id
        int form_id
        bigint ques_id
        text ans_value
        vector embedding
        text model
    }
```

- **Source tables** (from the Parakh app): `depertment_form_master`, `questionl_details`, `form_respond_details`.
- **Derived tables** (ours): `ques_embed` (unique `model, ques_id`), `ans_embed` (unique `model, response_id`).
- Vectors are untyped `vector` because the three models have different dimensions; every search filters by `model`.
- Option answers are stored as JSON (`[{"OptionID":"1","OptionText":"Rural"}]`); `answer_text()` turns them into readable text before embedding or showing them to the LLM.

## Configuration (`.env`, see `.env.example`)

| Variable | Used by |
|---|---|
| `PGHOST`, `PGPORT`, `PGDATABASE`, `DB_USER`, `DB_PASS` | `db.py` |
| `LLM_PROVIDER` (`nvidia` default / `openrouter` / `ollama`) | `llm.default_llm` |
| `NVIDIA_API_KEY`, `NVIDIA_MODEL`, `NVIDIA_URL` | `nvidia_chat` |
| `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_URL` | `openrouter_chat` |
| `OLLAMA_HOST`, `OLLAMA_MODEL` | `ollama_chat` |
| `IMAGE_BASE_URL` | `answers.image_urls` |

## Embedding models (`config.EMBEDDING_MODELS`)

| Model id | Stored as `model` |
|---|---|
| `krutrim-ai-labs/vyakyarth` (default) | `krutrim-ai-labs/vyakyarth` |
| `BAAI/bge-m3` | `bge-m3` |
| `google/embeddinggemma-300m` | `google/embeddinggemma-300m` |

The query must be embedded with the **same model** that embedded the stored
rows. `scripts/embed.py` asks for the model (Enter = Vyakyarth); the Streamlit
sidebar sets `MasterAgent.embedding_model`, which `_run` copies into
`tools.EMBEDDING_MODEL` for the search tools.
