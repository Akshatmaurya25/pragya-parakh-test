# Flows

Who calls what, step by step. For the component overview see
[architecture.md](./architecture.md).

## 1. Ingestion: question embeddings

`python form_embedder.py` → enter a form id (or `*` for all forms)

```mermaid
sequenceDiagram
    participant U as User (CLI)
    participant FE as form_embedder.py
    participant EMB as embedder.py
    participant DB as db_conn.py
    participant PG as PostgreSQL

    U->>FE: main() → _parse_form_id()
    FE->>FE: embed_form(form_id)
    FE->>DB: fetch_form_questions() → execute(QUESTION_QUERY)
    DB->>PG: SELECT ... FROM questionl_details
    FE->>EMB: prepare_embedding_rows() → _embed_caption_column() ×2 (EN, HI) → embedder()
    FE->>DB: insert_form_embeddings() → execute_many(INSERT_QUERY)
    DB->>PG: INSERT ... ON CONFLICT (model, ques_id) DO UPDATE → ques_embed
```

## 2. Ingestion: answer embeddings

`python answer_embedder.py` → enter a form id

```mermaid
sequenceDiagram
    participant U as User (CLI)
    participant AE as answer_embedder.py
    participant EMB as embedder.py
    participant DB as db_conn.py
    participant PG as PostgreSQL

    U->>AE: main() → embed_answers(form_id)
    AE->>DB: execute(ANSWER_QUERY)
    DB->>PG: SELECT ... FROM form_respond_details WHERE form_id = ?
    AE->>AE: answer_text() per row (JSON options → "Rural"), drop blanks
    AE->>EMB: embed_distinct() → embedder() once per distinct answer text
    AE->>DB: execute_many(INSERT_QUERY)
    DB->>PG: INSERT ... ON CONFLICT (model, response_id) DO UPDATE → ans_embed
```

Logging for both scripts goes to `form_embedder.log` (`answer_embedder` reuses
`form_embedder.configure_logging/log_message`).

## 3. Chat request (main flow)

`streamlit run app.py` → Chat tab

```mermaid
sequenceDiagram
    participant O as Officer
    participant APP as app.py
    participant MA as master_agent.MasterAgent
    participant LLM as LLM API
    participant T as tools.py
    participant X as query_resolver / answer_embedder / db_conn

    O->>APP: chat_input(prompt)
    APP->>MA: agent.llm, agent.embedding_model = selections
    APP->>MA: handle(prompt)
    MA->>MA: _run(): tools.EMBEDDING_MODEL = embedding_model
    loop up to MAX_STEPS (6)
        MA->>LLM: _decide(): PLANNER_SYSTEM + describe_tools() + history + steps so far
        LLM-->>MA: JSON {action: call_tool | final | clarify}
        alt call_tool
            MA->>T: call_tool(name, args) (validate + coerce)
            T->>X: tool function
            X-->>T: rows
            T-->>MA: {ok, result | error} → appended to steps
        else final / clarify
            MA-->>APP: AgentResponse(action, message, matches, trace, answers)
        end
    end
    Note over MA: LLM error / bad JSON / repeated call / steps exhausted → _fallback(): plain search_questions + similarity threshold
    APP->>O: message + "Matches & tool calls" expander (answers, matches, trace)
```

`MasterAgent.handle` also appends the user message and reply to
`self.history`; the last 6 turns go into every `_decide` prompt, so follow-ups
like "show the full submission" resolve against earlier turns.

### Typical tool chains the planner uses

| User asks | Tool chain |
|---|---|
| "Which questions are about toilets?" | `search_questions` → final |
| "What does form 459 contain?" | `get_form_questions` → final |
| "How was the counsellor question answered?" | `search_questions` (find form_id, ques_id) → `answer_summary` → final |
| "Find hostels reporting broken toilets in form 459" | `search_answers` (optionally after `search_questions` for ques_id) → final |
| "Show that full inspection" | `search_answers` → `get_submission(ans_submission_id)` → final |
| "Hi" | final (no tool) |
| "status" (too vague) | clarify |

### What each tool calls

```mermaid
flowchart LR
    SQ[search_questions] -->|per phrasing| RET[_default_retriever] --> RTK[query_resolver.retrieve_top_k] --> E1[embedder] & Q1[(ques_embed)]
    SQ -->|RRF fuse + dedupe| OUT1[ranked questions]
    GFQ[get_form_questions] --> R1[_db_rows] --> Q2[(questionl_details)]
    LF[list_forms] --> R2[_db_rows] --> Q3[(ques_embed)]
    SA[search_answers] --> RA[answer_embedder.retrieve_answers] --> E2[embedder] & Q4[(ans_embed ⟕ ques_embed)]
    AS[answer_summary] --> R3[_db_rows] --> Q5[(form_respond_details)]
    AS --> AT1[answer_text → merge counts]
    GS[get_submission] --> R4[_db_rows] --> Q6[(form_respond_details ⟕ questionl_details)]
    GS --> AT2[answer_text per row]
```

All `_db_rows` / `retrieve_*` calls go through `db_conn.execute`, which opens a
connection, runs the query and returns rows as a JSON string.

## 4. Forms tab

```mermaid
sequenceDiagram
    participant O as Officer
    participant APP as app.py
    participant DB as db_conn.py
    participant T as tools.get_form_questions

    O->>APP: open Forms tab
    APP->>DB: embedded_forms() (cached 5 min)
    Note over APP,DB: ques_embed LEFT JOIN depertment_form_master → form_id, form_name<br/>(no form master table → ids only)
    APP-->>O: rows: S.No | Form ID | Form name | View
    O->>APP: click View
    APP->>T: show_form() dialog → get_form_questions(form_id)
    APP-->>O: dialog table: sno | ques_id | question | hindi
```

## 5. Standalone CLIs (not used by the agent)

- `python query_resolver.py` → `main()` → `resolve_query()` → `retrieve_top_k()` → prints matching questions as JSON.
- `python combined_query_resolver.py` → `main()` → `resolve_all_models()` → `retrieve_model_matches()` for each model → `print_table()` per model + combined comparison.
- `python master_agent.py` → `main()` → `MasterAgent.handle()` loop in the terminal, printing the trace.

## Setup order on a fresh database

1. Fill `.env` (DB + LLM keys).
2. `psql -f sql/ques_embed.sql` and `psql -f sql/ans_embed.sql`.
3. `python form_embedder.py` (per form or `*`).
4. `python answer_embedder.py` (per form).
5. `streamlit run app.py`.
