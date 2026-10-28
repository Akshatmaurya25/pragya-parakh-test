-- Answer embeddings for form_respond_details rows. Run once: psql -f sql/ans_embed.sql
-- Column types are assumed; match them to public.form_respond_details if they differ.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.ans_embed (
    id                BIGSERIAL PRIMARY KEY,
    response_id       BIGINT NOT NULL,     -- form_respond_details.id
    ans_submission_id VARCHAR,             -- form_respond_details.ans_submissiom_id (varchar there)
    form_id           INT NOT NULL,
    department_id     INT,
    ques_id           BIGINT,
    user_id           BIGINT,
    ans_value         TEXT NOT NULL,
    embedding         vector NOT NULL,     -- untyped: models have different dims (same as ques_embed)
    model             TEXT NOT NULL,
    insert_dt         TIMESTAMP,           -- copied from the source row
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model, response_id)
);

-- ponytail: no ANN index (untyped vector can't take HNSW); search is always
-- filtered by form_id, so a per-form scan is fine. Add a typed column + HNSW
-- per model if a single form grows past ~1M answers.
CREATE INDEX IF NOT EXISTS ans_embed_form_model_idx ON public.ans_embed (form_id, model);
