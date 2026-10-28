-- Question embeddings written by form_embedder.py. Run once: psql -f sql/ques_embed.sql
-- Mirrors the table in the old parakhdb.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.ques_embed (
    id                     BIGSERIAL PRIMARY KEY,
    form_id                INT,
    department_id          INT,
    ques_id                BIGINT,
    question_caption       TEXT,
    question_caption_hindi TEXT,
    embedding_en           vector,   -- untyped: models have different dims
    embedding_hi           vector,
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model                  TEXT DEFAULT 'bge-m3',
    CONSTRAINT unique_model_ques_id UNIQUE (model, ques_id)  -- form_embedder's ON CONFLICT needs this
);
