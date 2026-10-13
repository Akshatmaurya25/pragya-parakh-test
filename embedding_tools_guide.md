# Question Embedding Tools: Plain-Language Guide

This guide explains what [`form_embedder.py`](./form_embedder.py) and
[`query_resolver.py`](./query_resolver.py) do, how their functions work, and
how to run them. In short:

- `form_embedder.py` reads questions for one form, turns their English and
  Hindi captions into numeric vectors, and saves the captions and vectors in
  `public.ques_embed`.
- `query_resolver.py` turns a user-entered question or phrase into a vector,
  compares it with saved vectors, and prints the closest matches.

## A few terms in everyday language

- **Caption**: the text of a question, in English or Hindi.
- **Embedding/vector**: a list of numbers representing the meaning of text.
  Similar meanings tend to have vectors that are close to each other.
- **Form**: a group of questions, identified by `form_id`.
- **Similarity score**: a score calculated from the distance between two
  vectors. The resolver displays the score multiplied by 100 and adds a `%`
  sign. It is a ranking score, not a guaranteed probability that the match is
  correct.
- **`ques_embed`**: the table used by these files to store question captions
  and their English/Hindi vectors.

## How the files work together

Both files use two existing modules:

- [`embedder.py`](./embedder.py) loads the multilingual `BAAI/bge-m3` model and
  converts text into 1024-number vectors. The model is cached after its first
  load so it can be reused in the same Python process.
- [`db_conn.py`](./db_conn.py) connects to PostgreSQL and provides database
  query and insert helpers. Connection settings are read from environment
  variables such as `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, and
  `PGPASSWORD`.

## `form_embedder.py`: create and save embeddings

### What happens when you run it

Run:

```text
python form_embedder.py
```

The program asks for a form ID. It then:

1. Checks that the form ID is a positive whole number.
2. Reads that form's questions from `public.questionl_details`, ordered by
   `qus_ordering` and then `id`.
3. Embeds the available English captions and Hindi captions separately.
4. Matches each generated vector back to the question it came from.
5. Inserts each question's form, department, question ID, captions, vectors,
   and current timestamps into `public.ques_embed`.
6. Prints progress and records progress in `form_embedder.log`, next to the
   Python file.

### Functions, in order of their role

#### Constants: `QUESTION_QUERY` and `INSERT_QUERY`

`QUESTION_QUERY` selects `form_id`, `department_id`, `ques_id`,
`question_captions`, and `question_caption_hindi` from
`public.questionl_details` for the requested form. The `%s` placeholder is
filled safely by the database helper rather than by inserting the form ID
directly into SQL text.

`INSERT_QUERY` describes how to add rows to `public.ques_embed`. The database
generates `id` automatically. The SQL fills `created_at` and `updated_at` with
the current time. `::vector` tells PostgreSQL to interpret each supplied
embedding string as a pgvector value.

#### `configure_logging()`

Sets up a log file named `form_embedder.log` in the same folder as
`form_embedder.py`. It avoids adding the same file handler more than once if
logging is configured repeatedly in one Python process. Log messages include
their time and severity.

#### `log_message(message, level=logging.INFO)`

Sends a message to two places: the terminal and the log file. The `level`
indicates whether the message is ordinary progress or an error.

#### `fetch_form_questions(form_id)`

Runs `QUESTION_QUERY` for the given form. The database helper returns its rows
as JSON text, so this function decodes that text into a Python list of
dictionaries and checks that the result has the expected shape. It logs how
many questions were fetched.

#### `_embed_caption_column(rows, column)`

This helper handles one caption language at a time:

1. Looks at the requested caption field in every question row.
2. Skips captions that are `None` or an empty string.
3. Checks that every other caption is text.
4. Sends all available captions to `embedder()` together.
5. Associates each returned vector with the original question row's position.

If there are no captions for that language, it returns no vectors. That
language's embedding will be stored as SQL `NULL` for those rows.

The leading underscore is a Python naming convention meaning this is intended
as an internal helper rather than the main public operation.

#### `_to_pgvector_literal(vector)`

Converts a list of numbers into text formatted like `[0.1,0.2,...]`, which
PostgreSQL can cast to its `vector` type. If there is no vector, it returns
`None`, which the database stores as `NULL`.

#### `prepare_embedding_rows(form_id, questions)`

Calls `_embed_caption_column()` for the English field
(`question_captions`) and the Hindi field (`question_caption_hindi`). It then
builds one database-ready tuple for every question. Each tuple contains:

1. Form ID
2. Department ID
3. Question ID
4. English caption
5. Hindi caption
6. English vector, or `NULL`
7. Hindi vector, or `NULL`

It stops with an error if a question has no `ques_id`, because the question ID
is needed to identify the question in the destination table.

#### `insert_form_embeddings(insert_rows)`

If there are no rows to save, logs that fact and returns `0`. Otherwise it
passes all tuples and `INSERT_QUERY` to `execute_many()` in `db_conn.py`. That
helper executes the insert and commits the transaction. The function logs and
returns the database's inserted-row count.

#### `embed_form(form_id)`

This is the main reusable function. It checks the form ID, configures logging,
fetches the questions, prepares their embeddings, inserts them, and returns
the inserted-row count.

Other Python code can call it directly:

```python
from form_embedder import embed_form

inserted_count = embed_form(299)
print(f"Inserted {inserted_count} rows")
```

#### `main()`

Provides the command-line interaction: asks for a form ID, converts it to an
integer, and calls `embed_form()`. Invalid form IDs/data and unexpected
failures are logged; failures are re-raised so they are not silently treated
as success.

The `if __name__ == "__main__":` block calls `main()` only when the file is
run directly. Importing the module from another Python file does not prompt
the user automatically.

## `query_resolver.py`: find similar saved questions

### What happens when you run it

Run:

```text
python query_resolver.py
```

The program asks for:

1. A search phrase in English or Hindi.
2. A form ID, or a blank value to search across all forms.

It embeds the search phrase, compares that vector with saved English and Hindi
vectors in `public.ques_embed`, and prints up to the 3 closest vector matches.
It prints the row details but leaves out the large vector columns. The
similarity appears as a percentage-like score, for example `"87.65%"`.

### Functions, in order of their role

#### `_to_pgvector_literal(vector)`

Formats the search vector as text that PostgreSQL can interpret as a pgvector
value. This is the same kind of conversion used for saved vectors in
`form_embedder.py`.

#### `_build_search_query(form_id)`

Builds the SQL used for matching. It searches the English and Hindi embedding
columns separately, skipping rows where that language's vector is `NULL`.
When a form ID was supplied, it adds a filter to both language searches. With
no form ID, it searches all forms.

The `<=>` operator calculates cosine distance in pgvector. The SQL subtracts
that distance from 1 to make a score where larger values sort nearer the top.
It combines the two language searches, joins them back to `ques_embed` to get
the complete stored row, sorts by score, and limits the result count.

The SQL uses placeholders for query vectors, form IDs, and result count. Those
values are supplied separately to the database helper.

#### `retrieve_top_k(query_text, form_id=None, top_k=3)`

This is the reusable search function. It:

1. Checks that the search phrase is not empty.
2. Checks that a provided form ID and result count are positive integers.
3. Creates one multilingual vector for the search phrase using `embedder()`.
4. Builds the matching SQL and its parameter list.
5. Runs the query and decodes the returned JSON into a list of row
   dictionaries.
6. Checks the returned data shape and returns the full rows.

The full rows include the stored embedding columns and the raw numeric
`similarity` value. The terminal printer filters those large/internal values;
the returned Python data is not filtered.

Example use from another Python file:

```python
from query_resolver import retrieve_top_k

matches = retrieve_top_k("आंगनवाड़ी केंद्र का निरीक्षण", form_id=299, top_k=5)
```

Leave out `form_id` (or pass `None`) to search all forms.

#### `resolve_query(query_text, form_id=None, top_k=3)`

Calls `retrieve_top_k()`, creates a separate display copy of each result, and
prints that copy as readable JSON. It removes `embedding_en`, `embedding_hi`,
and the raw `similarity` value from the printed copy. If a similarity value is
present, it adds `similarity_percentage`, formatted to two decimal places.
The original full results are returned unchanged.

For example, if the database score is `0.87654`, the terminal displays
`"similarity_percentage": "87.65%"`. This is just the score multiplied by 100;
it should not be interpreted as an 87.65% probability of correctness.

#### `_parse_form_id(value)`

Converts the terminal's form ID text into either a positive integer or `None`.
A blank input means “search all forms.” Non-numeric, zero, or negative values
raise a clear error instead.

#### `main()`

Prompts for the phrase and optional form ID, then calls `resolve_query()` with
the default of 3 results. To change the number of results when calling it from
Python code, provide `top_k`; the current terminal prompts do not ask for
`top_k`.

As with `form_embedder.py`, the `if __name__ == "__main__":` block means the
prompt runs only when the file is started directly.

## Important behavior to know

- **Each vector is a candidate.** The search ranks English and Hindi vectors
  independently. If both vectors belonging to one saved question rank in the
  top results, that same database row can appear more than once.
- **The terminal output is smaller than the returned data.** The resolver
  hides vectors only while printing. Code receiving the return value from
  `retrieve_top_k()` or `resolve_query()` still gets the vector fields and raw
  similarity.
- **Both languages use one query vector.** The `BAAI/bge-m3` embedding model is
  multilingual, so the phrase is embedded once and compared to both language
  columns.
- **Embedding inserts are additions.** `form_embedder.py` inserts new rows; it
  does not update or replace rows that are already in `ques_embed`. Running it
  again for the same form can therefore create duplicate question records
  unless the database schema or application has a separate duplicate-prevention
  rule.
- **The two files expect PostgreSQL pgvector.** Their SQL casts values to
  `vector` and uses the `<=>` distance operator, so the database must have the
  pgvector extension and compatible vector columns.
