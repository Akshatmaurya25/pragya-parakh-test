import json

from db_conn import execute, execute_many
from embedder import embedder


QUESTION_CAPTIONS_QUERY = """
SELECT ques_id, question_captions, form_id, question_caption_hindi,
       embedding_en, embedding_hi
FROM public.questionl_details
WHERE form_id = %s
"""


def _embed_column(rows, column):
	indices = []
	captions = []
	for index, row in enumerate(rows):
		caption = row.get(column)
		if caption is None:
			continue
		if not isinstance(caption, str):
			raise TypeError(f"{column} must contain only strings or None")
		indices.append(index)
		captions.append(caption)

	if not captions:
		return {}

	vectors = embedder(captions)
	if len(vectors) != len(indices):
		raise ValueError(f"Expected {len(indices)} embeddings for {column}")
	return {
		index: vector.tolist()
		for index, vector in zip(indices, vectors)
	}


def generate_and_print_embeddings(form_id=299):
	"""Embed and print English/Hindi captions for a form; do not update the DB."""
	rows = json.loads(execute(QUESTION_CAPTIONS_QUERY, (form_id,)))
	if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
		raise TypeError("Database query must return a JSON list of row objects")

	english_embeddings = _embed_column(rows, "question_captions")
	hindi_embeddings = _embed_column(rows, "question_caption_hindi")
	results = []

	for index, row in enumerate(rows):
		result = row.copy()
		result["generated_embedding_en"] = english_embeddings.get(index)
		result["generated_embedding_hi"] = hindi_embeddings.get(index)
		results.append(result)

		print(f"Question ID: {row.get('ques_id')}")
		print(f"English caption: {row.get('question_captions')!r}")
		print(f"English embedding: {result['generated_embedding_en']}")
		print(f"Hindi caption: {row.get('question_caption_hindi')!r}")
		print(f"Hindi embedding: {result['generated_embedding_hi']}")
		print()

	return results


def _to_pgvector_literal(vector):
	return "[" + ",".join(str(float(value)) for value in vector) + "]"


def generate_and_insert_embeddings(form_id=299):
	"""Generate bilingual caption embeddings and save them for a form.

	The target embedding columns are expected to use the PostgreSQL pgvector type.
	Rows without a caption for one language retain their existing embedding.
	"""
	rows = json.loads(execute(QUESTION_CAPTIONS_QUERY, (form_id,)))
	if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
		raise TypeError("Database query must return a JSON list of row objects")

	english_embeddings = _embed_column(rows, "question_captions")
	hindi_embeddings = _embed_column(rows, "question_caption_hindi")
	updates = []

	for index, row in enumerate(rows):
		ques_id = row.get("ques_id")
		if ques_id is None:
			raise ValueError("Every question row must have a ques_id")
		english = english_embeddings.get(index)
		hindi = hindi_embeddings.get(index)
		if english is None and hindi is None:
			continue
		updates.append((
			_to_pgvector_literal(english) if english is not None else None,
			_to_pgvector_literal(hindi) if hindi is not None else None,
			ques_id,
			form_id,
		))

	if not updates:
		return 0

	return execute_many(
		"""
		UPDATE public.questionl_details
		SET embedding_en = COALESCE(%s::vector, embedding_en),
		    embedding_hi = COALESCE(%s::vector, embedding_hi)
		WHERE ques_id = %s AND form_id = %s
		""",
		updates,
	)


def retrieve_top_k_similar_questions(query_text, form_id=299, top_k=3):
	"""Find the most similar questions using either stored language embedding."""
	if not isinstance(query_text, str) or not query_text.strip():
		raise ValueError("query_text must be a non-empty string")
	if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
		raise ValueError("top_k must be a positive integer")

	query_vector = _to_pgvector_literal(embedder(query_text.strip()).tolist())
	query = """
	WITH candidates AS (
		SELECT ques_id, question_captions, question_caption_hindi, form_id,
		       'en' AS matched_language,
		       1 - (embedding_en <=> %s::vector) AS similarity
		FROM public.questionl_details
		WHERE form_id = %s AND embedding_en IS NOT NULL
		UNION ALL
		SELECT ques_id, question_captions, question_caption_hindi, form_id,
		       'hi' AS matched_language,
		       1 - (embedding_hi <=> %s::vector) AS similarity
		FROM public.questionl_details
		WHERE form_id = %s AND embedding_hi IS NOT NULL
	),
	best_match_per_question AS (
		SELECT ques_id, question_captions, question_caption_hindi, form_id,
		       matched_language, similarity,
		       ROW_NUMBER() OVER (
		           PARTITION BY ques_id
		           ORDER BY similarity DESC, matched_language
		       ) AS language_rank
		FROM candidates
	)
	SELECT ques_id, question_captions, question_caption_hindi, form_id,
	       matched_language, similarity
	FROM best_match_per_question
	WHERE language_rank = 1
	ORDER BY similarity DESC, ques_id
	LIMIT %s
	"""
	return json.loads(execute(
		query,
		(query_vector, form_id, query_vector, form_id, top_k),
	))


def test_retrieve_top_3(query_text, form_id=299):
	"""Search a form with a text query and print the three closest questions."""
	results = retrieve_top_k_similar_questions(
		query_text,
		form_id=form_id,
		top_k=3,
	)
	for rank, result in enumerate(results, start=1):
		print(
			f"{rank}. Question ID: {result['ques_id']} "
			f"(similarity: {result['similarity']:.4f}, "
			f"matched: {result['matched_language']})"
		)
		print(f"   English caption: {result['question_captions']!r}")
		print(f"   Hindi caption: {result['question_caption_hindi']!r}")
	return results