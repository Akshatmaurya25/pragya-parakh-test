"""Core checks that need no database, model download or LLM (everything is faked).

Run: python -m tests.test_core      (or: python -m pytest tests)
"""

import json

import numpy as np

from parakh import embeddings, questions
from parakh.agent import master_agent
from parakh.agent.llm import extract_json
from parakh.agent.registry import call_tool
from parakh.answers import answer_text, image_urls


def test_answer_text():
    assert answer_text('[{"OptionID":"1","OptionText":"Rural"}]') == "Rural"
    assert answer_text('[{"OptionID":"1","OptionText":"A"},{"OptionID":"2","OptionText":"B"}]') == "A, B"
    assert answer_text('[{"OptionID":"","OptionText":""}]') == ""
    assert answer_text(" free text ") == "free text"
    assert answer_text("[not json") == "[not json"
    assert answer_text(None) == ""


def test_image_urls():
    base = "https://img.example"
    plain = '[{"OptionID":"jpg,77.42,23.24","OptionText":"Citizen/SurveyImages/9_1184_SurveyImg.jpg"}]'
    assert image_urls(plain, base) == [f"{base}/Citizen/SurveyImages/9_1184_SurveyImg.jpg"]
    nested = (
        r'[{"OptionID":"0","OptionText":"[[{\"answerValue\":{\"optionID\":\"png,79.89,22.02\",'
        r'\"optionText\":\"Tribal Affairs Department/SurveyImages/0_5338.png\"}}]]"}]'
    )
    assert image_urls(nested, base) == [f"{base}/Tribal%20Affairs%20Department/SurveyImages/0_5338.png"]
    assert image_urls(plain, "") == []
    assert image_urls('[{"OptionID":"1","OptionText":"Rural"}]', base) == []


def test_embed_unique():
    calls = []

    def fake_embed(texts, model):
        calls.append(list(texts))
        return np.array([[float(len(t))] for t in texts])

    original, embeddings.embed = embeddings.embed, fake_embed
    try:
        result = embeddings.embed_unique(["Yes", "No", "Yes", "हाँ"])
    finally:
        embeddings.embed = original
    assert calls == [["Yes", "No", "हाँ"]], calls
    assert result == {"Yes": "[3.0]", "No": "[2.0]", "हाँ": "[3.0]"}, result
    assert embeddings.embed_unique([]) == {}


def _fake_search(query, model, form_id=None, top_k=10):
    """q1 ranks first for every phrasing; q2 only for 'toilet'. en+hi rows repeat."""
    rows = [{"form_id": 1, "ques_id": 1, "question_caption": "Water?", "similarity": 0.9}]
    rows.append(dict(rows[0], similarity=0.8))  # same question, other language
    if "toilet" in query:
        rows.append({"form_id": 1, "ques_id": 2, "question_caption": "Toilet?", "similarity": 0.7})
    return rows


def _with_fake_search(func):
    original, questions.search_questions = questions.search_questions, _fake_search
    try:
        return func()
    finally:
        questions.search_questions = original


def test_search_questions_tool_fuses_and_dedupes():
    out = _with_fake_search(
        lambda: call_tool("search_questions", {"query": "water", "variants": ["toilet", "WATER"], "top_k": "5"})
    )
    assert out["ok"], out
    result = out["result"]
    assert [r["ques_id"] for r in result] == [1, 2], result
    assert result[0]["matched_queries"] == 2  # "WATER" deduped against "water"
    assert result[0]["best_similarity"] == 0.9


def test_call_tool_validation():
    assert not call_tool("nope", {})["ok"]
    assert "missing" in call_tool("search_questions", {})["error"]
    assert "unknown arguments" in call_tool("search_questions", {"query": "x", "bad": 1})["error"]
    assert "integer" in call_tool("get_form_questions", {"form_id": "abc"})["error"]


def test_extract_json():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_agent_tool_then_final():
    replies = iter([
        json.dumps({"action": "call_tool", "tool": "search_questions", "args": {"query": "water"}}),
        json.dumps({"action": "final", "message": "Question about water."}),
    ])
    agent = master_agent.MasterAgent(llm=lambda system, user, json_mode=True: next(replies))
    response = _with_fake_search(lambda: agent.handle("water?"))
    assert response.action == "answer" and response.message == "Question about water."
    assert response.matches[0]["ques_id"] == 1
    assert response.trace["steps"][0]["tool"] == "search_questions"
    assert len(agent.history) == 2


def test_agent_falls_back_when_llm_fails():
    def broken_llm(system, user, json_mode=True):
        raise ConnectionError("down")

    response = _with_fake_search(lambda: master_agent.MasterAgent(llm=broken_llm).handle("water"))
    assert response.trace["fallback"] is True
    assert response.action == "answer" and response.matches[0]["ques_id"] == 1


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} tests ok")
