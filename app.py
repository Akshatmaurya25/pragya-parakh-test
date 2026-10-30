"""Streamlit chat UI for the Parakh master agent.

Run: streamlit run app.py
"""

import streamlit as st

from parakh.agent.llm import LLMS
from parakh.agent.master_agent import MasterAgent
from parakh.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS
from parakh.questions import get_form_questions, list_forms

st.set_page_config(page_title="Parakh Assistant", page_icon="💬")
st.title("Parakh Assistant")

# Settings: first LLM (NVIDIA) and Vyakyarth are the defaults.
llm_name = st.sidebar.selectbox("LLM", list(LLMS))
embedding_model = st.sidebar.selectbox(
    "Embedding model",
    list(EMBEDDING_MODELS),
    index=list(EMBEDDING_MODELS).index(DEFAULT_EMBEDDING_MODEL),
)
if "agent" not in st.session_state or st.sidebar.button("Clear chat"):
    st.session_state.agent = MasterAgent()
    st.session_state.messages = []


def show_details(matches, trace, answers=()):
    if not matches and not answers and not trace.get("steps"):
        return
    photos = [
        (url, a.get("question_caption") or a.get("question_captions") or "")
        for a in answers
        for url in a.get("photos") or []
    ]
    if photos:
        with st.expander(f"Photos ({len(photos)})", expanded=True):
            st.image([u for u, _ in photos], caption=[c for _, c in photos], width=180)
    with st.expander("Matches & tool calls"):
        if answers:
            st.caption("Answers")
            st.dataframe(answers, hide_index=True)
        if matches:
            st.dataframe(
                [
                    {
                        "form": m.get("form_id"),
                        "ques": m.get("ques_id"),
                        "question": m.get("question_caption"),
                        "hindi": m.get("question_caption_hindi"),
                        "similarity": m.get("best_similarity"),
                    }
                    for m in matches
                ],
                hide_index=True,
            )
        st.json(trace, expanded=False)


@st.cache_data(ttl=300)
def cached_forms(model):
    return list_forms(model)


@st.dialog("Form questions", width="large")
def show_form(form_id, form_name):
    st.subheader(f"{form_id} · {form_name or 'Unnamed form'}")
    questions = get_form_questions(form_id)
    if not questions:
        st.info("No questions found for this form.")
        return
    st.dataframe(
        [
            {
                "sno": i,
                "ques_id": q.get("ques_id"),
                "question": q.get("question_captions"),
                "hindi": q.get("question_caption_hindi"),
            }
            for i, q in enumerate(questions, 1)
        ],
        hide_index=True,
    )


chat_tab, forms_tab = st.tabs(["Chat", "Forms"])

with forms_tab:
    try:
        forms = cached_forms(embedding_model)
    except Exception as error:
        st.error(f"Could not load forms: {error}")
        forms = []
    if not forms:
        st.info("No forms embedded with this model yet. Run: python -m scripts.embed")
    else:
        widths = [1, 2, 5, 2, 2]
        headers = ["S.No", "Form ID", "Form name", "Questions", "View"]
        for col, label in zip(st.columns(widths), headers):
            col.markdown(f"**{label}**")
        for i, form in enumerate(forms, 1):
            sno, fid, name, count, view = st.columns(widths, vertical_alignment="center")
            sno.write(str(i))
            fid.write(str(form["form_id"]))
            name.write(form["form_name"] or "—")
            count.write(str(form["questions"]))
            if view.button("View", key=f"view_{form['form_id']}"):
                show_form(form["form_id"], form["form_name"])

with chat_tab:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                show_details(msg["matches"], msg["trace"], msg["answers"])
    prompt = st.chat_input("Ask about inspection forms (English / हिंदी)")

if prompt:
    agent = st.session_state.agent
    agent.llm = LLMS[llm_name]
    agent.embedding_model = embedding_model
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        try:
            result = agent.handle(prompt)
            reply = {
                "content": result.message,
                "matches": result.matches,
                "trace": result.trace,
                "answers": result.answers,
            }
        except Exception as error:  # DB down, model download failed, ...
            reply = {"content": f"Error: {error}", "matches": [], "trace": {}, "answers": []}
    st.session_state.messages.append({"role": "assistant", **reply})
    st.rerun()
