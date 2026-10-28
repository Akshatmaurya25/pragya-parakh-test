"""Streamlit chat UI for the Parakh master agent.

Run: streamlit run app.py
"""

import json

import streamlit as st

from db_conn import execute
from embedder import MODEL_OPTIONS
from master_agent import LLMS, MasterAgent
from tools import get_form_questions

EMBEDDING_MODELS = dict(MODEL_OPTIONS.values())  # display name -> model id

st.set_page_config(page_title="Parakh Assistant", page_icon="💬")
st.title("Parakh Assistant")

if "agent" not in st.session_state or st.sidebar.button("Clear chat"):
    st.session_state.agent = MasterAgent()
    st.session_state.messages = []


def show_details(matches, trace, answers=()):
    if not matches and not answers and not trace.get("steps"):
        return
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
def embedded_forms():
    """Forms present in ques_embed, with a name when the form master has one."""
    try:
        return json.loads(execute(
            """
            SELECT qe.form_id, MAX(f.form_name) AS form_name
            FROM public.ques_embed AS qe
            LEFT JOIN public.depertment_form_master AS f ON f.form_id = qe.form_id
            GROUP BY qe.form_id
            ORDER BY qe.form_id
            """
        ))
    except Exception:  # no form master table in this DB: list ids only
        return json.loads(execute(
            "SELECT DISTINCT form_id, NULL AS form_name FROM public.ques_embed ORDER BY form_id"
        ))


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
        forms = embedded_forms()
    except Exception as error:
        st.error(f"Could not load forms: {error}")
        forms = []
    if not forms:
        st.info("No embedded forms yet. Run form_embedder.py first.")
    else:
        widths = [1, 2, 6, 2]
        for col, label in zip(st.columns(widths), ["S.No", "Form ID", "Form name", "View"]):
            col.markdown(f"**{label}**")
        for i, form in enumerate(forms, 1):
            sno, fid, name, view = st.columns(widths, vertical_alignment="center")
            sno.write(str(i))
            fid.write(str(form["form_id"]))
            name.write(form["form_name"] or "—")
            if view.button("View", key=f"view_{form['form_id']}"):
                show_form(form["form_id"], form["form_name"])

with chat_tab:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                show_details(msg["matches"], msg["trace"], msg.get("answers", ()))

    with st.container():  # inline, so the selectors sit below it
        prompt = st.chat_input("Ask about inspection forms (English / हिंदी)")

    llm_col, emb_col = st.columns(2)
    llm_name = llm_col.selectbox("LLM", list(LLMS), label_visibility="collapsed")
    emb_name = emb_col.selectbox("Embedding model", list(EMBEDDING_MODELS), label_visibility="collapsed")

if prompt:
    agent = st.session_state.agent
    agent.llm = LLMS[llm_name]
    agent.embedding_model = EMBEDDING_MODELS[emb_name]
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
            reply = {"content": f"Error: {error}", "matches": [], "trace": {}}
    st.session_state.messages.append({"role": "assistant", **reply})
    st.rerun()
