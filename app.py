"""Streamlit chat UI for the Parakh master agent.

Run: streamlit run app.py
"""

import streamlit as st

from embedder import MODEL_OPTIONS
from master_agent import LLMS, MasterAgent

EMBEDDING_MODELS = dict(MODEL_OPTIONS.values())  # display name -> model id

st.set_page_config(page_title="Parakh Assistant", page_icon="💬")
st.title("Parakh Assistant")

if "agent" not in st.session_state or st.sidebar.button("Clear chat"):
    st.session_state.agent = MasterAgent()
    st.session_state.messages = []


def show_details(matches, trace):
    if not matches and not trace.get("steps"):
        return
    with st.expander("Matches & tool calls"):
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


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_details(msg["matches"], msg["trace"])

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
            reply = {"content": result.message, "matches": result.matches, "trace": result.trace}
        except Exception as error:  # DB down, model download failed, ...
            reply = {"content": f"Error: {error}", "matches": [], "trace": {}}
    st.session_state.messages.append({"role": "assistant", **reply})
    st.rerun()
