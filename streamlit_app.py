from __future__ import annotations

import streamlit as st

from rag_pi.config import load_settings
from rag_pi.rag import answer_question


st.set_page_config(page_title="RAG Propriete intellectuelle", layout="wide")
st.title("Assistant propriete intellectuelle")
st.caption("Reponses ancrees dans le Code de la propriete intellectuelle via Legifrance.")


@st.cache_resource
def get_settings():
    return load_settings()


try:
    settings = get_settings()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

question = st.chat_input("Posez une question sur la propriete intellectuelle")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les sources Legifrance..."):
            result = answer_question(question, settings)
        st.markdown(result.answer)
        if result.sources:
            st.markdown("**Sources retrouvees**")
            for source in result.sources:
                st.markdown(f"- [{source['title']}]({source['url']})")

    st.session_state.messages.append({"role": "assistant", "content": result.answer})
