from __future__ import annotations

import streamlit as st
import subprocess

from rag_pi.config import load_settings
from rag_pi.rag import answer_question

# Interface utilisateur simple en Streamlit.
# Permet de poser une question et de voir la réponse avec les sources et scores.


st.set_page_config(page_title="RAG Propriete intellectuelle", layout="wide")
st.title("Assistant propriete intellectuelle")
st.caption("Reponses ancrees dans le Code de la propriete intellectuelle via Legifrance.")


@st.cache_resource
def get_settings():
    # Charge la configuration une seule fois lors du premier appel.
    return load_settings()


try:
    settings = get_settings()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

@st.cache_resource
def run_ingest():
    result = subprocess.run(
        ["python", "-m", "scripts.ingest_legifrance", "--reset"],
        capture_output=True,
        text=True
    )
    return result.returncode

run_ingest()

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
            # Appel au pipeline RAG pour récupérer les documents et générer la réponse.
            result = answer_question(question, settings)
        st.markdown(result.answer)
        if result.sources:
            st.markdown("**Sources retrouvees**")
            for index, source in enumerate(result.sources, start=1):
                score = result.doc_scores[index - 1] if index - 1 < len(result.doc_scores) else None
                if score is not None:
                    st.markdown(f"- [{source['title']}]({source['url']}) — score: {score:.3f}")
                else:
                    st.markdown(f"- [{source['title']}]({source['url']})")

            with st.expander("Voir les détails des documents récupérés"):
                for index, doc in enumerate(result.context_docs, start=1):
                    score = result.doc_scores[index - 1] if index - 1 < len(result.doc_scores) else None
                    title = doc.metadata.get("article_title") or doc.metadata.get("title") or "Source"
                    source_url = doc.metadata.get("source", "")
                    st.markdown(f"**Document {index} - {title}**")
                    if score is not None:
                        st.markdown(f"- Score de similarité: **{score:.3f}**")
                    if source_url:
                        st.markdown(f"- URL: {source_url}")
                    st.write(doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""))

    st.session_state.messages.append({"role": "assistant", "content": result.answer})
