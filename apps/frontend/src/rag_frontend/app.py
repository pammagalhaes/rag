import streamlit as st
import requests
import os

st.set_page_config(page_title="RAG Agentic Demo")
st.title("RAG Agentic")

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://backend:8000"))

st.markdown(
    "This app uses a pre-built RAG knowledge base. Ask questions and get answers based on the indexed documents."
)

st.write("---")
st.subheader("Ask a question")

question = st.text_input("Enter your question", "O que é um modelo de classificação?")

if st.button("Ask") and question:
    payload = {"question": question}
    with st.spinner("Getting answer..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json=payload,
                timeout=60
            )
            if response.ok:
                answer = response.json().get("answer", "No answer returned.")
                st.markdown("**Answer:**")
                st.write(answer)
            else:
                st.error(f"API error: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

st.info(
    "Note: this interface only queries the existing RAG index. Upload is disabled in this demo."
)

