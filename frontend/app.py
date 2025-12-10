import streamlit as st
import requests
import os


st.set_page_config(page_title="RAG Agentic Demo")
st.title("RAG Agentic — Demo (Streamlit)")

# Pega a URL da API do secrets.toml ou fallback para localhost
API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://127.0.0.1:8000"))

st.markdown("Upload documents (PDF/TXT/PPTX/PNG) and ask questions about the content.")

uploaded_file = st.file_uploader(
    "Upload file", type=["pdf", "txt", "md", "pptx", "png", "jpg", "jpeg"]
)
if uploaded_file:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    with st.spinner("Indexing..."):
        try:
            r = requests.post(
                f"{API_URL}/upload", files=files, timeout=30
            )  # timeout de 30s
            if r.ok:
                st.success("Document indexed successfully!")
            else:
                st.error(f"Failed to index document: {r.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {e}")

st.write("---")
question = st.text_input("Ask something about the indexed documents:")

if st.button("Send"):
    if not question.strip():
        st.warning("Please write a question.")
    else:
        with st.spinner("Searching..."):
            try:
                r = requests.post(
                    f"{API_URL}/ask", json={"question": question}, timeout=30
                )
                if r.ok:
                    data = r.json()
                    st.write("**Answer:**")
                    st.write(data.get("answer"))
                else:
                    st.error(f"Error accessing API: {r.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {e}")
