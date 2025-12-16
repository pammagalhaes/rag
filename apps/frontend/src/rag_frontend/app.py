import streamlit as st
import requests
import os



st.set_page_config(page_title="RAG Agentic Demo")
st.title("RAG Agentic — Demo (Streamlit)")

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://backend:8000"))

st.markdown("Upload documents (PDF/TXT/PPTX/PNG) and ask questions about the content.")

uploaded_file = st.file_uploader("Upload file", type=['pdf','txt','md','pptx','png','jpg','jpeg'])
if uploaded_file:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    with st.spinner("Indexing..."):
        try:
            r = requests.post(f"{API_URL}/upload", files=files, timeout=30)  # timeout de 30s
            if r.ok:
                st.success("Document indexed successfully!")
            else:
                st.error(f"Failed to index document: {r.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {e}")

st.write("---")
st.subheader("Chat with your documents")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input box
prompt = st.chat_input("Ask something about the indexed content...")

if prompt:
    # Save user message
    st.session_state["messages"].append({
        "role": "user",
        "content": prompt
    })

    # Payload sent to backend
    payload = {
        "question": prompt,
        "history": st.session_state["messages"]
    }

    with st.spinner("Generating answer..."):
        try:
            response = requests.post(
                f"{API_URL}/chat",
                json=payload,
                timeout=60
            )

            if response.ok:
                answer = response.json().get("answer", "")

                # Save assistant response
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer
                })

                # Display assistant message
                with st.chat_message("assistant"):
                    st.write(answer)

            else:
                st.error(f"API error: {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

