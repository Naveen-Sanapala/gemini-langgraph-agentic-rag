import streamlit as st
from rag_graph import build_graph
import requests
# Initialize the LangGraph pipeline
st.set_page_config(page_title="Agentic RAG Demo", layout="wide")
st.title("Agentic RAG System")

# User input
query = st.text_input("Enter your question:")

if st.button("Ask Gemini"):
    if not query.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner("Thinking... retrieving + reasoning..."):
            # Send POST request to FastAPI endpoint
            try:
                response = requests.post(
                    "http://localhost:8000/ask",
                    json={"question": query}
                )
                response.raise_for_status()
                # Run the graph
                result = response.json()
                answer = result.get("answer", "No answer returned.")
                
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the API: {e}")
        st.success("Answer generated!")
        st.markdown("###Response")
        st.write(answer)
