# gemini-langgraph-agentic-rag
This project implements an Agentic Retrieval-Augmented Generation (RAG) pipeline powered by Google Gemini, LangGraph, and Pinecone — exposed via a FastAPI backend and a Streamlit frontend.

The system retrieves relevant context from a knowledge base (KB), generates an initial answer, critiques it, and if required, refines it for improved completeness — delivering citation-backed, self-correcting AI responses.

# Flow
User Query → Retriever → LLM Answer → Self-Critique → Refinement (if needed) → Final Answer

Node	            Description
Retriever Node	    Retrieves top 5 KB snippets from Pinecone using Gemini embeddings.
LLM Answer Node	    Uses Gemini (Vertex AI) to generate initial answer with citations.
Self-Critique Node	Evaluates answer completeness. Returns COMPLETE or REFINE.
Refinement Node	    If refinement is required, pulls one more snippet and regenerates the final answer.

# Project structure
gemini_langgraph_agentic_rag
 ┣ .env
 ┣ requirements.txt
 ┣ self_critique_loop_dataset.json
 ┣ agentic_rag_notebook.ipynb  # python notebook showcasing the entire flow 
 ┣ main.py                # FastAPI backend
 ┣ gcp_embeddings.py           # Embedding generator (Gemini embeddings)
 ┣ retriever.py           # Pinecone retriever
 ┣ llm_utils.py           # LLM initialization (Gemini)
 ┣ rag_graph.py           # LangGraph workflow definition
 ┣ ui.py       # Streamlit frontend UI
 ┗ README.md              # Documentation
 
# Setup Instructions 
1. Clone repository
git clone https://github.com/Naveen-Sanapala/gemini-langgraph-agentic-rag.git
cd gemini-langgraph-agentic-rag

2. Install Dependencies
pip install -r requirements.txt

3. Add Environment variables in .env file
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-rag-index

4. Preprocess and Index KB
python retriever.py

5. Start Backend
uvicorn main:app --reload --port 8000

6. Run streamlit front end
streamlit run ui.py






