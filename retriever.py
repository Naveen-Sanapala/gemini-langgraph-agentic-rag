import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from gcp_embeddings import Embedder

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "self-critique-index")

def init_pinecone():
    """Initialize Pinecone client and create index if not exists (Serverless)."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"🪶 Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"✅ Pinecone index already exists: {INDEX_NAME}")

    return pc.Index(INDEX_NAME)

def build_index():
    """Load dataset, create embeddings with Gemini, and upsert into Pinecone."""
    index = init_pinecone()
    embedder = Embedder()

    with open("self_critique_loop_dataset.json") as f:
        data = json.load(f)[:30]

    texts = [d["answer_snippet"] for d in data]
    ids = [d["doc_id"] for d in data]
    vectors = embedder.embed_texts(texts)

    to_upsert = [
        {"id": ids[i], "values": vectors[i].tolist(), "metadata": {"question": data[i]["question"]}}
        for i in range(len(ids))
    ]

    index.upsert(vectors=to_upsert)
    print(f"✅ Indexed {len(ids)} documents into Pinecone")

def pinecone_query(query: str, top_k: int = 5):
    """Query top-k snippets from Pinecone."""
    index = init_pinecone()
    embedder = Embedder()
    query_vec = embedder.embed_texts([query])[0].tolist()

    res = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    snippets = [f"{match['metadata']['question']} [KB{match['id']}]" for match in res['matches']]
    return snippets

if __name__ == "__main__":
    print("🚀 Starting Pinecone setup and indexing...")
    build_index()