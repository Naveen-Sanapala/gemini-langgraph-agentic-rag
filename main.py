from fastapi import FastAPI
from pydantic import BaseModel
from rag_graph import build_graph

app = FastAPI()
graph = build_graph()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    result = graph.invoke({"user_query": query.question})
    return {"question": query.question, "answer": result["answer"]}
