from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from llm_utils import llm
from retriever import pinecone_query

class State(TypedDict):
    user_query: str
    snippets: list[str]
    answer: str
    critique: str
    missing_keywords: str

def retrieve_kb(state: State):
    snippets = pinecone_query(state["user_query"], top_k=5)
    return {"snippets": snippets}

def generate_answer(state: State):
    context = "\n".join(state["snippets"])
    prompt = f"""Answer the question: {state['user_query']}\nUsing these KB snippets:\n{context}. Quote the knowlege base article number in the response like [KBKBXXX].
    **-For example the output should be in this format**
    *   **Monitor and Analyze:** Regularly monitor your system's performance to identify bottlenecks. This includes analyzing various metrics to understand where the system is struggling [KBKB002, KBKB012].
*   **Optimize Database Queries:** Ensure your database queries are efficient. This can involve indexing, query optimization, and avoiding unnecessary data retrieval [KBKB002].
*   **Implement Caching Strategies:** Caching is a crucial aspect of performance tuning.
    *   Employ effective caching strategies to store frequently accessed data, reducing the need to re-fetch it [KBKB013, KBKB003].
    *   Consider different caching mechanisms and their suitability for your application [KBKB013].
    *   Properly manage cache invalidation to ensure data consistency [KBKB003].
*   **Resource Management:** Optimize the use of system resources such as CPU, memory, and network bandwidth [KBKB002].
*   **Code Optimization:** Review and optimize your application code for efficiency. This can involve algorithmic improvements, reducing redundant operations, and efficient data structure usage [KBKB002, KBKB012].
*   **Load Balancing:** Distribute incoming traffic across multiple servers to prevent any single server from becoming overwhelmed [KBKB012].
*   **Asynchronous Operations:** Utilize asynchronous processing where appropriate to avoid blocking operations and improve responsiveness [KBKB012].
*   **Regular Tuning:** Performance tuning is an ongoing process, not a one-time task. Regularly review and adjust your tuning strategies as your system evolves [KBKB002, KBKB012].
    """
    print("prompt",prompt)
    res = llm.invoke(prompt)
    return {"answer": res.content}

def critique_answer(state: State):
    prompt = f"""Critique this answer for completeness:
Question: {state['user_query']}
Answer: {state['answer']}
Respond only with:
- COMPLETE
- REFINE: <missing keywords>"""
    res = llm.invoke(prompt)
    text = res.content.strip()
    print("critique content",text)
    if text.startswith("REFINE"):
        return {"critique": "REFINE", "missing_keywords": text.split(":", 1)[1].strip()}
    return {"critique": "COMPLETE"}

def refine_answer(state: State):
    new_query = f"{state['user_query']} {state['missing_keywords']}"
    new_snippets = pinecone_query(new_query, top_k=1)
    context = "\n".join(state["snippets"] + new_snippets)
    prompt = f"Refine your answer to: {state['user_query']}\nUsing this info:\n{context}"
    res = llm.invoke(prompt)
    print("refine_answer",res.content)
    return {"answer": res.content}

def build_graph():
    builder = StateGraph(State)
    builder.add_node("retrieve_kb", retrieve_kb)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("critique_answer", critique_answer)
    builder.add_node("refine_answer", refine_answer)

    builder.add_edge(START, "retrieve_kb")
    builder.add_edge("retrieve_kb", "generate_answer")
    builder.add_edge("generate_answer", "critique_answer")

    def decide(state):
        return state["critique"]

    builder.add_conditional_edges(
        "critique_answer",
        decide,
        {"COMPLETE": END, "REFINE": "refine_answer"}
    )
    builder.add_edge("refine_answer", END)
    return builder.compile()
