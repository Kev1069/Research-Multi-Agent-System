from openai import OpenAI
from schemas import PaperSummary
from dotenv import load_dotenv
import os
from config import get_instructor_client
from langsmith import traceable

load_dotenv()

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://apac.api.smith.langchain.com")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "Research MAS")

# client, model = get_instructor_client()

# @traceable(name="smoke_test_paper_summary")
# def run_smoke_test():
#     result = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         response_model=PaperSummary,
#         messages=[{
#             "role": "user",
#             "content": "Extract paper summary..."
#         }]
#     )
#     return result

# result = run_smoke_test()
# print(result.model_dump_json(indent=2))

# # add to main.py temporarily
# from tools.arxiv_tool import search_arxiv
# results = search_arxiv("efficient LLM inference 2024", max_results=3)
# for r in results:
#     print(r["title"], r["published"])



# from agents.retriever import retriever_agent

# summaries = retriever_agent("efficient inference optimization large language models quantization distillation", max_results=5)
# relevant = [s for s in summaries if s.is_relevant]
# print(f"{len(relevant)}/{len(summaries)} papers relevant")
# for s in relevant:
#     print(f"\n{s.title}")
#     print(f"Relevance: {s.relevance_score} | Confidence: {s.confidence}")
#     print(f"Contribution: {s.key_contribution}")



# from agents.retriever import retriever_agent
# from memory.store import query_semantic_memory

# # First run — fetches from arXiv
# summaries = retriever_agent(
#     "efficient inference optimization large language models quantization distillation",
#     max_results=5
# )
# relevant = [s for s in summaries if s.is_relevant]
# print(f"\nFetched: {len(relevant)} relevant papers")

# # Second run — should hit memory
# print("\n--- Running same query again ---")
# cached = query_semantic_memory(
#     "efficient inference optimization large language models quantization distillation"
# )
# print(f"Memory returned: {len(cached)} cached papers")
# for c in cached:
#     print(f"  - {c['metadata']['title']}")



import sqlite3
conn = sqlite3.connect("./memory/episodic.db")
sessions = conn.execute("SELECT * FROM sessions").fetchall()
events = conn.execute("SELECT * FROM events").fetchall()
print("Sessions:", sessions)
print("Events:", events)
conn.close()