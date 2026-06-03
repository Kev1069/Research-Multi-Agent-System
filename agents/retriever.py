from langsmith import traceable
from schemas import PaperSummary
from tools.arxiv_tool import search_arxiv, fetch_paper_abstract
from config import get_instructor_client

from memory.store import (
    query_semantic_memory,
    write_to_semantic_memory,
    write_episodic_event,
    create_session
)

RETRIEVER_PROMPT = """You are a research retrieval agent. Given a query, you will:
1. Analyze the search results provided
2. Extract structured summaries for the most relevant papers
3. Score relevance honestly - not everything will be highly relevant
4. Be concise in key_contribution = one or two sentences maximum

Query: {query}

Papers found:
{papers}

Extract a PaperSummary for each paper. Be critical with relevance_score - only score above 0.8 if directly relevant to the query."""


@traceable(name="retriever_agent")
def retriever_agent(query: str, max_results: int = 5) -> list[PaperSummary]:
    client, model = get_instructor_client()

    # Check memory first
    cached = query_semantic_memory(query, n_results=3)
    if cached:
        print(f"[Memory] Found {len(cached)} cached results for similar query")

    # Fetch from arXiv
    raw_papers = search_arxiv(query, max_results=max_results)
    if not raw_papers:
        return []

    # Format and extract
    papers_text = ""
    for i, p in enumerate(raw_papers):
        papers_text += f"""
Paper {i+1}:
Title: {p['title']}
ArXiv ID: {p['arxiv_id']}
Published: {p['published']}
Authors: {', '.join(p['authors'])}
Abstract: {p['abstract'][:500]}...
---"""
        
    # from instructor import Maybe
    from pydantic import BaseModel

    class PaperSummaryList(BaseModel):
        papers: list[PaperSummary]
        
    result = client.chat.completions.create(
        model=model,
        response_model=PaperSummaryList,
        messages=[{
            "role": "user",
            "content": RETRIEVER_PROMPT.format(query=query, papers=papers_text)  
        }]
    )

    summaries = result.papers
    relevant = [s for s in summaries if s.is_relevant]

    # Write to memory
    session_id = create_session(query, len(summaries), len(relevant))
    write_to_semantic_memory(relevant, query)
    write_episodic_event(session_id, "retriever", "papers_retrieved", {
        "query": query,
        "total": len(summaries),
        "relevant": len(relevant),
        "arxiv_ids": [s.arxiv_id for s in relevant]
    })

    return summaries