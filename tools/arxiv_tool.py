import arxiv
from schemas import PaperSummary
from langsmith import traceable
import time

@traceable(name="search_arxiv")
def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    time.sleep(3)
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3,
        num_retries=2
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for paper in client.results(search):
        results.append({
            "arxiv_id": paper.entry_id.split("/")[-1],
            "title": paper.title,
            "abstract": paper.summary,
            "published": str(paper.published.date()),
            "authors": [a.name for a in paper.authors[:3]],
            "url": paper.pdf_url
        })
    return results

@traceable(name="fetch_paper_abstract")
def fetch_paper_abstract(arxiv_id: str) -> dict:
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search))
    return {
        "arxiv_id": arxiv_id,
        "title": paper.title,
        "abstract": paper.summary,
        "published": str(paper.published.date()),
        "authors": [a.name for a in paper.authors[:3]],
        "url": paper.pdf_url
    }