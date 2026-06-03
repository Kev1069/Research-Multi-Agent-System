import chromadb
import sqlite3
import json
from datetime import datetime
from schemas import PaperSummary
from langsmith import traceable

# Establishing semantic memory with ChromaDB
chroma_client = chromadb.PersistentClient(path="./memory/chroma_db")
collection = chroma_client.get_or_create_collection(
  name="papers"
)

@traceable(name="write_semantic_memory")
def write_to_semantic_memory(papers: list[PaperSummary], query: str):
    for paper in papers:
        if not paper.is_relevant:
            continue
        collection.upsert(
            ids=[paper.arxiv_id],
            documents=[f"{paper.title}. {paper.key_contribution}"],
            metadatas=[{
                "title": paper.title,
                "topic": paper.topic,
                "relevance_score": paper.relevance_score,
                "confidence": paper.confidence,
                "tags": json.dumps(paper.tags),
                "query": query,
                "stored_at": datetime.utcnow().isoformat()
            }]
        )

@traceable(name="query_semantic_memory")
def query_semantic_memory(query: str, n_results: int = 5) -> list[dict]:
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()) if collection.count() > 0 else 1
    )
    if not results["ids"][0]:
        return []
    papers = []
    for i, doc_id in enumerate(results["ids"][0]):
        papers.append({
            "arxiv_id": doc_id,
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i]
        })
    return papers

# Episodic Memory with SQLite
def init_episodic_db():
    conn = sqlite3.connect('./memory/episodic.db')
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            papers_found INTEGER,
            papers_relevant INTEGER,
            timestamp TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            agent TEXT,
            event_type TEXT,
            payload TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

@traceable(name="write_episodic_memory")
def write_episodic_event(session_id: int, agent: str, event_type: str, payload: dict):
    conn = sqlite3.connect("./memory/episodic.db")
    conn.execute(
        "INSERT INTO events (session_id, agent, event_type, payload, timestamp) VALUES (?, ?, ?, ?, ?)",
        (session_id, agent, event_type, json.dumps(payload), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def create_session(query: str, papers_found: int, papers_relevant: int) -> int:
    conn = sqlite3.connect("./memory/episodic.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (query, papers_found, papers_relevant, timestamp) VALUES (?, ?, ?, ?)",
        (query, papers_found, papers_relevant, datetime.utcnow().isoformat())
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

init_episodic_db()