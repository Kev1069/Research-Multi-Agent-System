from langsmith import traceable
from schemas import PaperSummary, CriticVerdict, AgentMessage, MessageType
from config import get_instructor_client
from memory.store import query_semantic_memory, write_episodic_event

CRITIC_PROMPT = """"You are a critical peer reviewer for ML/AI research papers.
You receive a paper summary produced by a Retriever AGent and must validate it.

Your job is NOT to critique the paper itself, but to validate whether:
1. The relevance_score is justified given the query (allow +/- 0.2 variance)
2. The key_contribution is accurate and not actively misleading
3. The paper is genuinely off-topic for the query

Query the paper was retrieved for: {query}

Paper Summary to review:
Title: {title}
ArXiv ID: {arxiv_id}
Key Contribution: {key_contribution}
Relevance Score: {relevance_score}
Tags: {tags}

Similar papers already in memory:
{memory_context}

Approval guidelines:
- Approve if the paper is reasonably relevant to the query, even partially
- Only reject if relevance_score is inflated by more than 0.3, or contribution is actively misleading
- Brief summaries are acceptable — do not reject for being concise
- Only flag duplication if arxiv_id already exists in memory

Be fair. Err on the side of approval unless there is a clear problem."""


@traceable(name="critic_agent")
def critic_agent(message: AgentMessage) -> AgentMessage:
    assert message.message_type == MessageType.PAPER_SUMMARY
    assert message.receiver == "critic"

    client, model = get_instructor_client()
    payload = message.payload
    query = payload["query"]

    # Check memory for similar papers
    similar = query_semantic_memory(query, n_results=3)
    memory_context = ""
    if similar:
        for s in similar:
            if s["arxiv_id"] != payload["arxiv_id"]:
                memory_context += f"- {s['metadata']['title']}: {s['document']}\n"
    if not memory_context:
        memory_context = "No similar paperes in memory yet."

    verdict = client.chat.completions.create(
        model=model,
        response_model=CriticVerdict,
        messages=[{
            "role": "user",
            "content": CRITIC_PROMPT.format(
                query=query,
                title=payload["title"],
                arxiv_id=payload["arxiv_id"],
                key_contribution=payload["key_contribution"],
                relevance_score=payload["relevance_score"],
                # confidence=payload["confidence"],
                tags=payload["tags"],
                memory_context=memory_context
            )
        }]
    )

    write_episodic_event(
        session_id=message.session_id,
        agent="critic",
        event_type="verdict_issued",
        payload={
            "arxiv_id": verdict.arxiv_id,
            "approved": verdict.approved,
            "revised_relevance_score": verdict.revised_relevance_score,
            "requires_recheck": verdict.requires_recheck,
            "issues": verdict.issues
        }
    )

    return AgentMessage(
        sender="critic",
        receiver=message.sender,
        message_type=MessageType.CRITIC_VERDICT,
        payload=verdict.model_dump(),
        session_id=message.session_id
    )