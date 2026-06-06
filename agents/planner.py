from langsmith import traceable
from schemas import (
    ResearchPlan, ResearchReport, PaperSummary,
    AgentMessage, MessageType
)
from config import get_instructor_client
from prompts import load_prompt
from memory.store import query_semantic_memory, write_episodic_event, create_session

PLANNER_PROMPT = load_prompt("planner_v1")
SYNTHESIZER_PROMPT = load_prompt("synthesizer_v1")


@traceable(name="plan_research")
def plan_research(query: str, previous_queries: list[str] = []) -> ResearchPlan:
    client, model = get_instructor_client()

    plan = client.chat.completions.create(
        model=model,
        response_model=ResearchPlan,
        messages=[{
            "role": "user",
            "content": PLANNER_PROMPT.format(
                query=query,
                previous_queries="\n".join(previous_queries) if previous_queries else "None"
            )
        }]
    )

    return plan

# @traceable(name="plan_research")
# def plan_research(query: str, previous_queries: list[str] = []) -> ResearchPlan:
#     client, model = get_instructor_client()

#     try:
#         plan = client.chat.completions.create(
#             model=model,
#             response_model=ResearchPlan,
#             messages=[{
#                 "role": "user",
#                 "content": PLANNER_PROMPT.format(
#                     query=query,
#                     previous_queries="\n".join(previous_queries) if previous_queries else "None"
#                 )
#             }]
#         )
#         return plan
#     except Exception as e:
#         print(f"[Planner] plan_research failed: {e}")
#         # Fallback — return a simple plan directly
#         return ResearchPlan(
#             original_query=query,
#             sub_queries=[
#                 "large language model inference optimization techniques",
#                 "LLM quantization pruning efficiency",
#                 "speculative decoding transformer inference speed",
#                 "knowledge distillation language model compression"
#             ],
#             reasoning="Fallback plan due to LLM parsing error",
#             expected_topics=["quantization", "pruning", "distillation", "speculative decoding"]
#         )


@traceable(name="synthesize_report")
def synthesize_report(query: str, papers: list[PaperSummary]) -> ResearchReport:
    client, model = get_instructor_client()

    papers_text = ""
    for p in papers:
        papers_text += f"""
- Title: {p.title}
  Contribution: {p.key_contribution}
  Relevance: {p.relevance_score}
  Tags: {', '.join(p.tags)}
"""
        
    report = client.chat.completions.create(
        model=model,
        response_model=ResearchReport,
        messages=[{
            "role": "user",
            "content": SYNTHESIZER_PROMPT.format(
                query=query,
                papers=papers_text
            )
        }]
    )

    report.original_query = query
    report.paper_count = len(papers)
    return report


@traceable(name="planner_agent")
def planner_agent(query: str) -> ResearchReport:
    from agents.retriever import retriever_agent

    client, model = get_instructor_client()
    session_id = create_session(query, 0, 0)

    # Check memory for existing coverage
    cached = query_semantic_memory(query, n_results=5)
    print(f"[Planner] Memory has {len(cached)} related papers already")

    # Generate research plan via Tree of Thought
    plan = plan_research(query)
    print(f"[Planner] Decomposed into {len(plan.sub_queries)} sub-queries:")
    for q in plan.sub_queries:
        print(f" -> {q}")

    write_episodic_event(session_id, "planner", "plan_created", {
        "original_query": query,
        "sub_queries": plan.sub_queries,
        "reasoning": plan.reasoning
    })

    # dispatch sub-queries to Retriever
    all_papers: list[PaperSummary] = []
    seen_ids = set()

    for sub_query in plan.sub_queries:
        print(f"\n[Planner] Dispatching: {sub_query}")
        summaries = retriever_agent(sub_query, max_results=3)

        for paper in summaries:
            if paper.arxiv_id not in seen_ids and paper.is_relevant:
                all_papers.append(paper)
                seen_ids.add(paper.arxiv_id)

    print(f"\n[Planner] Total unique relevant papers: {len(all_papers)}")

    write_episodic_event(session_id, "planner", "retrieval_complete", {
        "total_papers": len(all_papers),
        "sub_queries": plan.sub_queries
    })

    # synthesize report
    if not all_papers:
        print("[Planner] No relevant papers found")
        return None
    
    report = synthesize_report(query, all_papers)

    write_episodic_event(session_id, "planner", "report_synthesized", {
        "paper_count": report.paper_count,
        "gaps": report.gaps,
        "follow_up_queries": report.follow_up_queries
    })

    return report