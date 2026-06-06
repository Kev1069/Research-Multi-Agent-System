from langsmith import traceable
from graph.state import ResearchState
from agents.planner import plan_research, synthesize_report
from agents.retriever import retriever_agent
from schemas import ResearchPlan, SubQuery

# rate limiting
MAX_ITERATIONS = 2


@traceable(name="node_plan")
def node_plan(state: ResearchState) -> dict:
    print(f"\n[Graph] Planning iteration {state['iteration'] + 1}")

    plan = plan_research(
        query=state["original_query"],
        previous_queries=state["previous_queries"]
    )

    return {"current_plan": plan}


@traceable(name="node_retrieve")
def node_retrieve(state: ResearchState) -> dict:
    plan = state["current_plan"]
    seen_ids = set(state.get("seen_arxiv_ids", []))
    new_papers = []
    new_queries = []

    for sub_query in plan.get_query_strings():
        if sub_query in state["previous_queries"]:
            print(f"[Graph] Skipping already searched: {sub_query}")
            continue
    
        print(f"[Graph] Retrieving: {sub_query}")
        summaries = retriever_agent(sub_query, max_results=3)
        new_queries.append(sub_query)

        for paper in summaries:
            if paper.arxiv_id not in seen_ids and paper.is_relevant:
                new_papers.append(paper)
                seen_ids.add(paper.arxiv_id)

    return {
        "all_papers": new_papers,
        "seen_arxiv_ids": list(seen_ids),
        "previous_queries": new_queries
    }


@ traceable(name="node_synthesize")
def node_synthesize(state: ResearchState) -> dict:
    print(f"\n[Graph] Synthesizing {len(state['all_papers'])} papers")

    report = synthesize_report(state["original_query"], state["all_papers"])
    return {"report": report}


@traceable(name="node_evaluate_gaps")
def node_evaluate_gaps(state: ResearchState) -> dict:
    report = state["report"]
    iteration = state["iteration"] + 1

    # decide whether to loop back
    has_gaps = len(report.gaps) > 0
    has_follow_ups = len(report.follow_up_queries) > 0
    under_limit = iteration < state["max_iterations"]
    enough_papers = len(state["all_papers"]) < 8 # loop if we have few papers

    should_continue = has_gaps and has_follow_ups and under_limit and enough_papers

    if should_continue:
        print(f"[Graph] Gaps detected, looping back (iteration {iteration}/{state['max_iterations']})")
        # use follow-up queries as next plan
        next_plan = state["current_plan"].model_copy(update={
            "sub_queries": [SubQuery(query=q) for q in report.follow_up_queries[:3]], # take top 3
            "reasoning": "Follow-up queries based on detected gaps"
        })
        return {
            "iteration": iteration,
            "should_continue": True,
            "current_plan": next_plan
        }
    else:
        print(f"[Graph] Stopping after iteration {iteration}")
        return {
            "iteration": iteration,
            "should_continue": False
        }