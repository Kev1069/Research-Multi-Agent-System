from langgraph.graph import StateGraph, END
from graph.state import ResearchState
from graph.nodes import node_plan, node_retrieve, node_synthesize, node_evaluate_gaps
from graph.edges import should_continue


def build_research_graph():
    graph = StateGraph(ResearchState)

    # adding nodes
    graph.add_node("plan", node_plan)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("synthesize", node_synthesize)
    graph.add_node("evaluate_gaps", node_evaluate_gaps)

    # adding edges
    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "evaluate_gaps")

    # conditional edge: loop or end
    graph.add_conditional_edges(
        "evaluate_gaps",
        should_continue,
        {
            "retrieve": "retrieve", # Loop back to retrieve with new queires
            "end": END
        }
    )

    return graph.compile()