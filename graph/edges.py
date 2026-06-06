from graph.state import ResearchState

def should_continue(state: ResearchState) -> str:
    if state.get("should_continue", False):
        return "retrieve"   # loop back
    return "end"