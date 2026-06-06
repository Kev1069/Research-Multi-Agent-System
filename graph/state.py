from typing import TypedDict, Annotated
import operator
from schemas import PaperSummary, ResearchReport, ResearchPlan

class ResearchState(TypedDict):
    # input
    original_query: str

    # planning
    current_plan: ResearchPlan | None
    previous_queries: Annotated[list[str], operator.add] # accumulates across loops

    # papers
    all_papers: Annotated[list[PaperSummary], operator.add] # accumulates across loops
    seen_arxiv_ids: list[str]

    # control
    iteration: int
    max_iterations: int
    should_continue: bool

    # output
    report: ResearchReport | None