from pydantic import BaseModel, Field
from typing import Optional

class PaperSummary(BaseModel):
    title: str
    arxiv_id: str
    topic: str
    key_contribution: str = Field(description="Main contribution in 1-2 sentences")
    relevance_score: float = Field(ge=0, le=1, description="How relevant to the query")
    confidence: float = Field(ge=0, le=1, description="Confidence in this summary")
    tags: list[str] = Field(default_factory=list)

    @property
    def is_relevant(self) -> bool:
        return self.relevance_score >= 0.6 and self.confidence >= 0.7