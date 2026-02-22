from dataclasses import dataclass
from typing import Any, Dict

GraphData = Dict[str, Any]


@dataclass
class RetrievalHit:
    url: str
    title: str
    snippet: str
    source_query: str
    retrieval_status: str = "verified"
    score: float = 0.0
