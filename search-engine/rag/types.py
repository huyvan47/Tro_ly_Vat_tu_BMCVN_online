from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class Hit:
    question: str
    alt_question: str
    answer: str
    score: float                      # embedding cosine
    category: Optional[str] = None
    tags: Optional[str] = None

    rerank_score: float = 0.0
    include_in_context: bool = False
    fused_score: float = 0.0

Profile = Dict[str, float]  # top1/top2/gap/mean5/n/conf

@dataclass
class PipelineResult:
    text: str
    img_keys: List[str]
    route: str
    norm_query: str
    strategy: str
    profile: Dict[str, Any]
