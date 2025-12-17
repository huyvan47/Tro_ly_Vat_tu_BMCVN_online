from dataclasses import dataclass

@dataclass(frozen=True)
class RAGConfig:
    min_score_main: float = 0.35
    min_score_suggest: float = 0.40
    max_suggest: int = 3

    use_llm_rerank: bool = True
    top_k_rerank: int = 60
    rerank_snippet_chars: int = 500
    debug_rerank: bool = False

    topk_router = 10
    max_source_chars_per_call = 12000
    
    max_ctx_strict: int = 16
    max_ctx_soft: int = 12

    code_boost_direct: bool = True
