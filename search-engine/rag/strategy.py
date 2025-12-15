# rag/strategy.py
from rag.text_utils import extract_codes_from_query

def decide_strategy(norm_query: str, prof: dict, has_main: bool, policy, code_boost_direct: bool = True) -> str:
    top1 = float(prof.get("top1", 0.0))
    gap  = float(prof.get("gap", 0.0))
    conf = float(prof.get("conf", 0.0))

    has_code = bool(extract_codes_from_query(norm_query))

    if code_boost_direct and has_code and top1 >= 0.55 and gap >= 0.05 and has_main:
        return "DIRECT_DOC"

    if conf >= policy.direct_conf_min and gap >= policy.direct_gap_min and has_main:
        return "DIRECT_DOC"

    if gap <= policy.frag_gap_max:
        return "RAG_SOFT"

    if conf >= policy.strict_conf_min:
        return "RAG_STRICT"

    return "RAG_SOFT"
