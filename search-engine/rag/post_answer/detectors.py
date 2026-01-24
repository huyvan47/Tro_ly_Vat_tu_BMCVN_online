import re
from typing import Optional

_RISK_INTENT_PATTERNS = [
    r"ảnh hưởng", r"có hại", r"an toàn", r"ngộ độc", r"độc", r"cháy", r"vàng",
    r"rụng", r"chết", r"ức chế", r"còi", r"stunt", r"stress",
    r"cây giống", r"cây con", r"cây non", r"mầm", r"nảy mầm", r"gieo", r"ươm",
]

_MISSING_EVIDENCE_PATTERNS = [
    r"không có thông tin",
    r"không có bất kỳ đoạn",
    r"không có hướng dẫn",
    r"Không thấy trong tài liệu",
    r"tài liệu không đề cập",
    r"không đề cập trực tiếp",
    r"chưa có thông tin",
    r"không có dữ liệu",
    r"KHÔNG có sản phẩm nào",
    r"Không có sản phẩm nào",
]

def _match_any(patterns, text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in patterns)

def should_enrich_post_answer(
    *,
    user_query: str,
    answer_text: str,
    answer_mode: str,
    route: str = "RAG",
) -> bool:
    """
    Chỉ enrich ở bước cuối khi:
    - answer_mode == product
    - query mang tính risk/safety/ảnh hưởng
    - answer RAG xác nhận thiếu bằng chứng trong tài liệu
    """
    if answer_mode not in ("product", "formula"):
        return False

    # Route không bắt buộc, nhưng giữ để debug/audit
    if (route or "").upper() != "RAG":
        return False

    # has_risk_intent = _match_any(_RISK_INTENT_PATTERNS, user_query)
    has_risk_intent = True
    missing_doc_evidence = _match_any(_MISSING_EVIDENCE_PATTERNS, answer_text)

    print('has_risk_intent:', has_risk_intent)
    print('missing_doc_evidence:', missing_doc_evidence)

    return bool(has_risk_intent and missing_doc_evidence)
