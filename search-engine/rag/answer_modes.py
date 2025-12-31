from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

# -----------------------------
# Types
# -----------------------------

@dataclass(frozen=True)
class AnswerPolicy:
    """
    intent: Loại tri thức cần trả lời (procedure/disease/product/registry/general)
    format: Hình thức trả lời (steps/bullets/short/verbatim/table/listing)
    require_grounding: Bắt buộc bám theo evidence (giảm hallucination)
    max_sources: Số nguồn (chunk/doc) tối đa đưa vào prompt
    """
    intent: str
    format: str
    require_grounding: bool
    max_sources: int


# -----------------------------
# Normalization helpers
# -----------------------------

def norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def join_text(*parts: str) -> str:
    return " ".join(p for p in parts if p).strip().lower()


# -----------------------------
# Keyword / pattern detectors
# -----------------------------

FORMULATION_RE = re.compile(r"\b(ec|sc|wp|sl|wg|wdg|gr|df)\b", re.IGNORECASE)
DOSAGE_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(ml|l|lit|lít|g|gr|kg|ppm|%|cc)\b", re.IGNORECASE)
STEP_RE = re.compile(r"\b(bước\s*\d+|step\s*\d+|\d+\s*[\)\.]\s*)\b", re.IGNORECASE)

KW_REGISTRY = [
    "thương phẩm", "tên thương phẩm", "nhãn hiệu", "đăng ký", "registry",
    "tổ chức", "tổ chức đăng ký", "chủ sở hữu", "số đăng ký"
]
KW_DISEASE = [
    "bệnh", "triệu chứng", "dấu hiệu", "xì mủ", "thối rễ", "cháy lá",
    "thán thư", "ghẻ", "nứt thân", "đốm lá", "thối trái", "tảo đỏ", "rong rêu"
]
KW_PRODUCT = [
    "thuốc", "đặc trị", "tác dụng", "hoạt chất", "thành phần", "công dụng",
    "chữa", "trị", "trừ", "phòng trừ", "pha", "phun", "tưới"
]
KW_PROCEDURE = [
    "quy trình", "quy trinh", "các bước", "hướng dẫn", "cách làm",
    "làm thế nào", "phương pháp", "quy cách", "thực hiện"
]
KW_LISTING = [
    "liệt kê", "danh sách", "tổng hợp", "bao gồm những", "gồm những",
    "các loại", "những loại", "những sản phẩm", "những thuốc"
]


def has_any_kw(text: str, kws: List[str]) -> bool:
    return any(kw in text for kw in kws)

def detect_listing(user_query: str) -> bool:
    q = norm(user_query)
    return has_any_kw(q, KW_LISTING)


# -----------------------------
# Core decision
# -----------------------------

ENTITY_TO_POLICY: Dict[str, AnswerPolicy] = {
    # Bạn có thể mở rộng các entity_type theo schema của bạn
    "procedure": AnswerPolicy(intent="procedure", format="steps", require_grounding=True, max_sources=12),
    "process":   AnswerPolicy(intent="procedure", format="steps", require_grounding=True, max_sources=12),
    "quy_trinh": AnswerPolicy(intent="procedure", format="steps", require_grounding=True, max_sources=12),

    "disease":   AnswerPolicy(intent="disease",   format="bullets", require_grounding=True, max_sources=10),
    "benh":      AnswerPolicy(intent="disease",   format="bullets", require_grounding=True, max_sources=10),

    "product":   AnswerPolicy(intent="product",   format="table", require_grounding=True, max_sources=10),
    "chemical":  AnswerPolicy(intent="product",   format="table", require_grounding=True, max_sources=10),
    "san_pham":  AnswerPolicy(intent="product",   format="table", require_grounding=True, max_sources=10),
    "thuoc":     AnswerPolicy(intent="product",   format="table", require_grounding=True, max_sources=10),

    "registry":  AnswerPolicy(intent="registry",  format="verbatim", require_grounding=True, max_sources=8),
    "dang_ky":   AnswerPolicy(intent="registry",  format="verbatim", require_grounding=True, max_sources=8),

    "faq":       AnswerPolicy(intent="general",   format="short", require_grounding=True, max_sources=8),
    "general":   AnswerPolicy(intent="general",   format="short", require_grounding=True, max_sources=8),
}

DEFAULT_POLICY = AnswerPolicy(intent="general", format="short", require_grounding=True, max_sources=8)


def decide_answer_policy(
    user_query: str,
    primary_doc: Dict,
    *,
    parsed_intent: Optional[str] = None,   # nếu bạn đã phân tích intent từ query
) -> AnswerPolicy:
    """
    Quyết định intent + format dựa trên:
    1) listing (nếu câu hỏi dạng liệt kê)
    2) parsed_intent (nếu bạn có module phân tích query)
    3) entity_type của doc (THAY category)
    4) heuristic keywords (fallback)
    """
    q = norm(user_query)
    doc_q = norm(primary_doc.get("question"))
    doc_a = norm(primary_doc.get("answer") or primary_doc.get("content") or "")
    ent = norm(primary_doc.get("entity_type"))

    text = join_text(q, doc_q, doc_a)

    # 1) Parsed intent override (nếu có)
    if parsed_intent:
        pi = norm(parsed_intent)
        if pi in ENTITY_TO_POLICY:
            return ENTITY_TO_POLICY[pi]
        # cho phép pi là intent chuẩn
        if pi in {"procedure", "disease", "product", "registry", "general"}:
            # gán policy mặc định theo intent
            for k, v in ENTITY_TO_POLICY.items():
                if v.intent == pi:
                    return v

    # 2) Entity_type is the new "category"
    if ent and ent in ENTITY_TO_POLICY:
        return ENTITY_TO_POLICY[ent]

    # 3) Heuristic fallback
    # Registry ưu tiên bắt sớm (rủi ro cao, cần bám evidence)
    if has_any_kw(text, KW_REGISTRY):
        return ENTITY_TO_POLICY["registry"]

    # Procedure: nếu có dấu bước hoặc từ khóa quy trình
    if STEP_RE.search(text) or has_any_kw(text, KW_PROCEDURE):
        return ENTITY_TO_POLICY["procedure"]

    # Disease
    if has_any_kw(text, KW_DISEASE):
        return ENTITY_TO_POLICY["disease"]

    # Product: có dạng chế phẩm hoặc liều lượng/đơn vị hoặc từ khóa thuốc
    if FORMULATION_RE.search(text) or DOSAGE_RE.search(text) or has_any_kw(text, KW_PRODUCT):
        return ENTITY_TO_POLICY["product"]

    return DEFAULT_POLICY


# -----------------------------
# Convenience: turn policy into prompt directives (optional)
# -----------------------------

def policy_to_prompt_directives(policy: AnswerPolicy) -> str:
    """
    Dòng hướng dẫn có thể chèn vào system prompt/user prompt.
    """
    rules = []
    rules.append(f"- Answer intent: {policy.intent}")
    rules.append(f"- Output format: {policy.format}")
    if policy.require_grounding:
        rules.append("- Only use information from provided documents. Do not invent details.")
        rules.append("- If documents are insufficient, say you don't have enough information and ask for missing details.")
    rules.append(f"- Use at most {policy.max_sources} sources/chunks.")
    return "\n".join(rules)
