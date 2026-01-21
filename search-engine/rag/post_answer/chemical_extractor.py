# rag/post_answer/chemical_extractor.py

import re
from typing import List, Set, Optional

# ------------------------------------------------------------
# 1. Extract chemical từ TAG MATCH (NGUỒN CHUẨN NHẤT – KB)
# ------------------------------------------------------------

def extract_chemicals_from_matched_tags(
    any_tags: Optional[List[str]],
    must_tags: Optional[List[str]] = None,
    limit: int = 2,
) -> List[str]:
    """
    Trích hoạt chất từ tag đã được tags_filter match.
    Ví dụ:
      tag = "chemical:acetochlor"
      tag = "chemical:s-metolachlor"
    """
    chems: List[str] = []

    for t in (any_tags or []) + (must_tags or []):
        if not isinstance(t, str):
            continue
        if t.lower().startswith("chemical:"):
            c = t.split(":", 1)[1].strip().lower()
            if c and c not in chems:
                chems.append(c)
        if len(chems) >= limit:
            break

    return chems


# ------------------------------------------------------------
# 2. Fallback: extract từ câu trả lời đã generate (RẤT HẠN CHẾ)
# ------------------------------------------------------------

_TEXT_CHEM_RE = re.compile(
    r"(?:hoạt\s*chất|active\s*ingredient)\s*:\s*([A-Za-z0-9\-\s]+)",
    re.IGNORECASE
)

def extract_primary_chemical_from_answer(answer_text: str) -> Optional[str]:
    """
    Fallback cuối cùng khi KHÔNG có chemical tag.
    Ví dụ:
      'Hoạt chất: Acetochlor 500g/lít'
    """
    if not answer_text:
        return None

    m = _TEXT_CHEM_RE.search(answer_text)
    if not m:
        return None

    c = m.group(1).strip()
    c = re.split(r"[;,(\n\r]", c)[0].strip()
    c = re.sub(r"\s+", " ", c).lower()

    if not c or len(c) > 50:
        return None

    return c


# ------------------------------------------------------------
# 3. LEGACY – KHÔNG KHUYẾN NGHỊ DÙNG (giữ lại nếu cần backward)
# ------------------------------------------------------------

_TAG_CHEM_RE = re.compile(
    r"(?:^|\|)\s*chemical:([a-z0-9\-]+)\s*(?=\||$)",
    re.IGNORECASE
)

def extract_chemicals_from_hits(hits: List[dict], limit: int = 3) -> List[str]:
    """
    LEGACY:
    - KHÔNG dùng cho enrichment nữa
    - Chỉ giữ nếu có module cũ còn phụ thuộc
    """
    chems: Set[str] = set()
    for h in hits or []:
        tv2 = str(h.get("tags_v2") or "")
        for m in _TAG_CHEM_RE.findall(tv2):
            c = (m or "").strip().lower()
            if c:
                chems.add(c)
        if len(chems) >= limit:
            break
    return list(chems)[:limit]
