# rag/multi_query.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


# -----------------------------
# 1) Build query variants
# -----------------------------

def build_query_variants(
    norm_query: str,
    must_tags: List[str],
    any_tags: List[str],
    *,
    max_variants: int = 4,
) -> List[Tuple[str, str]]:
    """
    Luôn tạo 3-4 variants để test:
    - main: câu chuẩn hoá
    - short: backbone (rút gọn)
    - intent: thêm prefix theo intent
    - formula_intent: nếu có tag formula (tuỳ schema bạn)
    """
    q = (norm_query or "").strip()
    if not q:
        return [("main", "")]

    variants: List[Tuple[str, str]] = [("main", q)]

    q_lower = q.lower()
    tags_all = (must_tags or []) + (any_tags or [])

    # (A) short backbone: lấy 6-8 token đầu, bỏ ký tự dư
    short = " ".join(q.split()[:8]).strip()
    if short and short != q:
        variants.append(("short", short))

    # (B) intent: dựa trên heuristic ngôn ngữ
    ask_recommend = bool(re.search(r"\b(thuoc|phun|tri|phong|xu\s*ly|dung\s*gi|nen\s*dung|loai\s*nao)\b", q_lower))
    if ask_recommend:
        variants.append(("treatment_intent", f"thuoc tri {q}"))

    # (C) formula intent: nếu có tag formula (tuỳ bạn có dùng tag này không)
    has_formula = any(t.startswith("formula:") for t in tags_all)
    if has_formula:
        variants.append(("formula_intent", f"cong thuc {q}"))

    # Dedup + cap
    seen = set()
    out: List[Tuple[str, str]] = []
    for purpose, qq in variants:
        qq = (qq or "").strip()
        if not qq:
            continue
        key = qq.lower()
        if key in seen:
            continue
        out.append((purpose, qq))
        seen.add(key)
        if len(out) >= max_variants:
            break

    return out if out else [("main", q)]


# -----------------------------
# 2) Multi-query retrieval + fusion
# -----------------------------

DEFAULT_WEIGHTS = {
    "main": 1.00,
    "treatment_intent": 0.95,
    "formula_intent": 0.95,
    "short": 0.90,
}

def retrieve_multi_query(
    *,
    retrieve_fn,          # function like rag.retriever.search
    client: Any,
    kb: Any,
    variants: List[Tuple[str, str]],
    must_tags: List[str],
    any_tags: List[str],
    top_k_each: int = 150,
    pool_cap: int = 700,
    weights: Dict[str, float] | None = None,
) -> List[Dict[str, Any]]:
    """
    Chạy retrieve cho từng variant, union theo doc_id, và tạo mq_score theo weighted-max.

    - top_k_each: top_k cho mỗi query (nhỏ hơn strict top_k để giảm noise)
    - pool_cap: giới hạn số doc sau merge (trước khi pipeline fused_score/rerank tiếp)
    """
    weights = weights or DEFAULT_WEIGHTS

    pool: Dict[str, Dict[str, Any]] = {}  # doc_id -> doc (giữ doc tốt nhất)
    score_pool: Dict[str, float] = {}     # doc_id -> mq_score (weighted max)

    for purpose, q in variants:
        docs = retrieve_fn(
            client=client,
            kb=kb,
            norm_query=q,
            top_k=top_k_each,
            must_tags=must_tags,
            any_tags=any_tags,
        ) or []

        w = float(weights.get(purpose, 0.85))

        for d in docs:
            doc_id = str(d.get("id") or "")
            if not doc_id:
                continue

            base_score = float(d.get("score", 0.0))
            mq = w * base_score

            # update mq_score = max
            if mq > score_pool.get(doc_id, 0.0):
                score_pool[doc_id] = mq

            # keep a representative doc object (ưu tiên doc có score cao hơn)
            if doc_id not in pool:
                pool[doc_id] = d
            else:
                # nếu doc mới có score cao hơn, replace doc (giữ thông tin tốt hơn)
                if float(d.get("score", 0.0)) > float(pool[doc_id].get("score", 0.0)):
                    pool[doc_id] = d

    merged: List[Dict[str, Any]] = []
    for doc_id, d in pool.items():
        d2 = dict(d)
        d2["mq_score"] = float(score_pool.get(doc_id, 0.0))
        d2["score"] = d2["mq_score"]
        d2["mq_purpose"] = "multi"  # debug
        merged.append(d2)

    # sort by mq_score desc, then by original score desc
    merged.sort(key=lambda x: (float(x.get("mq_score", 0.0)), float(x.get("score", 0.0))), reverse=True)

    # cap pool
    if pool_cap and len(merged) > pool_cap:
        merged = merged[:pool_cap]

    return merged
