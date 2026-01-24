import json
from typing import List, Dict, Tuple, Any

from rag.config import RAGConfig
from rag.logging.multi_hop_logger import write_multi_hop_logs
from rag.retriever import search as retrieve_search


# -----------------------------
# 1) Intent strategy (optional)
# -----------------------------
def analyze_intent_strategy(client, query: str) -> Dict[str, Any]:
    """
    Nhẹ, không bắt buộc. Dùng để tạo 'hint' cho fallback logic.
    Không dùng để hard-code tags (vì bạn đã có tagger mạnh).
    """
    sys = """
Bạn là chuyên gia nông nghiệp.

Nhiệm vụ:
- Hiểu mục tiêu thực sự của người dùng
- Xác định:
  - primary_target (mục tiêu chính)
  - fallback_targets (mục tiêu thay thế hợp lý)

Chỉ trả về JSON hợp lệ theo schema:
{
  "primary_target": string,
  "fallback_targets": [string, ...]
}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": query},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        if not isinstance(data, dict):
            return {"primary_target": "", "fallback_targets": []}
        data.setdefault("primary_target", "")
        data.setdefault("fallback_targets", [])
        if not isinstance(data.get("fallback_targets"), list):
            data["fallback_targets"] = []
        return data
    except Exception:
        return {"primary_target": "", "fallback_targets": []}


# -----------------------------------------
# 2) Decide next hop + next query (LLM-loop)
# -----------------------------------------
def analyze_need_next_hop(
    *,
    client,
    query: str,
    hits: List[dict],
    hop: int,
    max_hops: int,
) -> Tuple[bool, str, str]:
    """
    LLM chỉ được phép:
      - quyết định có cần hop tiếp không
      - nếu cần, sinh next_query (NGẮN, KHÔNG TRÙNG query cũ)
    Hệ thống vẫn có stop cứng riêng.
    """
    if hop >= max_hops:
        return False, "", "reach_max_hops"

    sys = """
Bạn là module điều phối truy vấn cho hệ thống RAG nông nghiệp.

Nhiệm vụ:
- Nhìn vào câu hỏi và danh sách tài liệu đã tìm được
- Quyết định có cần truy vấn thêm bước nữa (multi-hop) hay không
- Nếu cần, tạo ra MỘT truy vấn tiếp theo, ngắn gọn, cụ thể hơn

Quy tắc:
- Chỉ trả về JSON hợp lệ
- KHÔNG trả lời nội dung câu hỏi
- KHÔNG sinh lại chính câu query cũ
- KHÔNG tạo truy vấn quá dài
- Chỉ tạo query mới khi thực sự thiếu thông tin

Output JSON schema:
{
  "need_next_hop": boolean,
  "next_query": string,
  "reason": string
}
"""

    sample_hits = [
        {
            "id": h.get("id"),
            "question": h.get("question"),
            "tags": h.get("tags_v2"),
        }
        for h in (hits or [])[:10]
    ]

    payload = {
        "original_query": query,
        "hop_index": hop,
        "num_hits": len(hits or []),
        "sample_hits": sample_hits,
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.1,
            max_completion_tokens=300,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        data = json.loads(resp.choices[0].message.content)

        need = bool(data.get("need_next_hop", False))
        nq = str(data.get("next_query", "")).strip()
        reason = str(data.get("reason", "")).strip()

        # Guard: không cho next_query rỗng hoặc trùng
        if not need or not nq:
            return False, "", reason or "llm_stop_or_empty_next_query"

        return True, nq, reason or "need_next_hop"

    except Exception:
        return False, "", "llm_exception"


# -----------------------------------------
# 3) Utility: dedupe hits + bookkeeping
# -----------------------------------------
def _dedupe_hits(hits: List[dict], seen_ids: set) -> List[dict]:
    out = []
    for h in (hits or []):
        hid = h.get("id")
        if not hid or hid in seen_ids:
            continue
        seen_ids.add(hid)
        out.append(h)
    return out


def _get_cfg(name: str, default):
    return getattr(RAGConfig, name, default)


# -----------------------------------------
# 4) Multi-hop controller (production-grade)
# -----------------------------------------
def multi_hop_controller(
    *,
    client,
    kb,
    base_query: str,
    must_tags: List[str],
    any_tags: List[str],
    timer=None,
) -> List[dict]:
    """
    Multi-hop đúng chuẩn (LLM-in-the-loop) trên hạ tầng hiện có:

    - Hop 1: strict retrieval theo must_tags (query = base_query)
    - Hop 2..N: LLM quyết định cần hop tiếp và sinh next_query
        + retrieval relax (ưu tiên any_tags nếu có, nếu không thì free search)
    - Stop cứng:
        1) max_hops
        2) không thêm doc mới (added == 0)
        3) đạt ngưỡng coverage (min_docs_for_answer)
        4) LLM nói dừng (need_next_hop = false) / next_query invalid
        5) lặp query (anti-loop)

    - Giữ logging (hops_data) và timer
    """

    # ---- Config ----
    top_k = int(_get_cfg("multi_hop_top_k", 20))
    max_hops = int(_get_cfg("max_multi_hops", 3))  # nên 2~3
    min_docs = int(_get_cfg("min_docs_for_answer", 25))  # coverage threshold
    enable_logs = bool(_get_cfg("enable_multi_query_log", True))

    must_tags = list(must_tags or [])
    any_tags = list(any_tags or [])

    all_hits: List[dict] = []
    seen_ids = set()
    hops_data: List[dict] = []
    used_queries = set()

    if timer:
        timer.mark("multi_hop_start")

    # (Optional) intent hint — chỉ để log/diagnostic, không hard-code tags
    intent_hint = analyze_intent_strategy(client, base_query)

    # =========================
    # HOP 1: STRICT by must_tags
    # =========================
    used_queries.add(base_query)

    hits1 = retrieve_search(
        client=client,
        kb=kb,
        norm_query=base_query,
        top_k=top_k,
        must_tags=must_tags,
        any_tags=[],  # strict
    )
    if timer:
        timer.mark_sub("multi_hop", "hop_1_retrieve")

    unique_hits1 = _dedupe_hits(hits1, seen_ids)
    all_hits.extend(unique_hits1)

    hop1_record = {
        "hop": 1,
        "query": base_query,
        "num_hits": len(unique_hits1),
        "hits": unique_hits1,
        "decision": {
            "intent_hint": intent_hint,
            "stop_reason": "",
        },
    }
    hops_data.append(hop1_record)

    # stop cứng: hop1 không có gì
    if not unique_hits1:
        hop1_record["decision"]["stop_reason"] = "no_hits_in_hop1"
        if enable_logs:
            write_multi_hop_logs(
                original_query=base_query,
                hops_data=hops_data,
                final_hits=all_hits,
            )
        return all_hits

    # stop cứng: đủ coverage ngay
    if len(all_hits) >= min_docs:
        hop1_record["decision"]["stop_reason"] = f"enough_docs_after_hop1>={min_docs}"
        if timer:
            timer.mark("multi_hop_total")
        if enable_logs:
            write_multi_hop_logs(
                original_query=base_query,
                hops_data=hops_data,
                final_hits=all_hits,
            )
        return all_hits

    # ==========================================
    # HOP 2..N: LLM decide next_query + relax search
    # ==========================================
    current_query = base_query
    # hop index bắt đầu từ 2
    for hop in range(2, max_hops + 1):
        # 1) LLM decide
        need, next_query, llm_reason = analyze_need_next_hop(
            client=client,
            query=current_query,
            hits=all_hits,
            hop=hop,
            max_hops=max_hops,
        )

        decision = {
            "need_next_hop": need,
            "llm_reason": llm_reason,
            "stop_reason": "",
        }

        if not need or not next_query:
            decision["stop_reason"] = "llm_stop"
            hops_data.append({
                "hop": hop,
                "query": current_query,
                "num_hits": 0,
                "hits": [],
                "decision": decision,
            })
            break

        # anti-loop: không cho query lặp hoặc quá giống (đơn giản: exact match)
        if next_query in used_queries:
            decision["stop_reason"] = "repeat_query_blocked"
            hops_data.append({
                "hop": hop,
                "query": next_query,
                "num_hits": 0,
                "hits": [],
                "decision": decision,
            })
            break

        used_queries.add(next_query)

        # 2) SEARCH strategy:
        #    - ưu tiên any_tags nếu có (relax theo tag)
        #    - nếu không có any_tags → free search
        must_local: List[str] = []
        any_local: List[str] = any_tags if any_tags else []

        hits_h = retrieve_search(
            client=client,
            kb=kb,
            norm_query=next_query,
            top_k=top_k,
            must_tags=must_local,
            any_tags=any_local,
        )
        if timer:
            timer.mark_sub("multi_hop", f"hop_{hop}_retrieve")

        unique_hits_h = _dedupe_hits(hits_h, seen_ids)
        added = len(unique_hits_h)
        all_hits.extend(unique_hits_h)

        decision["added_new_docs"] = added
        decision["search_mode"] = "any_tags" if any_local else "free_search"

        hops_data.append({
            "hop": hop,
            "query": next_query,
            "num_hits": len(unique_hits_h),
            "hits": unique_hits_h,
            "decision": decision,
        })

        # 3) Stop cứng: không thêm doc mới
        if added == 0:
            decision["stop_reason"] = "no_new_docs_added"
            break

        # 4) Stop cứng: đủ coverage
        if len(all_hits) >= min_docs:
            decision["stop_reason"] = f"enough_docs>={min_docs}"
            break

        # update current query for next iteration
        current_query = next_query

    if timer:
        timer.mark("multi_hop_total")

    if enable_logs:
        write_multi_hop_logs(
            original_query=base_query,
            hops_data=hops_data,
            final_hits=all_hits,
        )

    return all_hits
