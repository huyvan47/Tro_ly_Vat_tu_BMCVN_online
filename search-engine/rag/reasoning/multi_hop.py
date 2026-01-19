import time
import json
from typing import List, Dict, Tuple, Any
from rag.config import RAGConfig
from rag.logging.multi_hop_logger import write_multi_hop_logs
from rag.retriever import search as retrieve_search


def analyze_need_next_hop(
    *,
    client,
    query: str,
    hits: List[dict],
    hop: int,
    max_hops: int,
) -> Tuple[bool, str]:
    """
    Dùng LLM để quyết định:
      - có cần hop tiếp không
      - nếu có, sinh ra query mới
    """

    if hop >= max_hops:
        return False, ""

    sys = """
Bạn là module điều phối truy vấn cho hệ thống RAG nông nghiệp.

Nhiệm vụ:
- Nhìn vào câu hỏi và danh sách tài liệu đã tìm được
- Quyết định có cần truy vấn thêm bước nữa (multi-hop) hay không
- Nếu cần, tạo ra MỘT truy vấn tiếp theo, ngắn gọn, cụ thể hơn

Quy tắc rất quan trọng:
- Chỉ trả về JSON hợp lệ
- KHÔNG trả lời nội dung câu hỏi
- KHÔNG sinh lại chính câu query cũ
- KHÔNG tạo truy vấn quá dài
- Chỉ tạo query mới khi thực sự thiếu thông tin

Điều kiện nên dừng (need_next_hop = false):
- Đã có đủ tài liệu liên quan
- Số lượng hit đủ lớn (>= 25)
- Các hit đã bao phủ ý chính của câu hỏi
- Truy vấn mới sẽ không mang lại thông tin mới

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
        for h in hits[:10]
    ]

    payload = {
        "original_query": query,
        "hop_index": hop,
        "num_hits": len(hits),
        "sample_hits": sample_hits,
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.1,
            max_completion_tokens=400,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )

        data = json.loads(resp.choices[0].message.content)

        need = bool(data.get("need_next_hop", False))
        nq = str(data.get("next_query", "")).strip()

        return need, nq

    except Exception:
        return False, ""


from rag.logging.multi_hop_logger import write_multi_hop_logs


def multi_hop_controller(
    *,
    client,
    kb,
    base_query: str,
    must_tags: List[str],
    any_tags: List[str],
    timer=None,
) -> List[dict]:

    MAX_HOPS = RAGConfig.max_multi_hops

    # FIX 1: dùng đúng top_k cho multi-hop
    top_k = getattr(RAGConfig, "multi_hop_top_k", RAGConfig.multi_query_top_k)

    current_query = base_query
    visited_queries = set([base_query.strip().lower()])

    all_hits: List[dict] = []
    seen_ids = set()  # FIX 2: dedupe theo id
    hops_data = []

    if timer:
        timer.mark("multi_hop_start")

    for hop in range(1, MAX_HOPS + 1):

        hits = retrieve_search(
            client=client,
            kb=kb,
            norm_query=current_query,
            top_k=top_k,
            must_tags=must_tags,
            any_tags=any_tags,
        )

        if timer:
            timer.mark_sub("multi_hop", f"hop_{hop}_retrieve")

        # Dedupe hits ngay tại hop
        unique_hits = []
        for h in (hits or []):
            hid = h.get("id")
            if not hid:
                continue
            if hid in seen_ids:
                continue
            seen_ids.add(hid)
            unique_hits.append(h)

        # Ghi dữ liệu hop
        hop_record = {
            "hop": hop,
            "query": current_query,
            "num_hits": len(unique_hits),
            "hits": unique_hits,
            "decision": {},
        }

        if unique_hits:
            all_hits.extend(unique_hits)

        if not unique_hits:
            hop_record["decision"] = {"stop_reason": "no_hits"}
            hops_data.append(hop_record)
            break

        # FIX 3: hỏi LLM trước, không stop do threshold quá sớm
        need_next_hop, next_query = analyze_need_next_hop(
            client=client,
            # dùng base_query để LLM quyết định theo câu hỏi gốc (ổn định hơn)
            query=base_query,
            # đưa vào all_hits (đã dedupe) để LLM thấy bức tranh tổng thể
            hits=all_hits,
            hop=hop,
            max_hops=MAX_HOPS,
        )

        if timer:
            timer.mark_sub("multi_hop", f"hop_{hop}_analyze")

        hop_record["decision"] = {
            "need_next_hop": need_next_hop,
            "next_query": next_query,
            "total_unique_hits": len(all_hits),
        }

        # threshold chỉ là “phanh an toàn” sau khi LLM muốn đi tiếp
        if need_next_hop and len(all_hits) >= RAGConfig.multi_hop_stop_threshold:
            hop_record["decision"]["stop_reason"] = "threshold_reached_after_analyze"
            hops_data.append(hop_record)
            break

        hops_data.append(hop_record)

        if not need_next_hop:
            break

        nq = (next_query or "").strip()
        if not nq:
            break

        nq_norm = nq.lower().strip()

        if nq_norm == current_query.strip().lower():
            break

        if nq_norm in visited_queries:
            break

        visited_queries.add(nq_norm)
        current_query = nq

    if timer:
        timer.mark("multi_hop_total")

    if RAGConfig.enable_multi_query_log:
        write_multi_hop_logs(
            original_query=base_query,
            hops_data=hops_data,
            final_hits=all_hits,
        )

    return all_hits
