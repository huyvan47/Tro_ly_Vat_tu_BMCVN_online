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

    """
    Multi-hop theo thiết kế mới:

    - Hop 1: chỉ dùng must_tags (vai trò chính)
    - Hop 2: chỉ dùng any_tags (vai trò phụ)
    - Nếu hop 1 không có kết quả -> dừng luôn
    - Không sử dụng analyze_need_next_hop cho formula query
    - Giữ nguyên logging và timer
    """

    top_k = getattr(RAGConfig, "multi_hop_top_k", RAGConfig.multi_query_top_k)

    all_hits: List[dict] = []
    seen_ids = set()
    hops_data = []

    if timer:
        timer.mark("multi_hop_start")

    # ====== HOP 1: tìm theo MUST ======
    hits1 = retrieve_search(
        client=client,
        kb=kb,
        norm_query=base_query,
        top_k=top_k,
        must_tags=must_tags,
        any_tags=[],
    )

    if timer:
        timer.mark_sub("multi_hop", "hop_1_retrieve")

    unique_hits1 = []
    for h in (hits1 or []):
        hid = h.get("id")
        if not hid or hid in seen_ids:
            continue
        seen_ids.add(hid)
        unique_hits1.append(h)

    hop1_record = {
        "hop": 1,
        "query": base_query,
        "num_hits": len(unique_hits1),
        "hits": unique_hits1,
        "decision": {},
    }

    hops_data.append(hop1_record)

    if unique_hits1:
        all_hits.extend(unique_hits1)

    # Nếu hop 1 không có kết quả -> dừng toàn bộ
    if not unique_hits1:
        hop1_record["decision"] = {"stop_reason": "no_hits_in_hop1"}
        if RAGConfig.enable_multi_query_log:
            write_multi_hop_logs(
                original_query=base_query,
                hops_data=hops_data,
                final_hits=all_hits,
            )
        return all_hits

    # ====== HOP 2: tìm theo SOFT (any_tags) ======
    if any_tags:
        next_query = f"sản phẩm có cơ chế {any_tags[0].replace('mechanisms:', '')}"

        hits2 = retrieve_search(
            client=client,
            kb=kb,
            norm_query=next_query,
            top_k=top_k,
            must_tags=[],
            any_tags=any_tags,
        )

        if timer:
            timer.mark_sub("multi_hop", "hop_2_retrieve")

        unique_hits2 = []
        for h in (hits2 or []):
            hid = h.get("id")
            if not hid or hid in seen_ids:
                continue
            seen_ids.add(hid)
            unique_hits2.append(h)

        hop2_record = {
            "hop": 2,
            "query": next_query,
            "num_hits": len(unique_hits2),
            "hits": unique_hits2,
            "decision": {"stop_reason": "completed_soft_search"},
        }

        hops_data.append(hop2_record)

        if unique_hits2:
            all_hits.extend(unique_hits2)

    if timer:
        timer.mark("multi_hop_total")

    if RAGConfig.enable_multi_query_log:
        write_multi_hop_logs(
            original_query=base_query,
            hops_data=hops_data,
            final_hits=all_hits,
        )

    return all_hits

