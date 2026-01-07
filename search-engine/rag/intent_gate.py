import json, re, hashlib
from typing import Dict, Any, List, Tuple

# --------- 1) Heuristic để tránh gọi LLM quá nhiều ----------
_CONCEPTUAL_TRIGGERS = [
    "gần thu hoạch", "cách ly", "phi", "mrl", "an toàn",
    "tận gốc", "diệt tận gốc", "chỉ ức chế",
    "nên chọn", "khác nhau", "ưu nhược", "cơ chế",
    "tiếp xúc", "lưu dẫn", "nội hấp",
    "mùi", "hôi", "tuyến trùng",
]

def _need_intent_gate(norm_query: str) -> bool:
    q = (norm_query or "").lower()
    return any(k in q for k in _CONCEPTUAL_TRIGGERS)

# --------- 2) LLM phân loại intent + slots ----------
def analyze_intent_and_slots(
    *,
    client,
    norm_query: str,
    any_tags: List[str],
    max_tokens: int = 450,
) -> Dict[str, Any]:
    """
    Trả về JSON:
      intent_type: global_pure | global_conditional | rag_catalog
      required_slots: [...]
      missing_slots: [...]
      should_ask_back: bool
      ask_back_questions: [<=2 câu]
      route_override: GLOBAL | RAG | ""
      answer_mode_override: GLOBAL_BOUNDED | ""  (tuỳ bạn)
      confidence: 0..1
      reasons: [..]
    """
    sys = (
        "Bạn là bộ phân tích intent/slot cho hệ thống tư vấn BVTV.\n"
        "Nhiệm vụ: phân loại câu hỏi và xác định thông tin tối thiểu để trả lời CHẮC.\n"
        "QUY TẮC:\n"
        "- KHÔNG trả lời nội dung BVTV.\n"
        "- Chỉ trả JSON hợp lệ.\n"
        "- Ưu tiên an toàn: nếu thiếu thông tin quan trọng (PHI/cây trồng/sản phẩm/hoạt chất/thời điểm) "
        "thì should_ask_back=true.\n"
        "- ask_back_questions tối đa 2 câu, ngắn, dễ trả lời.\n"
        "\n"
        "Gợi ý phân loại:\n"
        "- global_pure: câu hỏi nguyên lý chung (vd: diệt nấm tận gốc?)\n"
        "- global_conditional: nhìn như chung nhưng phụ thuộc biến (vd: gần thu hoạch an toàn không?)\n"
        "- rag_catalog: hỏi danh mục/sản phẩm/phối trộn cụ thể.\n"
    )

    payload = {
        "norm_query": (norm_query or "").strip(),
        "any_tags": any_tags or [],
        "output_schema": {
            "intent_type": "global_pure|global_conditional|rag_catalog",
            "required_slots": ["string"],
            "missing_slots": ["string"],
            "should_ask_back": "boolean",
            "ask_back_questions": ["string"],
            "route_override": "GLOBAL|RAG|",
            "answer_mode_override": "GLOBAL_BOUNDED|",
            "confidence": "number",
            "reasons": ["string"]
        },
        "slot_hints": [
            "crop (cây trồng)", "days_to_harvest (còn bao nhiêu ngày thu hoạch)",
            "product_or_ai (tên sản phẩm/hoạt chất)", "phi_label (PHI theo nhãn)",
            "pest_or_disease (đối tượng)", "application_method (phun/tưới/rải)"
        ]
    }

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.0,
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        # sanitize nhỏ
        data["ask_back_questions"] = (data.get("ask_back_questions") or [])[:2]
        data["required_slots"] = data.get("required_slots") or []
        data["missing_slots"] = data.get("missing_slots") or []
        return data
    except Exception:
        return {}
