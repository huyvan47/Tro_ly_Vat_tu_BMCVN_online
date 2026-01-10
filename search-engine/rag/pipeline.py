from rag.config import RAGConfig
from rag.router import route_query
from rag.normalize import normalize_query
from rag.text_utils import is_listing_query, extract_img_keys
from rag.retriever import search as retrieve_search
from rag.scoring import fused_score, analyze_hits_fused
from rag.strategy import decide_strategy
from rag.text_utils import extract_codes_from_query
from rag.context_builder import choose_adaptive_max_ctx, build_context_from_hits
from rag.answer_modes import decide_answer_policy
from rag.formatter import format_direct_doc_answer
from rag.generator import call_finetune_with_context
from rag.verbatim import verbatim_export
from rag.tag_filter import infer_filters_from_query
from rag.debug_log import debug_log
from typing import List, Tuple, Dict, Any
from pathlib import Path
import re
import json, hashlib
from typing import Dict, Any

FORCE_MUST_TAGS = {
    "mechanisms:luu-dan-manh",
    "mechanisms:luu-dan",
    "mechanisms:tiep-xuc-manh",
    "mechanisms:tiep-xuc",
    "mechanisms:tiep-xuc-luu-dan-manh",
    "mechanisms:tiep-xuc-luu-dan",
    "mechanisms:xong-hoi-manh",
    "mechanisms:xong-hoi",
    "mechanisms:co-chon-loc",
    "mechanisms:khong-chon-loc",
}

_CONCEPTUAL_TRIGGERS = [
    "gần thu hoạch", "cách ly", "thời gian cách ly", "PHI", "mrl", "an toàn",
    "tận gốc", "diệt tận gốc", "chỉ ức chế",
    "nên chọn", "khác nhau", "ưu nhược", "cơ chế",
    "tiếp xúc", "lưu dẫn", "nội hấp",
    "mùi", "hôi", "tuyến trùng",
]

def promote_forced_tags(must_tags, any_tags):
    must = set(must_tags or [])
    anyt = set(any_tags or [])

    forced = anyt & FORCE_MUST_TAGS
    if forced:
        must |= forced
        anyt -= forced

    return list(must), list(anyt)


def _has_product_signal(norm_query: str, any_tags: List[str], code_candidates: List[str]) -> bool:
    q = (norm_query or "").lower()

    # 1) Có code/sku
    if code_candidates:
        return True

    # 2) Có tag product/entity (tuỳ schema bạn)
    if any(t.startswith("product:") or t.startswith("entity:product") for t in (any_tags or [])):
        return True

    # 3) Heuristic SKU phổ biến ngành BVTV
    if re.search(r"\b\d+(\.\d+)?\s?(ec|sc|wg|wp|sl|od|sp|wdg|gr|cs)\b", q):
        return True

    return False


def _need_intent_gate(norm_query: str, any_tags: List[str], code_candidates: List[str]) -> bool:
    # Có product signal thì vẫn có thể gate, nhưng mục tiêu là slots (không phải route)
    q = (norm_query or "").lower()
    conceptual = any(k in q for k in _CONCEPTUAL_TRIGGERS)

    # Nếu không conceptual thì khỏi gate
    if not conceptual:
        return False

    # Nếu conceptual nhưng có product signal -> gate = True (để lấy slots), nhưng tuyệt đối không auto GLOBAL
    return True



def _hits_have_slot_evidence(hits: List[dict], missing_slots: List[str]) -> bool:
    """Heuristic: nếu trong top hits đã có dấu hiệu thông tin cho các slot đang thiếu,
    thì không cần ask-back sớm."""
    if not hits:
        return False
    miss = set((missing_slots or []))
    if not miss:
        return True

    # map slot -> keywords (lower)
    slot_kw = {
        "phi_label": ["cách ly", "thời gian cách ly", "phi"],
        "days_to_harvest": ["thu hoạch", "gần thu hoạch", "trước thu hoạch"],
        "crop": ["cây", "trồng", "cây có múi", "cam", "quýt", "bưởi", "chanh"],
        "product_or_ai": ["hoạt chất", "thuốc", "sản phẩm"],
        "application_method": ["phun", "tưới", "rải", "xử lý đất", "tưới gốc"],
        "pest_or_disease": ["sâu", "bệnh", "tuyến trùng", "nấm", "rầy", "bọ trĩ"],
    }

    # build keyword list for missing slots
    kws = []
    for s in miss:
        kws.extend(slot_kw.get(s, []))
    kws = [k.lower() for k in kws if k]

    if not kws:
        return False

    # check top few docs for any keyword
    for h in hits[:6]:
        blob = f"{h.get('question','')}\n{h.get('answer','')}".lower()
        if any(k in blob for k in kws):
            return True
    return False
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

def _should_use_multi_query(norm_query: str, any_tags: List[str], answer_mode_hint: str = "") -> bool:
    """
    Heuristic bật multi-query:
    - Câu hỏi dạng liệt kê / taxonomy / "hoạt chất" / "nhóm" / "phân loại"
    - Hoặc có pest/crop nhưng query vẫn khó match (thường xảy ra)
    """
    q = (norm_query or "").lower()
    if any(k in q for k in ["hoạt chất", "active ingredient", "nhóm", "phân loại", "gồm những", "bao gồm", "so sánh", "phân biệt"]):
        return True
    # Nếu có pest/crop thì multi-query thường giúp recall tốt hơn
    if any(t.startswith("pest:") for t in (any_tags or [])) and any(t.startswith("crop:") for t in (any_tags or [])):
        return True
    # Nếu upstream đã hint answer_mode
    if answer_mode_hint in ("active_ingredient", "listing"):
        return True
    return False


def llm_build_sub_queries(
    *,
    client,
    norm_query: str,
    must_tags: List[str],
    any_tags: List[str],
    answer_mode_hint: str = "",
    max_variants: int = 5,
) -> List[Dict[str, str]]:
    """
    Dùng LLM để tạo sub-queries. Trả về list dict: {purpose, q}
    - Không tạo tags mới (tags đã có).
    - Output JSON ổn định để parse.
    """
    sys = (
        "Bạn là module tạo sub-query cho hệ thống RAG nông nghiệp.\n"
        "Nhiệm vụ: sinh các câu truy vấn ngắn, dễ match tài liệu, KHÔNG trả lời người dùng.\n"
        "Yêu cầu:\n"
        "- Chỉ trả về JSON hợp lệ.\n"
        "- Mỗi sub-query phải đánh vào một góc khác nhau: hoạt chất / sản phẩm / giai đoạn / từ đồng nghĩa / nhóm cây / (tuỳ chọn) tiếng Anh.\n"
        "- Tránh paraphrase đơn thuần. Không tạo câu quá dài.\n"
        "- Không bịa tags. Không thêm ký tự lạ.\n"
    )

    payload = {
        "query": (norm_query or "").strip(),
        "must_tags": must_tags or [],
        "any_tags": any_tags or [],
        "answer_mode_hint": answer_mode_hint or "",
        "max_variants": max_variants,
        "output_schema": {
            "variants": [
                {"purpose": "active_ingredient|product|stage|synonym|crop_group|en|general", "q": "string"}
            ]
        },
        "notes": [
            "Nếu query có 'hoạt chất' thì bắt buộc có ít nhất 1 variant purpose=active_ingredient.",
            "Nếu có giai đoạn (trái non/ra hoa/đậu trái) thì tạo 1 variant purpose=stage.",
            "Nếu crop là bưởi/cam/quýt/chanh thì tạo 1 variant purpose=crop_group (cây có múi/citrus).",
        ],
    }

    # Dùng chat.completions như code hiện tại của bạn
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.2,
        max_completion_tokens=600,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()

    try:
        data = json.loads(raw)
        variants = data.get("variants", []) or []
    except Exception:
        variants = []

    # sanitize + dedupe
    out: List[Dict[str, str]] = []
    seen = set()
    for it in variants:
        purpose = str(it.get("purpose", "general")).strip()
        q = str(it.get("q", "")).strip()
        if not q:
            continue
        q_norm = re.sub(r"\s+", " ", q).strip().lower()
        if q_norm in seen:
            continue
        if len(q_norm) > 180:
            continue
        seen.add(q_norm)
        out.append({"purpose": purpose, "q": q})
        if len(out) >= max_variants:
            break

    return out


def _purpose_weight(purpose: str) -> float:
    p = (purpose or "").strip().lower()
    # Weight theo mục tiêu (có thể chỉnh)
    if p == "active_ingredient":
        return 1.30
    if p == "stage":
        return 1.20
    if p == "crop_group":
        return 1.10
    if p == "synonym":
        return 1.00
    if p == "product":
        return 0.90
    if p == "en":
        return 0.80
    return 0.85


def _weighted_rrf_fuse(
    results_by_query: List[Dict[str, Any]],
    *,
    k: int = 60,
    top_n: int = 400,
) -> List[Dict[str, Any]]:
    """
    Weighted RRF fuse nhiều list hits (hit là dict có 'id', 'score', ...).
    Kết quả: list hits đã dedupe theo id, có thêm field:
      - hit['mq_rrf'] = điểm fused
      - hit['mq_sources'] = list purpose/query_index đã đóng góp (debug)
    """
    fused: Dict[str, float] = {}
    best: Dict[str, Dict[str, Any]] = {}
    sources: Dict[str, List[str]] = {}

    for item in results_by_query:
        weight = float(item["weight"])
        purpose = item["purpose"]
        qi = item["qi"]
        hits = item["hits"] or []

        for rank, h in enumerate(hits, start=1):
            doc_id = h.get("id")
            if not doc_id:
                continue
            rrf = weight / (k + rank)
            fused[doc_id] = fused.get(doc_id, 0.0) + rrf

            # giữ bản hit tốt nhất theo score gốc (để không mất fields)
            if doc_id not in best or float(h.get("score", 0.0)) > float(best[doc_id].get("score", 0.0)):
                best[doc_id] = h

            sources.setdefault(doc_id, []).append(f"{purpose}#{qi}")

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_n]
    out: List[Dict[str, Any]] = []
    for doc_id, s in ranked:
        h = best[doc_id]
        h["mq_rrf"] = float(s)
        h["mq_sources"] = sources.get(doc_id, [])
        out.append(h)
    return out

def render_global_bounded(
    *,
    client,
    user_query: str,
    norm_query: str,
    analysis: Dict[str, Any],
    max_tokens: int = 700,
) -> str:
    """
    Trả lời khung nguyên tắc + checklist + hỏi tối đa 2 câu.
    KHÔNG bịa nhãn/PHI/liều.
    """
    sys = (
        "Bạn là chuyên gia BVTV/nông học.\n"
        "Trả lời theo nguyên tắc an toàn:\n"
        "- Nếu thiếu dữ liệu quyết định (PHI/nhãn/cây trồng/hoạt chất/thời điểm) thì KHÔNG kết luận khẳng định.\n"
        "- Trả lời ngắn, đúng trọng tâm.\n"
        "- Có 3 phần:\n"
        "  (1) Nguyên tắc (2-4 gạch đầu dòng)\n"
        "  (2) Checklist thông tin cần có (gạch đầu dòng)\n"
        "  (3) Hỏi lại tối đa 2 câu (nếu analysis.should_ask_back=true)\n"
    )

    # Tận dụng analysis để chắc chắn không hỏi lan man
    payload = {
        "user_query": user_query,
        "norm_query": norm_query,
        "intent_type": analysis.get("intent_type", ""),
        "required_slots": analysis.get("required_slots", []),
        "missing_slots": analysis.get("missing_slots", []),
        "ask_back_questions": analysis.get("ask_back_questions", []),
    }

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.2,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

def _is_hard_global(q: str) -> bool:
    q = (q or "").lower()
    # Các câu dạng taxonomy/liệt kê/so sánh thường cần model mạnh hơn và output dài hơn
    return any(k in q for k in [
        "bao gồm", "gồm những", "gồm các", "phân loại", "nhóm nào",
        "khác gì", "so sánh", "phân biệt", "triệu chứng", "cơ chế"
    ])

def _global_system_prompt() -> str:
    return """
Bạn là chuyên gia BVTV/nông học.
Trả lời theo kiểu giáo trình, dùng thuật ngữ phổ biến tại Việt Nam.

Quy tắc:
- Trả lời theo cấu trúc rõ ràng, có tiêu đề.
- Bắt buộc có: (1) Định nghĩa ngắn gọn, (2) Tiêu chí nhận biết/đặc điểm chính.
- Nếu câu hỏi hỏi “bao gồm/gồm những loại nào/phân loại” thì bắt buộc có mục: (3) Phân loại + (4) Ví dụ đại diện (ưu tiên nhóm/loài thường gặp trong canh tác).
- Nếu không chắc một tên loài/thuật ngữ: ghi “thường gặp” và mô tả theo nhóm, không bịa.
- Ưu tiên trả lời đúng trọng tâm, không lan man sang công dụng phủ đất/chống xói mòn nếu không liên quan câu hỏi.
""".strip()

def choose_top_k(
    is_list: bool,
    must_tags: List[str],
    any_tags: List[str],
    norm_query: str,
    base_list: int = 80,
    base_normal: int = 50,
) -> int:
    """
    Quy tắc tăng top_k:
    - Nếu có entity:product => tăng mạnh (truy vấn về sản phẩm)
    - Nếu có pest:/disease: => tăng mạnh (truy vấn điều trị sâu/bệnh cần recall cao)
    - Nếu có crop: => tăng vừa (narrow theo cây trồng)
    - Nếu query có ý hỏi "thuốc gì/phun gì/trị gì" => tăng thêm
    """
    top_k = base_list if is_list else base_normal

    tags_all = (must_tags or []) + (any_tags or [])

    has_product = any(t == "entity:product" or t.startswith("entity:") or t.startswith("product:") for t in tags_all)
    has_pest = any(t.startswith("pest:") for t in tags_all)
    has_disease = any(t.startswith("disease:") for t in tags_all)
    has_crop = any(t.startswith("crop:") for t in tags_all)

    # Heuristic theo intent ngôn ngữ
    ask_recommend = bool(re.search(r"\b(thuoc|phun|tri|phong|xu ly|dung gi|nen dung|loai nao)\b", norm_query.lower()))

    # Tăng mạnh cho bài toán "tìm sản phẩm / tư vấn sâu bệnh"
    if has_product:
        top_k = max(top_k, 160 if not is_list else 220)

    if has_pest or has_disease:
        top_k = max(top_k, 220 if not is_list else 300)

    # Crop giúp thu hẹp, nhưng vẫn cần recall nếu kèm pest/disease
    if has_crop and not (has_pest or has_disease):
        top_k = max(top_k, 120 if not is_list else 160)

    if ask_recommend:
        top_k = int(top_k * 1.2)

    # Giới hạn trên để tránh quá tải
    top_k = min(top_k, 400)

    return top_k

def answer_with_suggestions(*, user_query, kb, client, cfg, policy):
    # 0) Route GLOBAL / RAG
    route = route_query(client, user_query)
    if route == "GLOBAL":
        hard = _is_hard_global(user_query)
        model = "gpt-4.1" if hard else "gpt-4.1-mini"
        resp = client.chat.completions.create(
            model=model,
            temperature=0.25 if hard else 0.35,
            max_completion_tokens=3500 if hard else 2500,
            messages=[
                {"role": "system", "content": _global_system_prompt()},
                {"role": "user", "content": user_query},
            ],
        )
        text = resp.choices[0].message.content.strip()

        # (Tuỳ chọn) Escalate lần 2 nếu dùng mini nhưng output thiếu cấu trúc/liệt kê
        if (not hard) and _is_hard_global(user_query):
            # nếu router vẫn GLOBAL nhưng mini trả lời quá ngắn/thiếu ý
            if len(text) < 900 or ("Định nghĩa" not in text and "Phân loại" not in text):
                resp2 = client.chat.completions.create(
                    model="gpt-4.1",
                    temperature=0.2,
                    max_completion_tokens=3800,
                    messages=[
                        {"role": "system", "content": _global_system_prompt()},
                        {"role": "user", "content": user_query},
                        {"role": "user", "content": "Hãy mở rộng theo đúng cấu trúc, bổ sung phân loại và ví dụ đại diện nếu câu hỏi yêu cầu liệt kê."},
                    ],
                )
                text = resp2.choices[0].message.content.strip()
        return {
            "text": text,
            "img_keys": [],
            "route": "GLOBAL",
            "norm_query": "",
            "strategy": f"GLOBAL/{model}",
            "profile": {"top1": 0, "top2": 0, "gap": 0, "mean5": 0, "n": 0, "conf": 0},
        }

    # 1) Normalize query
    norm_query = normalize_query(client, user_query)

    # 2) Listing?
    is_list = is_listing_query(norm_query)

    must_tags, any_tags = infer_filters_from_query(norm_query)

    code_candidates = extract_codes_from_query(norm_query)

    has_product_signal = _has_product_signal(norm_query, any_tags, code_candidates)

    must_tags, any_tags = promote_forced_tags(must_tags, any_tags)
    # ---- NEW: intent/slot gate ----
    analysis = {}
    if _need_intent_gate(norm_query, any_tags, code_candidates):
        analysis = analyze_intent_and_slots(
            client=client,
            norm_query=norm_query,
            any_tags=any_tags,
        )
    # 1) Override route nếu cần (GLOBAL vs RAG)
    route_override = (analysis.get("route_override") or "").strip().upper()

    # 2) Nếu global_pure -> có thể ép GLOBAL luôn (tuỳ bạn)
    if analysis.get("intent_type") == "global_pure" and route_override != "RAG":
        # Dùng luôn global prompt hiện có của bạn
        # (Có thể reuse nhánh route == "GLOBAL" ở đầu)
        route_override = "GLOBAL"

    # 3) Nếu global_conditional và thiếu slot:
    # - Nếu query có dấu hiệu sản phẩm cụ thể (SKU/tag/code) => KHÔNG ask-back sớm, ưu tiên RAG
    # - Nếu không có product signal => thử quick-retrieve; chỉ ask-back nếu KB không có dấu hiệu slot cần thiết
    if analysis.get("intent_type") == "global_conditional" and bool(analysis.get("should_ask_back")):
        if not has_product_signal:
            quick_hits = retrieve_search(
                client=client,
                kb=kb,
                norm_query=norm_query,
                top_k=40,
                must_tags=must_tags,
                any_tags=any_tags,
            )
            if not _hits_have_slot_evidence(quick_hits, analysis.get("missing_slots", [])):
                text = render_global_bounded(
                    client=client,
                    user_query=user_query,
                    norm_query=norm_query,
                    analysis=analysis,
                )
                return {
                    "text": text,
                    "img_keys": [],
                    "route": "GLOBAL_BOUNDED",
                    "norm_query": norm_query,
                    "strategy": "GLOBAL_BOUNDED",
                    "profile": {"top1": 0, "top2": 0, "gap": 0, "mean5": 0, "n": 0, "conf": 0},
                    "intent_type": analysis.get("intent_type", ""),
                    "missing_slots": analysis.get("missing_slots", []),
                }
        # nếu có product signal hoặc quick_hits đã có dấu hiệu slot => tiếp tục RAG như bình thường

    # 4) Nếu route_override = GLOBAL mà không cần hỏi -> chạy nhánh GLOBAL
    if route_override == "GLOBAL":
        hard = _is_hard_global(user_query)
        model = "gpt-4.1" if hard else "gpt-4.1-mini"
        resp = client.chat.completions.create(
            model=model,
            temperature=0.25 if hard else 0.35,
            max_completion_tokens=3500 if hard else 2500,
            messages=[
                {"role": "system", "content": _global_system_prompt()},
                {"role": "user", "content": user_query},
            ],
        )
        text = resp.choices[0].message.content.strip()
        return {
            "text": text,
            "img_keys": [],
            "route": "GLOBAL",
            "norm_query": norm_query,
            "strategy": f"GLOBAL/{model}",
            "profile": {"top1": 0, "top2": 0, "gap": 0, "mean5": 0, "n": 0, "conf": 0},
            "intent_type": analysis.get("intent_type", ""),
            "missing_slots": analysis.get("missing_slots", []),
        }


    top_k = choose_top_k(
        is_list=is_list,
        must_tags=must_tags,
        any_tags=any_tags,
        norm_query=norm_query,
        base_list=80,
        base_normal=50,
    )
    print("QUERY      :", norm_query)
    print("MUST TAGS  :", must_tags)
    print("ANY TAGS   :", any_tags)

    answer_mode_hint = ""  # hoặc bạn tự set theo entity_type/intent upstream nếu đã có

    use_mq = _should_use_multi_query(norm_query, any_tags, answer_mode_hint)

    if not use_mq:
        debug_log(
            f"norm_query: {norm_query}",
        )
        hits = retrieve_search(
            client=client,
            kb=kb,
            norm_query=norm_query,
            top_k=top_k,
            must_tags=must_tags,
            any_tags=any_tags,
        )
    else:
        # 1) LLM tạo sub-queries
        variants = llm_build_sub_queries(
            client=client,
            norm_query=norm_query,
            must_tags=must_tags,
            any_tags=any_tags,
            answer_mode_hint=answer_mode_hint,
            max_variants=5,
        )

        # 2) Luôn include query gốc như 1 "mũi tên" chính
        queries = [{"purpose": "main", "q": norm_query}] + variants

        print("[MQ] variants:")
        for i, v in enumerate(queries):
            print(f"  - #{i} purpose={v['purpose']} q={v['q']}")

        # 3) Retrieve từng sub-query (top_k nhỏ hơn để tránh rác)
        per_top_k = max(30, int(top_k * 0.45))  # bạn có thể chỉnh
        results_by_query = []
        for qi, v in enumerate(queries):
            qv = v["q"]
            debug_log(
                f"Sub question: {qv}",
            )
            purpose = v["purpose"]
            weight = 1.15 if purpose == "main" else _purpose_weight(purpose)

            sub_hits = retrieve_search(
                client=client,
                kb=kb,
                norm_query=qv,
                top_k=per_top_k,
                must_tags=must_tags,
                any_tags=any_tags,
            )
            results_by_query.append({
                "qi": qi,
                "purpose": purpose,
                "weight": weight,
                "hits": sub_hits,
            })

        # 4) Fuse (Weighted RRF) -> hits cuối
        hits = _weighted_rrf_fuse(results_by_query, k=60, top_n=top_k)

        # Debug
        print("[MQ] fused hits:", len(hits))
        if hits:
            print("[MQ] top1 id=", hits[0].get("id"), "mq_rrf=", hits[0].get("mq_rrf"), "score=", hits[0].get("score"))
    # === END MULTI-QUERY RETRIEVAL ===
    
    print("[DEBUG] after search: len(results)=", len(hits))
    if hits:
        print("[DEBUG] first id=", hits[0].get("id"), "score=", hits[0].get("score"))

    # Nếu có bước lọc/rerank sau đó, log tiếp ngay sau mỗi bước:
    # results = ...
    print("[DEBUG] after post-filter: len(results)=", len(hits))

    if not hits:
        return {
            "text": "Không tìm thấy dữ liệu phù hợp.",
            "img_keys": [],
            "route": "RAG",
            "norm_query": norm_query,
            "strategy": "NO_HITS",
            "profile": {"top1": 0, "top2": 0, "gap": 0, "mean5": 0, "n": 0, "conf": 0},
        }

    # 4) Filter by MIN_SCORE_MAIN
    def _count_tag_hits(h, any_tags, must_tags):
        tv2 = str(h.get("tags_v2") or "")
        score = 0
        # must: ưu tiên cao hơn any
        for t in (must_tags or []):
            if t and t in tv2:
                score += 3
        for t in (any_tags or []):
            if t and t in tv2:
                score += 1
        return score

    for h in hits:
        h["fused_score"] = fused_score(h)
        h["tag_hits"] = _count_tag_hits(h, any_tags, must_tags)

    # re-rank: tag_hits trước, rồi fused_score
    hits = sorted(
        hits,
        key=lambda x: (
            x.get("tag_hits", 0),
            x.get("mq_rrf", 0.0),       # NEW: ưu tiên điểm fused từ multi-query
            x.get("fused_score", 0.0),
        ),
        reverse=True,
    )

    filtered_for_main = [h for h in hits if h["fused_score"] >= policy.min_score_main]

    # NEW: fallback nếu threshold lọc sạch (đặc biệt listing/brand)
    if not filtered_for_main:
        fallback_n = min(len(hits), 10 if is_list else 5)
        filtered_for_main = hits[:fallback_n]

    # 5) Decide strategy (DIRECT_DOC / RAG_STRICT / RAG_SOFT)
    has_main = len(filtered_for_main) > 0
    prof = analyze_hits_fused(hits)
    strategy = decide_strategy(
        norm_query=norm_query,
        prof=prof,
        has_main=has_main,
        policy=policy,
        code_boost_direct=cfg.code_boost_direct,
    )

    # 6) Prefer include_in_context if available
    context_candidates = [h for h in filtered_for_main if h.get("include_in_context", False)]
    if not context_candidates:
        context_candidates = filtered_for_main

    # NEW: guard cuối
    if not context_candidates:
        return {
            "text": "Không tìm thấy ngữ cảnh phù hợp theo tiêu chí hiện tại.",
            "img_keys": [],
            "route": "RAG",
            "norm_query": norm_query,
            "strategy": "EMPTY_CONTEXT",
            "profile": analyze_hits_fused(hits),
        }

    # 7) Pick primary_doc (prefer code match)
    primary_doc = None
    if code_candidates:
        target = code_candidates[0].lower()
        for h in context_candidates:
            if target in h["question"].lower() or target in h["answer"].lower():
                primary_doc = h
                break
    if primary_doc is None:
        primary_doc = context_candidates[0]

    # 10) Build context and call GPT answer (STRICT/SOFT)
    policy = decide_answer_policy(user_query, primary_doc, force_listing=is_list)
    answer_mode = "listing" if policy.format == "listing" else policy.intent
    # === VERBATIM short-circuit ===
    if answer_mode == "verbatim":
        vb = verbatim_export(
            kb=kb,
            hits_router=hits,        # dùng hits hiện có (đã sorted/reranked), khỏi retrieve lại
        )
        return {
            "text": vb.get("text", ""),
            "img_keys": vb.get("img_keys", []),
            "route": "RAG",
            "norm_query": norm_query,
            "strategy": "VERBATIM",
            "profile": prof,
        }

    # 9) DIRECT_DOC: KB đủ mạnh -> trả trực tiếp doc (không ép QA)
    if strategy == "DIRECT_DOC":
        img_keys = extract_img_keys(primary_doc.get("answer", ""))
        text = format_direct_doc_answer(user_query, primary_doc)
        return {
            "text": text,
            "img_keys": img_keys,
            "route": "RAG",
            "norm_query": norm_query,
            "strategy": strategy,
            "profile": prof,
        }

    # adaptive ctx, nhưng giới hạn theo mode
    base_ctx = choose_adaptive_max_ctx(hits, is_listing=is_list)
    if strategy == "RAG_SOFT":
        max_ctx = min(RAGConfig.max_ctx_soft, base_ctx)
        rag_mode = "SOFT"
    else:
        max_ctx = min(RAGConfig.max_ctx_strict, base_ctx)
        rag_mode = "STRICT"

    main_hits = [primary_doc]
    for h in context_candidates:
        if h is not primary_doc and len(main_hits) < max_ctx:
            main_hits.append(h)

    def _short(x, n=160):
        x = "" if x is None else str(x)
        return x if len(x) <= n else x[:n] + "..."
    
    # Sau đó build context như bình thường
    context = build_context_from_hits(main_hits)

    Path("debug_ctx").mkdir(exist_ok=True)
    with open("debug_ctx/context_last.txt", "w", encoding="utf-8") as f:
        f.write(context)

    final_answer = call_finetune_with_context(
        client=client,
        user_query=user_query,
        context=context,
        answer_mode=answer_mode,
        rag_mode=rag_mode,  
    )

    img_keys = extract_img_keys(primary_doc.get("answer", ""))
    return {
        "text": final_answer,
        "img_keys": img_keys,
        "route": "RAG",
        "norm_query": norm_query,
        "strategy": strategy,
        "profile": prof,
    }