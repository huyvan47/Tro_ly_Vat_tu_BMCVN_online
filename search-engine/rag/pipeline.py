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
from rag.logging.multi_query_logger import write_multi_query_logs
# from rag.verbatim import verbatim_export
from rag.tag_filter import tag_filter_pipeline
from rag.logging.timing_logger import TimingLog
from rag.reasoning.multi_hop import multi_hop_controller
from typing import List, Tuple, Dict, Any
from rag.logging.debug_log import debug_log
from rag.post_answer.enricher import enrich_answer_if_needed
from typing import Optional
import unicodedata
import re
import json


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

FORMULA_TRIGGERS = [
    "công thức",
    "phối ",
    "phối trộn",
    "phối hợp thuốc",
    "pha thuốc",
    "phối hợp",
    "kết hợp", "combo",
    "phối thuốc", 
    "kết hợp thuốc",
    "công thức trị",
    "công thức trừ",
    "công thức diệt",
    "phác đồ",
    "liều phối",
    "hoạt chất lưu dẫn phù hợp",
]

_CONCEPTUAL_TRIGGERS = [
    "gần thu hoạch", "cách ly", "thời gian cách ly", "PHI", "mrl", "an toàn",
    "tận gốc", "diệt tận gốc", "chỉ ức chế",
    "nên chọn", "khác nhau", "ưu nhược", "cơ chế",
    "tiếp xúc", "lưu dẫn", "nội hấp",
    "mùi", "hôi", "tuyến trùng",
]

FORMULA_KEYWORDS = [
        "phối", "trộn", "pha", "kết hợp", "mix",
        "phối hợp", "pha chung", "dùng chung",
        "xài chung", "kết hợp thuốc",
        "phối thuốc", "trộn thuốc",
    ]


def is_formula_query(query: str, tags: dict) -> bool:
    """
    Nhận diện truy vấn dạng phối công thức.
    """

    q = (query or "").lower()

    # 1. Cú pháp
    has_plus = "+" in q

    # 2. Tag semantics
    all_tags = (tags.get("must", []) or []) + (tags.get("soft", []) or [])
    mech_tags = [t for t in all_tags if t.startswith("mechanisms:")]
    has_mechanisms = len(mech_tags) > 0
    num_mechs = len(mech_tags)

    # 3. Ngôn ngữ tự nhiên
    has_formula_keywords = any(k in q for k in FORMULA_KEYWORDS)

    # Heuristic quyết định
    if has_formula_keywords:
        return True

    if has_mechanisms and (has_plus or num_mechs >= 2):
        return True

    if has_plus and len(all_tags) >= 2:
        return True

    return False


def formula_mode_search(
    *,
    client,
    kb,
    norm_query: str,
    must_tags: List[str],
    soft_tags: List[str],
    top_k: int
):
    """
    Tìm kiếm theo chế độ công thức (không dùng multi-hop).
    """

    all_results = []

    # ---- ROLE 1: MUST TAG ----
    for m in must_tags:
        hits = retrieve_search(
            client=client,
            kb=kb,
            norm_query=norm_query,
            top_k=top_k,
            must_tags=[m],
            any_tags=[]
        )
        all_results.extend(hits)

    # ---- ROLE 2: SOFT TAG ----
    for s in soft_tags:
        hits = retrieve_search(
            client=client,
            kb=kb,
            norm_query=norm_query,
            top_k=top_k,
            must_tags=[s],
            any_tags=[]
        )
        all_results.extend(hits)

    # Dedupe theo id
    unique = {}
    for h in all_results:
        hid = h.get("id")
        if hid:
            unique[hid] = h

    return list(unique.values())

def preserve_search_order(hits):
    """
    Đánh dấu thứ tự gốc từ search() để pipeline KHÔNG làm xáo trộn.
    """
    for idx, h in enumerate(hits):
        h["_search_rank"] = idx
    return hits

def _count_tag_hits(h, any_tags, must_tags):
    tv2 = str(h.get("tags_v2") or "")
    score = 0
    for t in (must_tags or []):
        if t and t in tv2:
            score += 3
    for t in (any_tags or []):
        if t and t in tv2:
            score += 1
    return score

_space_re = re.compile(r"\s+")

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("đ", "d") 
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = _space_re.sub(" ", s)
    return s


def _need_intent_gate(norm_query: str) -> bool:
    # Có product signal thì vẫn có thể gate, nhưng mục tiêu là slots (không phải route)
    q = (norm_query or "").lower()
    conceptual = any(k in q for k in _CONCEPTUAL_TRIGGERS)

    # Nếu không conceptual thì khỏi gate
    if not conceptual:
        return False

    # Nếu conceptual nhưng có product signal -> gate = True (để lấy slots), nhưng tuyệt đối không auto GLOBAL
    return True

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
        "- Mỗi sub-query 100% là tiếng Việt (trừ tên hoạt chất)."
        "- Mỗi sub-query phải đánh vào một góc khác nhau: hoạt chất / sản phẩm / giai đoạn / từ đồng nghĩa / nhóm cây.\n"
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

def infer_answer_intent(q: str, found_groups: Dict[str, List[str]]):
    qn = _norm(q)

    has_product_term = bool(re.search(r"\b(thuoc|thuốc|san pham|sản phẩm)\b", qn))
    has_symptom_term = bool(re.search(r"\b(trieu chung|triệu chứng|benh|bệnh)\b", qn))

    if has_symptom_term:
        return "disease"
    if has_product_term:
        return "product"
    return "general"

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

def extract_products_from_tags(tags):
    products = []
    for t in tags:
        if t.startswith("product:"):
            products.append(t.split(":", 1)[1])
    return products

def extract_primary_chemical_from_docs(docs):
    seen = set()
    chemicals = []

    for doc in docs:
        tv2 = doc.get("tags_v2") or ""          # tags_v2 là string
        if not tv2:
            continue

        tags = tv2.split("|")                  # <<< QUAN TRỌNG

        for t in tags:
            t = (t or "").strip()
            if t.startswith("chemical:"):
                chem = t.split(":", 1)[1].strip()
                if chem and chem not in seen:
                    seen.add(chem)
                    chemicals.append(chem)

    if not chemicals:
        return None

    # chỉ lấy 1 hoạt chất đầu tiên
    return chemicals[0]


def build_soft_tags(primary_chemical):
    tags = []
    if primary_chemical:
        tags.append(f"chemical:{primary_chemical}")
    return tags

def retrieve_product_docs(*, client, kb, norm_query, product_slug, top_k=50):
    raw = retrieve_search(
        client=client,
        kb=kb,
        norm_query=norm_query,
        top_k=top_k,
        must_tags=[product_slug],
    )
    return [
        d for d in raw
        if f"product:{product_slug}" in (d.get("tags_v2") or "")
    ]

def build_soft_tags_from_chemicals(chemicals: List[str], max_chemicals: int = 2) -> List[str]:
    out = []
    seen = set()
    for c in chemicals:
        c = (c or "").strip()
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(f"chemical:{c}")
        if len(out) >= max_chemicals:
            break
    return out

def infer_primary_chemical_for_product(*, client, kb, norm_query: str, product_slug: str) -> Optional[str]:
    docs_raw = retrieve_search(
        client=client,
        kb=kb,
        norm_query=norm_query,
        top_k=80,                 # lấy rộng để không miss
        must_tags=[product_slug], # giữ contract must_tags list
    )

    # filter cứng đúng product
    docs = [
        d for d in (docs_raw or [])
        if f"product:{product_slug}" in (d.get("tags_v2") or "")
    ]

    return extract_primary_chemical_from_docs(docs)

def answer_with_suggestions(*, user_query, kb, client, cfg, policy):
    timer = TimingLog(user_query)
    # -----------------------------------------------------
    # 0) ROUTER – QUYỀN CAO NHẤT
    # -----------------------------------------------------
    route = route_query(client, user_query)

    norm_query = normalize_query(client, user_query)
    norm_lower = norm_query.lower()

    force_rag = any(k in norm_lower for k in FORMULA_TRIGGERS)

    if force_rag:
        route = "RAG"

    print("route:", route)
    timer.mark("router")

    # KHÓA ROUTE: nếu router đã chọn RAG thì CẤM quay về GLOBAL
    route_locked = (route == "RAG")

    # -----------------------------------------------------
    # 1) NHÁNH GLOBAL NGAY TỪ ĐẦU (chỉ khi router cho phép)
    # -----------------------------------------------------
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
        return {
            "text": text,
            "img_keys": [],
            "route": "GLOBAL",
            "norm_query": "",
            "strategy": f"GLOBAL/{model}",
            "profile": {"top1": 0, "top2": 0, "gap": 0, "mean5": 0, "n": 0, "conf": 0},
        }

    # -----------------------------------------------------
    # 2) NORMALIZE + TAGS
    # -----------------------------------------------------
    norm_query = normalize_query(client, user_query)
    norm_lower = norm_query.lower()

    force_rag = False

    if any(k in norm_lower for k in FORMULA_TRIGGERS):
        force_rag = True

    if re.search(r"hoạt chất.*lưu dẫn", norm_lower):
        force_rag = True

    timer.mark("normalize")
    is_list = is_listing_query(norm_query)

    # LUÔN chạy tag_filter_pipeline để đảm bảo có result
    result = tag_filter_pipeline(norm_query)
    timer.mark("tag_filter")

    if force_rag:
        route = "RAG"
        route_locked = True


    must_tags = result.get("must", [])
    soft_tags = result.get("soft", [])
    any_tags = result.get("any", [])
    found = result.get("found", {})

    products = extract_products_from_tags(must_tags)
    print("extract_products_from_tags:", products)

    if is_formula_query(norm_query, result):

        # CASE 1: Product + Chemical (giữ như bạn đang làm)
        if len(products) == 1:
            product_slug = products[0]

            docs_raw = retrieve_search(
                client=client,
                kb=kb,
                norm_query=norm_query,
                top_k=80,
                must_tags=[product_slug],
            )

            docs = [
                d for d in (docs_raw or [])
                if f"product:{product_slug}" in (d.get("tags_v2") or "")
            ]

            primary_chemical = extract_primary_chemical_from_docs(docs)
            soft_tags = build_soft_tags(primary_chemical)

            print("primary_chemical:", primary_chemical)
            print("soft_tags (chemical-only):", soft_tags)

        # CASE 2: Product + Product (NÂNG CẤP: suy chemical cho từng product)
        elif len(products) >= 2:
            p1, p2 = products[0], products[1]

            c1 = infer_primary_chemical_for_product(
                client=client, kb=kb, norm_query=norm_query, product_slug=p1
            )
            c2 = infer_primary_chemical_for_product(
                client=client, kb=kb, norm_query=norm_query, product_slug=p2
            )

            print("p1,c1:", p1, c1)
            print("p2,c2:", p2, c2)

            chemicals = [c for c in [c1, c2] if c]
            soft_tags = build_soft_tags_from_chemicals(chemicals, max_chemicals=2)

            # OPTIONAL (nếu muốn tăng recall evidence theo sản phẩm):
            # soft_tags += [f"product:{p1}", f"product:{p2}"]

            print("soft_tags (chemical-first):", soft_tags)

        else:
            soft_tags = []
            print("[WARN] Formula query but no product resolved")

        print("[MODE] Formula-based retrieval")
        hits = formula_mode_search(
            client=client,
            kb=kb,
            norm_query=norm_query,
            must_tags=[],          # formula path: không dùng must cũ để tránh ép sai
            soft_tags=soft_tags,   # chemical-only (hoặc chemical-first)
            top_k=RAGConfig.multi_query_top_k
        )

    else:
        print("[MODE] Knowledge multi-hop retrieval")
        hits = multi_hop_controller(
            client=client,
            kb=kb,
            base_query=norm_query,
            must_tags=must_tags,
            any_tags=any_tags,
            timer=timer,
        )


    effective_must = must_tags
    print("effective_must: ", effective_must)
    effective_any = any_tags + soft_tags
    print("effective_any: ", effective_any)
    # answer_mode = result.get("answer_mode", "")

    code_candidates = extract_codes_from_query(norm_query)

    # -----------------------------------------------------
    # 3) INTENT / SLOT ANALYZER (LLM)
    # -----------------------------------------------------
    analysis: Dict[str, Any] = {}
    if _need_intent_gate(norm_query):
        analysis = analyze_intent_and_slots(
            client=client,
            norm_query=norm_query,
            any_tags=any_tags,
        )
        timer.mark("intent_analysis")

    route_override = (analysis.get("route_override") or "").strip().upper()

    # ❗ INTENT CHỈ ĐƯỢC ÉP GLOBAL KHI ROUTER KHÔNG KHÓA
    if analysis.get("intent_type") == "global_pure":
        if not route_locked and route_override != "RAG":
            route_override = "GLOBAL"

    # ❗ GUARD CUỐI: ROUTER LUÔN THẮNG
    if route_locked:
        route_override = "RAG"

    # -----------------------------------------------------
    # 4) NHÁNH GLOBAL SAU INTENT (CHỈ KHI ĐƯỢC PHÉP)
    # -----------------------------------------------------
    if route_override == "GLOBAL" and not force_rag:
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
        }

    # -----------------------------------------------------
    # 5) RAG PIPELINE (KHÔNG BAO GIỜ QUAY VỀ GLOBAL)
    # ----------------------------------------------------

    print("QUERY      :", norm_query)
    print("MUST TAGS  :", must_tags)
    print("ANY TAGS   :", any_tags)
    print("SOFT TAGS   :", soft_tags)
    debug_log("QUERY      :", norm_query)
    debug_log("MUST TAGS  :", must_tags)
    debug_log("SOFT TAGS   :", soft_tags)

    # -----------------------------------------------------
    # 5) RAG PIPELINE – SINGLE hoặc MULTI QUERY
    # -----------------------------------------------------

    if not hits:
        return {
            "text": "Không tìm thấy dữ liệu phù hợp.",
            "img_keys": [],
            "route": "RAG",
            "norm_query": norm_query,
            "strategy": "NO_HITS",
            "profile": {"top1": 0, "top2": 0, "gap": 0, "mean5": 0, "n": 0, "conf": 0},
        }

    hits = preserve_search_order(hits)

    for h in hits:
        # chỉ để phân tích / debug / profile
        h["fused_score"] = fused_score(h)
        h["tag_hits"] = _count_tag_hits(h, any_tags, must_tags)

    prof = analyze_hits_fused(hits)
    strategy = decide_strategy(
        norm_query=norm_query,
        prof=prof,
        has_main=True,
        policy=policy,
        code_boost_direct=cfg.code_boost_direct,
    )

    primary_doc = hits[0]

    # -----------------------------------------------------
    # 7) DIRECT DOC
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # 8) BUILD CONTEXT + GENERATE
    # -----------------------------------------------------
    base_ctx = choose_adaptive_max_ctx(hits, is_listing=is_list)
    max_ctx = min(RAGConfig.max_ctx_strict, base_ctx)

    context = build_context_from_hits(hits[:max_ctx])
    timer.mark("build_context")

    answer_intent = infer_answer_intent(user_query, found)
    policy = decide_answer_policy(user_query, primary_doc, parsed_intent=answer_intent, force_listing=is_list)
    answer_mode = "listing" if policy.format == "listing" else policy.intent
    if policy.format == "listing":
        answer_mode_final = "listing"
    else:
        answer_mode_final = answer_mode
    final_answer = call_finetune_with_context(
        client=client,
        user_query=user_query,
        context=context,
        answer_mode=answer_mode_final,
        rag_mode="STRICT",
    )
    timer.mark("llm_generate")

    final_answer = enrich_answer_if_needed(
        client=client,
        user_query=user_query,
        answer_text=final_answer,
        answer_mode=answer_mode_final,
        any_tags=effective_any,
        must_tags=effective_must,
        route="RAG",
    )
    timer.mark("enrich_answer_if_needed")

    img_keys = extract_img_keys(primary_doc.get("answer", ""))

    timer.finish(RAGConfig.enable_timing_log)
    return {
        "text": final_answer,
        "img_keys": img_keys,
        "route": "RAG",
        "norm_query": norm_query,
        "strategy": strategy,
        "profile": prof,
        "context_build": context,
    }
