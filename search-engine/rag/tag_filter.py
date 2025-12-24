import re
import unicodedata
from typing import List

ENTITY_TYPES = ["registry", "product", "disease", "procedure", "pest", "weed", "general"]

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    # bỏ dấu
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    # chuẩn hoá khoảng trắng
    s = re.sub(r"\s+", " ", s)
    return s

# Alias hoạt chất: bổ sung dần theo KB của bạn
CHEMICAL_ALIASES = {
    "mancozeb": ["mancozeb", "m45", "mz"],
    "metalaxyl": ["metalaxyl", "metaxyl"],
    "propiconazole": ["propiconazole", "propicol"],
}

PEST_ALIASES = {
    "ant":   ["kien", "con kien", "dan kien", "kien vang", "kien den", "ant", "ants"],
    "snail": ["oc", "oc buou", "oc buou vang", "buou vang", "snail", "snails"],
}

def extract_pests(q: str) -> List[str]:
    qn = _norm(q)
    found = []
    for pest, aliases in PEST_ALIASES.items():
        for a in aliases:
            if re.search(rf"\b{re.escape(a)}\b", qn):
                found.append(pest)
                break
    return found

def extract_chemicals(q: str) -> List[str]:
    """
    Trả về danh sách hoạt chất chuẩn hóa (vd: ['mancozeb']) nếu bắt được trong query.
    """
    qn = _norm(q)
    found = []
    for chem, aliases in CHEMICAL_ALIASES.items():
        for a in aliases:
            if re.search(rf"\b{re.escape(a)}\b", qn):
                found.append(chem)
                break
    # dedup giữ thứ tự
    out, seen = [], set()
    for x in found:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def infer_entity_type(q: str):
    qn = _norm(q)

    patterns = {
        "procedure": [
            (r"\b(quy trinh|cac buoc|huong dan|lam the nao|cach)\b", 2),
            (r"\b(pha|phun|xit|tuoi|bon|rai|tron|xu ly|ngam)\b", 1),
            (r"\b(lieu|lieu luong|nong do|dinh ky|thoi diem)\b", 1),
            (r"\b(binh\s*(16|25)l|ml|lit|l|g|kg|ha|%)\b", 1),
        ],
        "product": [
            (r"\b(thuoc gi|ten thuoc|san pham|hang|nha san xuat)\b", 2),
            (r"\b(hoat chat|ai|thanh phan)\b", 2),
            (r"\b(wp|wg|sc|ec|sl|gr|od|df|sp|cs|fs)\b", 1),
            (r"\b(gia|mua o dau|dai ly|phan phoi|tuong duong|thay the)\b", 1),
        ],
        "disease": [
            (r"\b(benh|nam benh|trieu chung|phong tri benh)\b", 2),
            (r"\b(thoi|dom|chay la|vang la|heo|xi mu|moc|ri sat)\b", 1),
            # nếu bạn muốn: tên tác nhân phổ biến
            (r"\b(phytophthora|fusarium|anthracnose)\b", 2),
        ],
        "pest": [
            (r"\b(sau|bo|ray|ruoi|rep|bo tri|nhen|mot|sung|tuyen trung)\b", 2),
            (r"\b(phong tru sau|diet ray|tru sau)\b", 2),
        ],
        "weed": [
            (r"\b(co dai|co)\b", 2),
            (r"\b(diet co|tru co)\b", 2),
            (r"\b(tien nay mam|hau nay mam|la rong|la hep)\b", 1),
        ],
        "registry": [
            (r"\b(dang ky|giay phep|so dang ky|ma so)\b", 2),
            (r"\b(danh muc|duoc phep|cam|han che)\b", 2),
            (r"\b(thong tu|nghi dinh|co quan|cuc|gia han|thoi han)\b", 1),
            (r"\b(tra cuu|verify|hop phap|tem nhan)\b", 1),
        ],
    }

    score = {k: 0 for k in ENTITY_TYPES}
    for et, rules in patterns.items():
        for pat, w in rules:
            if re.search(pat, qn):
                score[et] += w

    # chọn top1, top2
    ranked = sorted([(et, sc) for et, sc in score.items() if et != "general"], key=lambda x: x[1], reverse=True)
    top1, s1 = ranked[0]
    top2, s2 = ranked[1]

    # ngưỡng tối thiểu
    if s1 <= 0:
        return "general", score

    # tie/near-tie
    if (s1 - s2) <= 1 and s2 > 0:
        return (top1, top2), score

    return top1, score

def infer_filters_from_query(q: str):
    q0 = _norm(q)

    must, anyt = [], []

    # ===== NEW: nhận diện truy vấn liệt kê theo hoạt chất =====
    chems = extract_chemicals(q)
    pests = extract_pests(q)
    if chems:
        # các mẫu hỏi kiểu "hoạt chất X có trong sản phẩm nào / sản phẩm chứa X"
        if re.search(r"\b(hoat chat|chua|co trong)\b.*\b(san pham)\b", q0) or \
        re.search(r"\b(san pham)\b.*\b(chua|co)\b", q0):
            must.append("entity:product")
            for c in chems:
                must.append(f"chemical:{c}")
        else:
            # không chắc intent thì nới lỏng: đưa vào OR
            for c in chems:
                anyt.append(f"chemical:{c}")

    # ========= PEST =========
    if pests:
        # Các mẫu hỏi kiểu "trị/diệt/phòng/đặc trị X" hoặc "sâu/rầy/nhện/bọ... X"
        if re.search(r"\b(tri|diet|phong|dac tri|xu ly)\b", q0) or \
        re.search(r"\b(thuo[c|c]|san pham)\b.*\b(tri|diet|phong)\b", q0) or \
        re.search(r"\b(sau|ray|nhen|bo|rep|kien)\b", q0):
            # thường là đang hỏi sản phẩm cho đối tượng dịch hại
            must.append("entity:product")
            for p in pests:
                must.append(f"pest:{p}")
        else:
            # không chắc intent: nới lỏng
            for p in pests:
                anyt.append(f"pest:{p}")

    # crop ví dụ
    if "sau rieng" in q0:
        must.append("crop:sau-rieng")

    # entity type
    et, dbg = infer_entity_type(q)
    if isinstance(et, tuple):
        must.append(f"entity:{et[0]}")
        anyt.append(f"entity:{et[1]}")
    else:
        must.append(f"entity:{et}")

    # action/process (giữ như bạn đang làm)
    if re.search(r"\b(phun|xit)\b", q0): anyt.append("action:phun")
    if re.search(r"\bbon\b", q0): anyt.append("action:bon-phan")
    if re.search(r"\btuoi\b", q0): anyt.append("action:tuoi-goc")
    if re.search(r"\b(xu ly)\b", q0): anyt.append("process:xu-ly")

    # dedup giữ thứ tự
    def dedup(lst):
        seen, out = set(), []
        for t in lst:
            if t not in seen:
                out.append(t); seen.add(t)
        return out

    return dedup(must), dedup(anyt)
