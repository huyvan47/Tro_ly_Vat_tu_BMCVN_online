import pandas as pd
import json
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any, Union, Optional

INPUT_CSV = "data-kd-1-4-chuan copy.csv"
OUTPUT_JSON = "chemical_tags_from_kb.json"

# -----------------------------
# 1) NORMALIZE
# -----------------------------
LABEL_STOP_RX = re.compile(
    r"\b("
    r"sau hai\s*:|benh hai\s*:|co hai\s*:|pham vi\s*:|pham vi su dung\s*:|"
    r"lieu dung\s*:|thoi diem\s*:|hieu luc\s*:|luu y\s*:|thoi gian\s*:|"
    r"truong hop\s*:|kinh nghiem\s*:|pham vi ap dung\s*:"
    r")",
    re.IGNORECASE
)

def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # GIỮ . và :
    text = re.sub(r"[^a-z0-9\s\.\:;/\-,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_block_raw(answer_raw: str, markers: List[str]) -> str:
    """
    Trích nội dung sau marker:
    - ưu tiên cắt đến dấu chấm '.' đầu tiên
    - nếu trong đoạn có label khác (Sâu hại/Bệnh hại/...) thì cắt trước label đó
    """
    if not answer_raw:
        return ""

    # làm "mềm" xuống dòng để find ổn định
    s = " ".join(str(answer_raw).split())
    s_low = s.lower()

    for m in markers:
        m_low = m.lower()
        idx = s_low.find(m_low)
        if idx == -1:
            continue

        part = s[idx + len(m):]  # lấy raw (không lower) để giữ dấu chấm

        # 1) cắt trước label kế tiếp nếu có
        m2 = LABEL_STOP_RX.search(part)
        if m2:
            part = part[:m2.start()]

        # 2) cắt tới dấu chấm đầu tiên (nếu còn)
        dot = part.find(".")
        if dot != -1:
            part = part[:dot]

        return part.strip()

    return ""

# -----------------------------
# 2) SAFE PARSE tags_v2 (CSV hay bị ""..."" )
# -----------------------------
def safe_json_loads_maybe(s: str):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip()
    if not s:
        return None

    # case: đã là JSON chuẩn
    try:
        return json.loads(s)
    except Exception:
        pass

    # case: CSV/Excel escape kiểu ["\"a\"", "\"b\""] hoặc ["“”"] -> thay "" -> "
    s2 = s.replace('""', '"')

    # đôi khi cả chuỗi bị bọc ngoài bằng dấu "
    if len(s2) >= 2 and s2[0] == '"' and s2[-1] == '"':
        s2 = s2[1:-1]

    try:
        return json.loads(s2)
    except Exception:
        return None

def extract_chemicals(tags_v2_raw):
    tags = safe_json_loads_maybe(tags_v2_raw)
    if not isinstance(tags, list):
        return []
    out = []
    for t in tags:
        if isinstance(t, str) and t.startswith("chemical:"):
            out.append(t.split(":", 1)[1])
    return out

# -----------------------------
# 3) EXTRACT BLOCKS
# -----------------------------
STOP_MARKERS_RX = re.compile(
    r"(pham vi|lieu dung|thoi diem|hieu luc|luu y|thoi gian|truong hop|kinh nghiem|pham vi ap dung|$)"
)

def extract_block(answer_norm: str, markers):
    """
    Lấy text sau marker cho đến dấu chấm đầu tiên.
    """
    for m in markers:
        m_norm = normalize(m)
        idx = answer_norm.find(m_norm)
        if idx != -1:
            part = answer_norm[idx + len(m_norm):]

            # CẮT ĐẾN DẤU CHẤM ĐẦU TIÊN
            dot_pos = part.find(".")
            if dot_pos != -1:
                part = part[:dot_pos]

            return part.strip()

    return ""

def parse_items(text: str) -> List[str]:
    if not text:
        return []

    items = re.split(r"[;,]", text)
    out = []
    for it in items:
        it = it.strip(" -•\t ").strip()
        it = normalize(it)  # normalize TỪNG ITEM

        # lọc các “câu” dính liều dùng / lưu ý
        if any(k in it for k in ["lieu dung", "thoi diem", "hieu luc", "luu y", "thoi gian", "pham vi"]):
            continue

        if it:
            out.append(it)

    # dedup giữ thứ tự
    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


# -----------------------------
# 4) READ CSV (encoding fallback)
# -----------------------------
def read_csv_robust(path: str) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

df = read_csv_robust(INPUT_CSV)

# sanity check columns
required_cols = {"answer", "tags_v2"}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}. Found columns: {list(df.columns)}")

# -----------------------------
# 5) BUILD KB
# -----------------------------
kb = defaultdict(lambda: {
    "crops": set(),
    "pests": set(),
    "diseases": set(),
    "weeds": set(),
    "contexts": []
})

bad_tags_rows = 0
rows_with_chem = 0

for _, row in df.iterrows():
    chemicals = extract_chemicals(row.get("tags_v2"))
    if not chemicals:
        continue

    answer_raw = row.get("answer", "")

    crop_text = extract_block_raw(answer_raw, ["Đối tượng: Cây trồng:", "Cây trồng:"])
    pest_text = extract_block_raw(answer_raw, ["Sâu hại:"])
    disease_text = extract_block_raw(answer_raw, ["Bệnh hại:"])
    weed_text = extract_block_raw(answer_raw, ["Cỏ hại:"])

    crops = parse_items(crop_text)
    pests = parse_items(pest_text)
    diseases = parse_items(disease_text)
    weeds = parse_items(weed_text)

    for chem in chemicals:
        for c in crops:
            kb[chem]["crops"].add(c)

        for p in pests:
            kb[chem]["pests"].add(p)

        for d in diseases:
            kb[chem]["diseases"].add(d)

        for w in weeds:
            kb[chem]["weeds"].add(w)


output = {
    chem: {
        "crops": sorted(list(data["crops"])),
        "pests": sorted(list(data["pests"])),
        "diseases": sorted(list(data["diseases"])),
        "weeds": sorted(list(data["weeds"])),
    }
    for chem, data in kb.items()
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("DONE")
print("Total chemicals extracted:", len(output))
