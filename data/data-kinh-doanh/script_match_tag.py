import pandas as pd
import json
import re
import unicodedata
from collections import defaultdict

INPUT_CSV = "data-kd-1-4-chuan.csv"
OUT_CROP_CHEMICAL_INDEX = "crop_chemical_index.json"

# Debug theo product id (đặt "ohayo-100sc" để check đúng case anh nêu)
DEBUG_PRODUCT = "ohayo-100sc"
DEBUG_MAX_PRINT = 50

# Không dùng "bi" cho "bí"
CROP_SYNONYMS = {
    "che": ["chè", "trà"],
    "lua": ["lúa"],
    "dua": ["dưa", "dưa leo", "dưa hấu", "dưa lưới"],
    "bau-bi": ["bầu", "bí", "bầu bí"],
    "rau": ["rau", "rau các loại"],
    "hoa": ["hoa", "hoa các loại"],
}

# ============ UTILS ============
def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s:;/\-\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_tags_v2(raw):
    """
    Robust parse cho tags_v2:
    - raw có thể là JSON chuẩn
    - hoặc dạng CSV escaped: ["",""] với double quotes
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    # thử JSON thẳng
    try:
        x = json.loads(s)
        return x if isinstance(x, list) else []
    except Exception:
        pass
    # fix kiểu "" trong CSV: ["",""] -> ["",""]
    s2 = s.replace('""', '"')
    try:
        x = json.loads(s2)
        return x if isinstance(x, list) else []
    except Exception:
        return []

def extract_product_id(tags):
    for t in tags:
        if isinstance(t, str) and t.startswith("product:"):
            return t.split(":", 1)[1].strip().lower()
    return ""

def extract_chemicals(tags):
    out = []
    for t in tags:
        if isinstance(t, str) and t.startswith("chemical:"):
            out.append(t.split(":", 1)[1].strip().lower())
    # unique giữ thứ tự
    seen = set()
    uniq = []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

def pick_text_field(row):
    """
    Tự chọn cột có nội dung dài nhất trong các cột nghi ngờ.
    Ưu tiên answer; nếu answer rỗng thì fallback.
    """
    candidates = ["answer", "content", "text", "document", "kb_answer", "body"]
    best = ""
    best_col = ""
    for c in candidates:
        if c in row and pd.notna(row[c]):
            v = str(row[c])
            if len(v) > len(best):
                best = v
                best_col = c
    return best, best_col

def build_raw_regex(phrase: str):
    # match trên raw lower (có dấu), boundary mềm
    p = phrase.lower()
    return re.compile(rf"(?<!\w){re.escape(p)}(?!\w)")

# Pre-build regex per crop (match trên RAW có dấu, không normalize)
CROP_RX_RAW = {crop: [build_raw_regex(s) for s in syns] for crop, syns in CROP_SYNONYMS.items()}

def match_crop_in_text_raw(text_raw_lower: str, crop: str) -> bool:
    return any(rx.search(text_raw_lower) for rx in CROP_RX_RAW[crop])

def extract_target_snippet_raw(text_raw_lower: str) -> str:
    """
    Cắt đoạn liên quan "Đối tượng/Cây trồng" nếu có.
    Nếu không có thì trả về toàn text (để không bỏ ngoại lệ).
    """
    m = re.search(r"(đối tượng|cây trồng)\s*:\s*(.+?)(phạm vi|liều dùng|thời điểm|lưu ý|thời gian cách ly|$)",
                  text_raw_lower)
    if m:
        return m.group(2)
    return text_raw_lower

# ============ MAIN ============
df = pd.read_csv(INPUT_CSV)

crop_chemical_index = defaultdict(set)

debug_printed = 0
for _, row in df.iterrows():
    tags = parse_tags_v2(row.get("tags_v2"))
    if not tags:
        continue

    pid = extract_product_id(tags)
    chemicals = extract_chemicals(tags)
    if not chemicals:
        continue

    text_raw, used_col = pick_text_field(row)
    if not text_raw.strip():
        continue

    text_raw_lower = text_raw.lower()

    # Ưu tiên match trong snippet "Đối tượng/Cây trồng"
    target_snippet = extract_target_snippet_raw(text_raw_lower)

    for crop in CROP_SYNONYMS.keys():
        if match_crop_in_text_raw(target_snippet, crop):
            for chem in chemicals:
                crop_chemical_index[crop].add(chem)

            # debug case OHAYO
            if pid == DEBUG_PRODUCT and debug_printed < DEBUG_MAX_PRINT:
                debug_printed += 1
                print("---- DEBUG MATCH ----")
                print("product:", pid)
                print("used_col:", used_col)
                print("chemicals:", chemicals)
                print("crop matched:", crop)
                print("target_snippet (preview):", target_snippet[:300])

# SAVE
out = {k: sorted(list(v)) for k, v in crop_chemical_index.items() if v}
with open(OUT_CROP_CHEMICAL_INDEX, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("DONE")
print("Total crops:", len(out))

# Sanity check trực tiếp cho case anh nêu
if "bau-bi" in out:
    print("bau-bi count:", len(out["bau-bi"]))
    print("bau-bi sample:", out["bau-bi"][:30])
    print("bau-bi has chlorfenapyr:", "chlorfenapyr" in out["bau-bi"])
