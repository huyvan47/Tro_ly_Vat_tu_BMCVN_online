import csv
import re
from typing import List
from pathlib import Path

# ==========================
# CONFIG
# ==========================
ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "data-vat-tu/data-vat-tu-full-fixed-output_with_img.csv"          # file CSV gốc
OUTPUT_CSV = "output_chunk_overlap.csv"   # file CSV sau khi xử lý

# HARD CHUNK PARAMS
LONG_TEXT_THRESHOLD = 2200   # nếu answer > ngưỡng này thì chunk
CHUNK_SIZE = 1800
OVERLAP_SIZE = 260
SOFT_CUT_WINDOW = 220       # tìm điểm cắt đẹp

# =====================
# HELPERS
# =====================
def normalize_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def find_cut_point(text: str, target_end: int, window: int) -> int:
    if target_end >= len(text):
        return len(text)

    start = max(0, target_end - window)
    segment = text[start:target_end]

    patterns = [r"\n\n", r"\n", r"[\.!\?;:]", r","]
    candidates = []

    for p in patterns:
        for m in re.finditer(p, segment):
            candidates.append(start + m.start())

    if not candidates:
        return target_end

    cut = max(candidates)
    if cut < target_end - window * 0.9:
        return target_end

    return cut

def hard_chunk(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return [""]

    chunks = []
    i = 0
    n = len(text)
    step = max(1, CHUNK_SIZE - OVERLAP_SIZE)

    while i < n:
        raw_end = min(n, i + CHUNK_SIZE)
        end = find_cut_point(text, raw_end, SOFT_CUT_WINDOW) if raw_end < n else raw_end
        if end <= i:
            end = raw_end

        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        i = max(0, end - OVERLAP_SIZE)

    return chunks

# =====================
# MAIN
# =====================
def process_csv():
    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    out_rows = []

    for r in rows:
        base_id = (r.get("id") or "").strip()
        answer = normalize_text(r.get("answer") or "")

        if len(answer) > LONG_TEXT_THRESHOLD:
            chunks = hard_chunk(answer)
        else:
            chunks = [answer]

        for idx, ch in enumerate(chunks):
            new_id = f"{base_id}_chunk_{idx:02d}" if base_id else f"row_chunk_{idx:02d}"

            out_rows.append({
                "id": new_id,
                "question": r.get("question", ""),
                "answer": ch,
                "category": r.get("category", ""),
                "tags": r.get("tags", ""),
                "alt_questions": r.get("alt_questions", ""),
                "img_keys": r.get("img_keys", ""),
            })

    fieldnames = ["id", "question", "answer", "category", "tags", "alt_questions","img_keys"]

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"✅ Done. Wrote {len(out_rows)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_csv()
