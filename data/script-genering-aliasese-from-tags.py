# =========================================================
# Batch Alias Renderer from .txt (READY TO RUN)
# =========================================================

import re
import json
import unicodedata
from typing import List
from collections import defaultdict
from openai import OpenAI

# ================== CONFIG ==================

MODEL = "gpt-4.1-mini"
INPUT_TXT = "unique_tags_v2.txt"
OUTPUT_PY = "aliases_generated.py"
BATCH_SIZE = 40        # an toàn cho 1900 dòng
TEMPERATURE = 0.3

# ================== UTILS ==================

def normalize_vi(s: str) -> str:
    s = s.lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def load_txt(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            out.append(line)
    return out

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# ================== GPT ==================

SYSTEM_PROMPT = """Bạn là chuyên gia nông nghiệp Việt Nam.
Nhiệm vụ:
- Nhận vào danh sách tag dạng namespace:value (vd pest:thrips, pest:thoi-bap)
- Sinh alias mà người dùng Việt Nam hay gõ khi hỏi.

Quy tắc bắt buộc:
- Alias PHẢI là tiếng Việt KHÔNG DẤU
- Nếu value là tiếng Anh → dịch đúng sang tiếng Việt chuyên ngành
- Không sinh alias quá chung chung (vd: sau, bo, benh)
- Không bịa đối tượng không liên quan
- Trả về JSON đúng schema, không giải thích ngoài JSON
"""

def call_gpt_batch(client: OpenAI, tags: List[str]):
    payload = {
        "tags": tags,
        "examples": {
            "pest:thrips": ["bo tri", "tri tren la", "tri tren dua non"],
            "pest:stem-borer": ["sau duc than"],
            "pest:thoi-bap": ["bap bi thoi", "benh thoi bap"]
        },
        "schema": {
            "namespace": "string",
            "canonical": "string",
            "aliases": ["string"]
        }
    }

    resp = client.responses.create(
        model=MODEL,
        temperature=TEMPERATURE,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]
    )

    return json.loads(resp.output_text)

# ================== MAIN ==================

def main():
    print("▶ Loading input file...")
    tags = load_txt(INPUT_TXT)
    total = len(tags)

    if total == 0:
        print("❌ Input file is empty.")
        return

    print(f"▶ Loaded {total} tags")

    client = OpenAI(api_key="...")
    grouped = defaultdict(lambda: defaultdict(list))

    batch_no = 0
    for batch in chunked(tags, BATCH_SIZE):
        batch_no += 1
        print(f"▶ Processing batch {batch_no} ({len(batch)} items)")

        try:
            results = call_gpt_batch(client, batch)
        except Exception as e:
            print(f"❌ GPT error at batch {batch_no}: {e}")
            continue

        for item in results:
            ns = item.get("namespace")
            key = item.get("canonical")
            aliases = item.get("aliases", [])

            if not ns or not key:
                continue

            seen = set()
            for a in aliases:
                a2 = normalize_vi(a)
                if not a2 or a2 in seen:
                    continue
                grouped[ns][key].append(a2)
                seen.add(a2)

    # ================== WRITE OUTPUT ==================

    print("▶ Writing output file...")
    with open(OUTPUT_PY, "w", encoding="utf-8") as f:
        f.write("# =========================================\n")
        f.write("# AUTO-GENERATED ALIASES – REVIEW BEFORE USE\n")
        f.write("# =========================================\n\n")

        for ns, data in grouped.items():
            const_name = ns.upper() + "_ALIASES"
            f.write(f"{const_name} = {{\n")
            for k, vals in sorted(data.items()):
                f.write(f'    "{k}": {json.dumps(sorted(set(vals)), ensure_ascii=False)},\n')
            f.write("}\n\n")

    print(f"✅ DONE. Output saved to {OUTPUT_PY}")

if __name__ == "__main__":
    main()
