# -*- coding: utf-8 -*-
"""
Enrich CSV with tags using OpenAI (Responses API) + strict prompt.

Requirements:
  pip install --upgrade openai pandas

Env:
  setx OPENAI_API_KEY "YOUR_KEY"   (Windows)
  export OPENAI_API_KEY="YOUR_KEY" (macOS/Linux)

Usage:
  python enrich_tags.py --input in.csv --output out.csv --model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import os
import re
import time
import unicodedata
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from openai import OpenAI


# -----------------------------
# 1) Prompt (THEO ĐÚNG BẢN BẠN ĐƯA)
# -----------------------------
PROMPT_TEMPLATE = """NHIỆM VỤ:
Sinh tags cho 1 sản phẩm từ dữ liệu cung cấp.

CHỈ DÙNG THÔNG TIN CÓ TRONG DỮ LIỆU. KHÔNG SUY ĐOÁN.
Nếu không thấy thông tin cho key nào thì bỏ key đó.

ĐỊNH DẠNG:
- Trả về DUY NHẤT 1 dòng.
- Tags phân cách bằng ký tự "|".
- Mỗi tag có dạng "key:value".
- value: tiếng Việt không dấu, chữ thường, dùng dấu gạch nối.

KEY ĐƯỢC PHÉP:
- crop (có thể nhiều tag crop:*)
- pest (có thể nhiều tag pest:*)
- crop-group (tùy chọn, nếu xác định được từ danh mục nhóm)
- pest-group (tùy chọn, nếu xác định được từ danh mục nhóm)
- product-group
- active
- brand

QUY TẮC ĐA TRỊ:
- Nếu có nhiều cây trồng: tạo nhiều tag crop:* (không gộp vào 1 value).
- Nếu có nhiều sâu hại: tạo nhiều tag pest:*.
- Loại bỏ trùng lặp.
- Tối đa 25 tags. Nếu nhiều hơn, ưu tiên crop/pest cụ thể trước, rồi group.

DỮ LIỆU:
{BLOCK_DU_LIEU}
"""


# -----------------------------
# 2) Tag rules & utilities
# -----------------------------
ALLOWED_KEYS: Set[str] = {
    "crop",
    "pest",
    "crop-group",
    "pest-group",
    "product-group",
    "active",
    "brand",
}

# Ưu tiên khi vượt quá 25 tags: crop/pest trước, rồi group, rồi các key khác
KEY_PRIORITY: Dict[str, int] = {
    "crop": 0,
    "pest": 1,
    "crop-group": 2,
    "pest-group": 3,
    "product-group": 4,
    "active": 5,
    "brand": 6,
}


def slugify_vi(text: str) -> str:
    """
    Chuẩn hoá tiếng Việt -> không dấu, lowercase, dấu gạch nối.
    """
    if text is None:
        return ""
    s = str(text).strip().lower()
    if not s:
        return ""
    # remove accents
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("đ", "d")
    # keep letters/numbers/space/hyphen
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    return s


def normalize_tag(tag: str) -> Optional[str]:
    """
    Validate + normalize a single tag "key:value" -> "key:slug".
    """
    if not tag:
        return None
    tag = tag.strip()
    if ":" not in tag:
        return None

    key, val = tag.split(":", 1)
    key = key.strip()
    val = val.strip()

    if key not in ALLOWED_KEYS:
        return None

    val_slug = slugify_vi(val)
    if not val_slug:
        return None

    return f"{key}:{val_slug}"


def parse_and_clean_tags(line: str, max_tags: int = 25) -> str:
    """
    Parse raw model output -> cleaned tags string.
    - split by |
    - normalize
    - dedupe
    - enforce priority & cap
    """
    if not line:
        return ""

    # Lấy đúng 1 dòng (phòng model trả nhiều dòng)
    line = line.strip().splitlines()[0].strip()

    raw_tags = [t.strip() for t in line.split("|") if t.strip()]
    normed: List[str] = []
    seen: Set[str] = set()

    for t in raw_tags:
        nt = normalize_tag(t)
        if nt and nt not in seen:
            normed.append(nt)
            seen.add(nt)

    # sort by priority then lexicographic for stability
    normed.sort(key=lambda x: (KEY_PRIORITY.get(x.split(":", 1)[0], 999), x))

    # cap
    normed = normed[:max_tags]
    return "|".join(normed)


def build_block_du_lieu(row: pd.Series, field_map: Dict[str, str]) -> str:
    # field_map có thể map "Câu hỏi"->question, "Nội dung"->answer, "Danh mục"->category
    lines = []
    for label, col in field_map.items():
        val = row.get(col, "")
        if pd.isna(val):
            val = ""
        val = str(val).strip()
        if val:
            lines.append(f"{label}: {val}")
    return "\n".join(lines).strip()



# -----------------------------
# 3) OpenAI call (Responses API)
# -----------------------------
def llm_generate_tags(
    client: OpenAI,
    model: str,
    block_du_lieu: str,
    temperature: float = 0.0,
    max_output_tokens: int = 120,
) -> str:
    prompt = PROMPT_TEMPLATE.replace("{BLOCK_DU_LIEU}", block_du_lieu)

    # Responses API (recommended for new builds)
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    # output_text is the simplest accessor for text output
    return (resp.output_text or "").strip()


# -----------------------------
# 4) Main enrichment
# -----------------------------
def enrich_csv_tags(
    input_path: str,
    output_path: str,
    model: str,
    field_map: Dict[str, str],
    tags_col: str = "tags",
    id_col: Optional[str] = "id",
    overwrite: bool = False,
    sleep_s: float = 0.0,
    limit: Optional[int] = None,
) -> None:
    
    client = OpenAI(api_key="...")

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    if tags_col not in df.columns:
        df[tags_col] = ""



    total = len(df) if limit is None else min(len(df), limit)
    updated = 0

    for i in range(total):
        row = df.iloc[i]

        existing = str(row.get(tags_col, "")).strip()
        if existing and not overwrite:
            continue

        block = build_block_du_lieu(row, field_map)
        if not block:
            # Không có dữ liệu để sinh tag
            continue

        try:
            raw = llm_generate_tags(client, model=model, block_du_lieu=block)
            cleaned = parse_and_clean_tags(raw, max_tags=25)

            df.at[df.index[i], tags_col] = cleaned
            updated += 1

            # log nhẹ theo id nếu có
            rid = row.get(id_col, "") if id_col and id_col in df.columns else ""
            print(f"[{i+1}/{total}] updated id={rid} tags={cleaned}")

        except Exception as e:
            rid = row.get(id_col, "") if id_col and id_col in df.columns else ""
            print(f"[{i+1}/{total}] ERROR id={rid}: {e}")

        if sleep_s > 0:
            time.sleep(sleep_s)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Done. Updated rows: {updated}. Output: {output_path}")


# -----------------------------
# 5) CLI
# -----------------------------
def parse_kv_map(pairs: List[str]) -> Dict[str, str]:
    """
    Accepts list like:
      ["Tên thương phẩm=product_name", "Hoạt chất=active", "Nhóm=group", "Cây trồng=crops", "Đối tượng=sau_hai"]
    Returns mapping label -> column_name
    """
    out: Dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Invalid field mapping '{p}'. Use Label=column_name")
        label, col = p.split("=", 1)
        out[label.strip()] = col.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--model", default="gpt-4.1-mini", help="Model name")
    ap.add_argument("--tags-col", default="tags", help="Tags column name")
    ap.add_argument("--id-col", default="id", help="ID column name (optional)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing tags")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep between requests (seconds)")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N rows")
    ap.add_argument(
        "--map",
        nargs="+",
        required=True,
        help="Field mapping, e.g. --map 'Tên thương phẩm=product' 'Hoạt chất=active' 'Nhóm=group' 'Cây trồng=crops' 'Sâu hại=pests'",
    )
    args = ap.parse_args()

    field_map = parse_kv_map(args.map)

    enrich_csv_tags(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        field_map=field_map,
        tags_col=args.tags_col,
        id_col=args.id_col if args.id_col else None,
        overwrite=args.overwrite,
        sleep_s=args.sleep,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
