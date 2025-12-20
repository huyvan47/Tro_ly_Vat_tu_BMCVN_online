# -*- coding: utf-8 -*-
"""
entity_type_classifier.py

Deterministic (no-LLM) entity-type classification for your knowledge CSV.

Input CSV columns (expected):
  id, question, answer, category, tags, alt_questions, img_keys

Output:
  - Writes a new CSV with an added column: entity_type
  - Prints a small summary (counts and percentages)

Usage (Windows PowerShell):
  python .\entity_type_classifier.py `
    --input .\data\data-kinh-doanh\data-kd-full-1-4.csv `
    --output .\data\data-kinh-doanh\data-kd-full-1-4_with_entity_type.csv

Notes:
- This is intentionally rule-based and deterministic for stability and auditability.
- Tune keyword lists below to your domain vocabulary if needed.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


# -----------------------------
# Normalization helpers
# -----------------------------
def strip_accents_vi(text: str) -> str:
    """Lowercase + remove Vietnamese accents for robust matching."""
    if text is None:
        return ""
    s = str(text).lower().strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("đ", "d")
    return s


def norm_space(text: str) -> str:
    s = strip_accents_vi(text)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def contains_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


# -----------------------------
# Entity-type schema
# -----------------------------
ENTITY_TYPES = [
    "registry",   # danh mục/đăng ký lưu hành (công ty đăng ký, hoạt chất, tên thương phẩm, đối tượng/cây trồng)
    "product",    # sản phẩm / thuốc
    "procedure",  # quy trình / hướng dẫn thao tác kỹ thuật
    "skill",      # kỹ năng / nghiệp vụ kinh doanh (bài học dài)
    "pest",       # sâu hại / côn trùng / nhện
    "weed",       # cỏ dại
    "disease",    # bệnh cây / triệu chứng
    "general",    # còn lại
]


# -----------------------------
# Category mapping (strong signal)
# Keys are compared against normalized category string.
# Tune these keys to match your category conventions.
# -----------------------------
CATEGORY_MAP: Dict[str, str] = {
    "danh_muc_thuoc_bao_ve_thuc_vat": "registry",
    "danh muc thuoc bao ve thuc vat": "registry",
    "danh_muc_bvtv": "registry",
    "dang_ky_luu_hanh": "registry",
    "dang ky luu hanh": "registry",

    "san_pham": "product",
    "thuoc": "product",
    "sp": "product",

    "quy_trinh": "procedure",
    "quy trinh": "procedure",
    "huong_dan": "procedure",
    "huong dan": "procedure",

    "ky_nang": "skill",
    "kinh_doanh": "skill",
    "kinh doanh": "skill",
    "ban_hang": "skill",
    "ban hang": "skill",

    "sau_hai": "pest",
    "sau hai": "pest",
    "con_trung": "pest",
    "con trung": "pest",

    "co": "weed",

    "benh": "disease",
    "nam_benh": "disease",
    "nam benh": "disease",
}


# -----------------------------
# Keyword rules (medium signal)
# All lists MUST be accent-stripped + lowercase for matching.
# -----------------------------
PRODUCT_KW = [strip_accents_vi(x) for x in [
    "thuốc", "che pham", "đặc trị", "dac tri", "hoạt chất", "hoat chat",
    "tên thương phẩm", "ten thuong pham",
]]
FORMULATION_RE = re.compile(r"(?<![a-z0-9])(ec|sc|wp|sl|wg|wdg|sp|cs|ew|od|fs|gr|df)(?![a-z0-9])")

REGISTRY_KW = [strip_accents_vi(x) for x in [
    "đơn vị đăng ký", "don vi dang ky",
    "đăng ký", "dang ky",
    "lưu hành", "luu hanh",
    "được phép", "duoc phep",
    "danh mục thuốc bảo vệ thực vật", "danh muc thuoc bao ve thuc vat",
    "thuốc sử dụng trong nông nghiệp", "thuoc su dung trong nong nghiep",
    "rank:", "tt:", "group:", "num:",
]]
# Các dấu hiệu dạng dòng dữ liệu đăng ký lưu hành (registry)
REGISTRY_RE = re.compile(r"(don\s*vi\s*dang\s*ky|\bdang\s*ky\b|\bluu\s*hanh\b|\bthuoc\s*su\s*dung\s*trong\s*nong\s*nghiep\b|\brank\s*:|\btt\s*:|\bgroup\s*:|\bnum\s*:)", re.IGNORECASE)

PROCEDURE_KW = [strip_accents_vi(x) for x in [
    "quy trinh", "cac buoc", "huong dan", "quy dinh", "thao tac", "thuc hien",
    "kiem soat", "kiem tra", "bieu mau", "quy che",
]]

SKILL_KW = [strip_accents_vi(x) for x in [
    "gia ban", "chinh sach", "mo moi", "tang doanh so", "y kien kh", "khach hang",
    "cong no", "thi truong", "ke hoach", "muc tieu", "giao viec", "thanh toan", "thanh quyet toan",
    "tu van", "nhan rieng",
]]

PEST_KW = [strip_accents_vi(x) for x in [
    "sau", "bo", "ray", "nhen", "con trung", "chich hut", "doi duc la",
    "bo tri", "bo phan", "ray mem", "ray xanh",
]]

WEED_KW = [strip_accents_vi(x) for x in [
    "co", "co dai", "co long vuc", "co duoi phung", "co gau", "co tranh",
]]

DISEASE_KW = [strip_accents_vi(x) for x in [
    "benh", "trieu chung", "dau hieu", "nam", "vi khuan", "virus",
    "than thu", "xi mu", "thoi re", "chay la", "ghe", "dom la",
]]


# -----------------------------
# Length-based fallback (weak signal but powerful on long training docs)
# -----------------------------
SKILL_WORDS_THRESHOLD = 300  # tune if needed


def detect_entity_type(row: dict) -> str:
    """Deterministic classifier: category mapping -> keyword rules -> length fallback."""
    q = norm_space(row.get("question", ""))
    a = norm_space(row.get("answer", ""))
    cat = norm_space(row.get("category", ""))

    text = f"{q} {a}".strip()

    # 1) Category mapping
    for k, v in CATEGORY_MAP.items():
        if k in cat:
            return v

    # 2) Keyword rules
    # 2) Registry rules (đăng ký lưu hành) - ưu tiên trước product
    rid = norm_space(row.get("id", ""))
    if rid.startswith("danh_muc_thuoc_bao_ve_thuc_vat") or rid.startswith("danh-muc-thuoc-bao-ve-thuc-vat"):
        return "registry"
    # Nếu có dấu hiệu 'đơn vị đăng ký' + các trường cấu trúc thì coi là registry
    if REGISTRY_RE.search(text) or contains_any(text, REGISTRY_KW):
        # Giảm false-positive: registry thường có "Tên thương phẩm" + "Hoạt chất" + "Đơn vị đăng ký"
        if ("ten thuong pham" in text or "tên thương phẩm" in (q + " " + a).lower()) and ("hoat chat" in text or "hoạt chất" in (q + " " + a).lower()):
            return "registry"

    if FORMULATION_RE.search(text) or contains_any(text, PRODUCT_KW):
        return "product"

    if contains_any(text, PROCEDURE_KW):
        return "procedure"

    if contains_any(text, SKILL_KW):
        return "skill"

    # disease/pest/weed: check disease first, then pest, then weed ("co" is broad)
    if contains_any(text, DISEASE_KW):
        return "disease"

    if contains_any(text, PEST_KW):
        return "pest"

    if contains_any(text, WEED_KW):
        return "weed"

    # 3) Length fallback
    if len(a.split()) >= SKILL_WORDS_THRESHOLD and not (FORMULATION_RE.search(text) or contains_any(text, PRODUCT_KW)):
        if contains_any(text, PROCEDURE_KW):
            return "procedure"
        return "skill"

    return "general"


def summarize_counts(counts: Counter, total: int) -> str:
    lines = []
    for et in ENTITY_TYPES:
        n = counts.get(et, 0)
        pct = (n / total * 100.0) if total else 0.0
        lines.append(f"- {et:9s}: {n:5d}  ({pct:5.1f}%)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default: utf-8-sig)")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N rows (debug)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path.resolve()}", file=sys.stderr)
        return 2

    df = pd.read_csv(in_path, dtype=str, keep_default_na=False, encoding=args.encoding)

    expected = {"id", "question", "answer", "category", "tags", "alt_questions", "img_keys"}
    missing = expected - set(df.columns)
    if missing:
        print(f"[WARN] Missing expected columns: {sorted(missing)}", file=sys.stderr)

    n_total = len(df) if args.limit is None else min(len(df), args.limit)

    entity_types: List[str] = []
    for i in range(n_total):
        row = df.iloc[i].to_dict()
        entity_types.append(detect_entity_type(row))

    # Write back
    if args.limit is not None:
        df2 = df.iloc[:n_total].copy()
        df2["entity_type"] = entity_types
    else:
        df2 = df.copy()
        df2["entity_type"] = entity_types

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(out_path, index=False, encoding="utf-8-sig")

    counts = Counter(entity_types)
    print("Entity-type distribution")
    print(summarize_counts(counts, n_total))
    print(f"\nOutput: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
