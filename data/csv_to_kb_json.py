# csv_to_kb_json.py
# -*- coding: utf-8 -*-

import csv
import json
import re
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Helpers: parsing & normalize
# ----------------------------

def s(v: Any) -> str:
    return (v or "").strip()

def sl(v: Any) -> str:
    return s(v).lower()

def safe_json_loads_maybe_list(text: str) -> List[str]:
    """
    Accepts:
    - JSON array string: ["a","b"]
    - Python-ish string with doubled quotes: ["\"a\"", "\"b\""] (common in CSV exports)
    - Pipe-separated: a|b|c
    - Empty -> []
    """
    t = s(text)
    if not t:
        return []
    # If looks like JSON list
    if t.startswith("[") and t.endswith("]"):
        # Try normal json
        try:
            obj = json.loads(t)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
        # Try cleaning doubled quotes ("" -> ")
        try:
            t2 = t.replace('""', '"')
            obj = json.loads(t2)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
        # Fallback: split by comma inside brackets (best effort)
        inner = t[1:-1].strip()
        parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
        return [p for p in parts if p]
    # Pipe separated
    if "|" in t:
        parts = [p.strip() for p in t.split("|")]
        return [p for p in parts if p]
    return [t]

def parse_tags_pipe(tags_text: str) -> List[str]:
    # Accept "a|b|c" or JSON list
    tags = safe_json_loads_maybe_list(tags_text)
    # Normalize whitespace; keep original tokens as-is (you already use kebab-case)
    out = []
    for t in tags:
        t = s(t)
        if not t:
            continue
        out.append(t)
    # de-dup, preserve order
    seen = set()
    dedup = []
    for t in out:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return dedup

def parse_alt_questions(text: str) -> List[str]:
    alts = safe_json_loads_maybe_list(text)
    out = []
    for a in alts:
        a = s(a)
        if not a:
            continue
        # remove wrapping quotes if present
        if (a.startswith('"') and a.endswith('"')) or (a.startswith("'") and a.endswith("'")):
            a = a[1:-1].strip()
        if a:
            out.append(a)
    # de-dup
    seen = set()
    dedup = []
    for a in out:
        if a not in seen:
            seen.add(a)
            dedup.append(a)
    return dedup

def guess_title_from_question_or_name(row: Dict[str, str]) -> str:
    for k in ["title", "name", "product_name", "ten", "question"]:
        if k in row and s(row[k]):
            return s(row[k])
    return s(row.get("id", ""))

# ----------------------------
# Light struct extractors (optional, best-effort)
# ----------------------------

RE_PHI = re.compile(r"thời\s*gian\s*cách\s*ly\s*:\s*([0-9]{1,3})\s*ngày", re.IGNORECASE)
RE_FORMULATION = re.compile(r"\b(EC|SC|SL|WP|WG|WDG|GR|DF)\b", re.IGNORECASE)
RE_ACTIVE = re.compile(r"hoạt\s*chất\s*:\s*(.+?)(?:\.|nhóm|hãng|đặc\s*tính|đối\s*tượng|phạm\s*vi|liều\s*dùng|$)",
                       re.IGNORECASE | re.DOTALL)
RE_DOSAGE_TANK = re.compile(r"pha\s*([0-9]+(?:[.,][0-9]+)?)\s*-\s*([0-9]+(?:[.,][0-9]+)?)\s*ml\s*cho\s*bình\s*([0-9]+)\s*lít",
                            re.IGNORECASE)
RE_DOSAGE_HA = re.compile(r"liều\s*dùng.*?:\s*([0-9]+(?:[.,][0-9]+)?)\s*lít\s*/\s*ha", re.IGNORECASE)

def parse_product_struct_from_text(text: str) -> Dict[str, Any]:
    """
    Best-effort extraction from Vietnamese product descriptions.
    You can extend gradually. If it fails, it returns {}.
    """
    t = s(text)
    if not t:
        return {}

    product: Dict[str, Any] = {}

    # Formulation
    m = RE_FORMULATION.search(t)
    if m:
        product["formulation"] = m.group(1).upper()

    # Active ingredients: keep as raw string (you can later split by +)
    m = RE_ACTIVE.search(t)
    if m:
        active_raw = s(m.group(1))
        # collapse whitespace
        active_raw = re.sub(r"\s+", " ", active_raw)
        product["active_ingredients_raw"] = active_raw

    # PHI days
    m = RE_PHI.search(t)
    if m:
        try:
            product["phi_days"] = int(m.group(1))
        except Exception:
            pass

    # Dosage per ha
    m = RE_DOSAGE_HA.search(t)
    if m:
        product.setdefault("dosage", {})["per_ha_l"] = m.group(1).replace(",", ".")

    # Dosage per tank
    m = RE_DOSAGE_TANK.search(t)
    if m:
        lo = m.group(1).replace(",", ".")
        hi = m.group(2).replace(",", ".")
        tank_l = m.group(3)
        product.setdefault("dosage", {})["per_tank_ml_range"] = [lo, hi]
        product.setdefault("dosage", {})["tank_l"] = tank_l

    return product

# ----------------------------
# Core transform
# ----------------------------

def build_record(row: Dict[str, str], source_name: str) -> Dict[str, Any]:
    """
    Convert one CSV row to KB JSON record (base schema + optional structured fields).
    Expected common columns (flexible):
      - id
      - question
      - answer (or content)
      - entity_type
      - category (ignored unless you want to keep as legacy)
      - tags (pipe-separated or JSON list)
      - alt_questions (pipe-separated or JSON list)
      - entity_tags (JSON list) or extra_tags
    """
    rid = s(row.get("id") or row.get("ID") or row.get("doc_id") or row.get("uid") or "")
    if not rid:
        # deterministic fallback id
        rid = f"row_{abs(hash(json.dumps(row, ensure_ascii=False))) % 10**10}"

    entity_type = sl(row.get("entity_type") or row.get("entity") or row.get("type") or "general")
    question = s(row.get("question") or row.get("q") or row.get("cau_hoi") or "")
    answer = s(row.get("answer") or row.get("a") or row.get("tra_loi") or row.get("content") or "")

    tags = []
    tags += parse_tags_pipe(row.get("tags", ""))
    tags += parse_tags_pipe(row.get("entity_tags", ""))
    tags += parse_tags_pipe(row.get("extra_tags", ""))

    # de-dup tags
    seen = set()
    tags2 = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            tags2.append(t)
    tags = tags2

    alt_questions = parse_alt_questions(row.get("alt_questions", "") or row.get("alts", "") or row.get("alt", ""))

    title = guess_title_from_question_or_name(row)

    # Base schema (RAG-friendly)
    rec: Dict[str, Any] = {
        "id": rid,
        "entity_type": entity_type or "general",
        "title": title,
        "question": question,
        "answer": answer,
        "aliases": parse_alt_questions(row.get("aliases", "")),
        "tags": tags,
        "alt_questions": alt_questions,
        "source": {
            "type": "csv",
            "name": source_name,
        },
        "updated_at": datetime.now().strftime("%Y-%m-%d"),
    }

    # Build a single text field for embedding (you can customize)
    # Important: keep it compact; do NOT append huge tag lists here.
    parts = []
    if question:
        parts.append(f"Hỏi: {question}")
    if answer:
        parts.append(f"Đáp: {answer}")
    rec["content"] = "\n".join(parts).strip()

    # Optional structured extraction per entity_type
    if rec["entity_type"] == "product":
        rec["product"] = parse_product_struct_from_text(answer)

    # You can extend similarly:
    # if entity_type == "procedure": parse steps, etc.
    # if entity_type == "registry": parse registration_no, holder, etc.
    # if entity_type == "disease": parse symptoms/agent, etc.

    return rec

# ----------------------------
# IO
# ----------------------------

def read_csv_rows(csv_path: str, encoding: str = "utf-8-sig") -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
    return rows

def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def write_json(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Convert CSV KB to RAG-friendly JSON/JSONL")
    ap.add_argument("--input", "-i", required=True, help="Input CSV path")
    ap.add_argument("--out_jsonl", default="kb.jsonl", help="Output JSONL path")
    ap.add_argument("--out_json", default="kb.json", help="Output JSON path")
    ap.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default utf-8-sig for Excel)")
    ap.add_argument("--source", default="", help="Source name to store in records (default: filename)")
    args = ap.parse_args()

    source_name = args.source.strip() or args.input.split("\\")[-1].split("/")[-1]

    rows = read_csv_rows(args.input, encoding=args.encoding)
    records = [build_record(r, source_name=source_name) for r in rows]

    write_jsonl(args.out_jsonl, records)
    write_json(args.out_json, records)

    print(f"OK: {len(records)} records")
    print(f"- JSONL: {args.out_jsonl}")
    print(f"- JSON : {args.out_json}")

if __name__ == "__main__":
    main()
