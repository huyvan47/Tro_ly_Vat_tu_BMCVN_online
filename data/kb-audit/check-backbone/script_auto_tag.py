#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import urllib.request
import urllib.error


# ======================
# FORCE TAG BY ENTITY_TYPE (HARD TAG)
# ======================

ENTITY_TYPE_FORCE_TAG = {
    "registry":  "entity:registry",
    "product":   "entity:product",
    "disease":   "entity:disease",
    "procedure": "entity:procedure",
    "pest":      "entity:pest",
    "weed":      "entity:weed",
    "general":   "entity:general",
}

# ======================
# Config: Namespaces & Maps
# ======================

ALLOWED_NAMESPACES = [
    "domain",
    "entity",
    "pest",
    "disease",
    "crop",
    "chemical",
    "action",
    "process",
    "object",
]

# Normalize variations GPT might emit
NAMESPACE_ALIAS = {
    "thuoc": "chemical",
    "active": "chemical",
    "hoatchat": "chemical",
    "hoat-chat": "chemical",
    "hoat_chat": "chemical",
    "sanpham": "chemical",
    "sp": "chemical",
    "cay": "crop",
    "caytrong": "crop",
    "doi-tuong": "entity",
    "doituong": "entity",
    "sau": "pest",
    "con-trung": "pest",
    "nam": "disease",
    "benh": "disease",
    "quy-trinh": "process",
    "quytrinh": "process",
    "huong-dan": "process",
    "huongdan": "process",
}

# Map “slug-only” or variants to canonical tags (extend for your taxonomy)
TAG_MAP: Dict[str, str] = {
    # pests
    "ray-nau": "pest:ray-nau",
    "ray-lung-trang": "pest:ray-lung-trang",
    "bo-phan": "pest:bo-phan",
    "rep-sap": "pest:rep-sap",
    "nhen-do": "pest:nhen-do",
    "tri": "pest:tri",  # trĩ (pest)
    "oc-buu-vang": "pest:oc-buu-vang",

    # crops
    "lua": "crop:lua",
    "cam": "crop:cam",
    "quyt": "crop:quyt",
    "xoai": "crop:xoai",
    "sau-rieng": "crop:sau-rieng",

    # actions/process
    "phong-tru": "action:phong-tru",
    "pha-thuoc": "process:pha-thuoc",
    "phun": "action:phun",
    "tuoi-goc": "action:tuoi-goc",
    "bon-phan": "action:bon-phan",
    "xu-ly": "process:xu-ly",

    # domain/entity examples
    "danh-muc-thuoc": "domain:danh-muc-thuoc",
    "quy-trinh": "domain:quy-trinh",
}


def force_tags_by_entity_type(entity_type: str) -> List[str]:
    if not entity_type:
        return []
    et = str(entity_type).strip().lower()
    tag = ENTITY_TYPE_FORCE_TAG.get(et)
    return [tag] if tag else []


# ======================
# Utilities
# ======================

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(
        r"[^a-z0-9\-\:\_àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
        "-",
        s
    )
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_json_array(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict) and "tags" in obj and isinstance(obj["tags"], list):
            return [str(x) for x in obj["tags"]]
    except Exception:
        pass

    parts = re.split(r"[\n,;]+", s)
    return [p.strip() for p in parts if p.strip()]


def normalize_one_tag(tag: str) -> Optional[str]:
    if not tag:
        return None

    t = str(tag).strip()
    t = t.replace("—", "-").replace("–", "-")
    t = t.strip().strip('"').strip("'")
    t = slugify(t)
    if not t:
        return None

    # Direct map: slug-only or full
    if t in TAG_MAP:
        return TAG_MAP[t]

    # Already namespace:slug
    if ":" in t:
        ns, slug = t.split(":", 1)
        ns = NAMESPACE_ALIAS.get(ns, ns)
        slug = slugify(slug)
        if ns in ALLOWED_NAMESPACES and slug:
            key = f"{ns}:{slug}"
            return TAG_MAP.get(key, key)
        return None

    # Heuristic infer namespace (safe, minimal)
    inferred_ns = None
    if re.search(r"(mectin|zole|thrin|pyr|cide|azole)$", t):
        inferred_ns = "chemical"
    if t in {"ray-nau", "bo-phan", "rep-sap", "nhen-do", "tri", "oc-buu-vang"}:
        inferred_ns = "pest"
    if t in {"lua", "cam", "quyt", "xoai", "sau-rieng"}:
        inferred_ns = "crop"
    if t in {"pha-thuoc", "xu-ly"}:
        inferred_ns = "process"
    if t in {"phun", "tuoi-goc", "phong-tru", "bon-phan"}:
        inferred_ns = "action"

    if inferred_ns:
        key = f"{inferred_ns}:{t}"
        return TAG_MAP.get(key, key)

    # Cannot infer -> drop
    return None


def normalize_tags(raw_tags: List[str], max_tags: int = 6) -> List[str]:
    out: List[str] = []
    seen = set()
    for rt in raw_tags:
        nt = normalize_one_tag(rt)
        if not nt:
            continue
        if nt not in seen:
            out.append(nt)
            seen.add(nt)
        if len(out) >= max_tags:
            break
    return out


def ns_of(tag: str) -> str:
    return tag.split(":", 1)[0] if ":" in tag else ""


def enforce_entity_rules(tags: List[str], entity_type: str) -> List[str]:
    """
    Post-rule to prevent "procedure" tags from becoming a product list:
    - If entity_type == procedure:
      * chemical:* <= 2
      * At least 2 tags from {process, action, object} IF already present (we don't invent)
    """
    et = (entity_type or "").strip().lower()
    if et != "procedure":
        return tags

    process_like = [t for t in tags if ns_of(t) in {"process", "action", "object"}]
    chemicals = [t for t in tags if ns_of(t) == "chemical"]
    others = [t for t in tags if ns_of(t) not in {"process", "action", "object", "chemical"}]

    chemicals = chemicals[:2]

    rebuilt: List[str] = []
    seen = set()

    # Prefer: entity/crop + process/action/object + other + limited chemicals
    for t in process_like + others + chemicals:
        if t and t not in seen:
            rebuilt.append(t)
            seen.add(t)

    return rebuilt


# ======================
# OpenAI call (via HTTPS)
# ======================

@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-4.1-mini"
    timeout_sec: int = 60
    max_retries: int = 3
    base_url: str = "https://api.openai.com/v1/responses"


def call_openai_tags(cfg: OpenAIConfig, content: str, max_tags: int, entity_type: str = "") -> List[str]:
    system = (
        "You are a strict tagging engine for a RAG system. "
        "Generate compact, canonical tags used for metadata filtering and reranking. "
        "Return a JSON array of strings only."
    )

    et = (entity_type or "").strip().lower()
    extra_rules = ""
    if et == "procedure":
        extra_rules = (
            "Procedure-specific rules:\n"
            "- You MUST include at least 2 tags from these namespaces: process, action, object\n"
            "- Limit chemical:* tags to at most 2\n"
            "- Prefer process/action/object/crop tags over listing many products\n"
        )

    user = f"""
Given the content below, generate at most {max_tags} tags.

Rules:
- Use ONLY these namespaces: {", ".join(ALLOWED_NAMESPACES)}
- Output format MUST be a JSON array: ["ns:slug", ...]
- lowercase
- hyphen-separated slugs
- no extra text, no explanations

Entity type: {et}
{extra_rules}

Content:
{content}
""".strip()

    payload = {
        "model": cfg.model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user",   "content": [{"type": "input_text", "text": user}]},
        ],
        "temperature": 0,
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    last_err = None
    for attempt in range(cfg.max_retries):
        try:
            req = urllib.request.Request(cfg.base_url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")

            obj = json.loads(raw)

            texts = []
            for item in obj.get("output", []):
                for c in item.get("content", []):
                    if c.get("type") == "output_text" and "text" in c:
                        texts.append(c["text"])
            out_text = "\n".join(texts).strip()

            return ensure_json_array(out_text)

        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                err_body = ""
            last_err = f"{e} | body={err_body}"
            time.sleep(1.5 * (attempt + 1))

        except Exception as e:
            last_err = str(e)
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"OpenAI call failed after retries: {last_err}")


# ======================
# Main processing
# ======================

def build_content(row: pd.Series, fields: List[str], sep: str = "\n") -> str:
    parts = []
    for f in fields:
        if f in row and pd.notna(row[f]):
            v = str(row[f]).strip()
            if v:
                parts.append(f"{f}: {v}")
    return sep.join(parts).strip()


def load_cache(path: Path) -> Dict[str, List[str]]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(path: Path, cache: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_tags(existing: str, new_tags: List[str], mode: str) -> str:
    def parse_existing(s: str) -> List[str]:
        s = (s or "").strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except Exception:
            pass
        return [p.strip() for p in s.split("|") if p.strip()]

    ex = parse_existing(existing)
    if mode == "replace":
        merged = new_tags
    elif mode == "prepend":
        merged = []
        seen = set()
        for t in new_tags + ex:
            if t not in seen:
                merged.append(t); seen.add(t)
    elif mode == "append":
        merged = []
        seen = set()
        for t in ex + new_tags:
            if t not in seen:
                merged.append(t); seen.add(t)
    else:
        merged = new_tags

    return json.dumps(merged, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model (default: gpt-4.1-mini)")
    ap.add_argument("--fields", default="question,answer", help="Comma-separated fields to build content")

    # Migration columns
    ap.add_argument("--legacy-tags-col", default="tags", help="Legacy tags column to keep unchanged")
    ap.add_argument("--new-tags-col", default="tags_v2", help="New tags column to write (migration)")

    # Tagging controls
    ap.add_argument("--max-tags", type=int, default=6, help="Max tags to keep after normalization (final cap)")
    ap.add_argument("--raw-max-tags", type=int, default=8, help="Max tags GPT can output (before normalization)")
    ap.add_argument("--mode", choices=["replace", "prepend", "append"], default="replace", help="How to write tags_v2")

    # Range / perf
    ap.add_argument("--start", type=int, default=0, help="Start row index (0-based, inclusive)")
    ap.add_argument("--end", type=int, default=-1, help="End row index (0-based, inclusive). -1 means last row.")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between API calls")
    ap.add_argument("--cache", default="cache/tag_cache.json", help="Cache JSON path")
    ap.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default utf-8-sig)")
    ap.add_argument("--dry-run", action="store_true", help="Do NOT call API; only migrate/normalize legacy tags")

    # Entity type
    ap.add_argument("--entity-col", default="entity_type", help="Column name that contains entity_type")

    args = ap.parse_args()

    # API key from env (recommended)
    api_key = "..."
    if not args.dry_run and not api_key:
        print("ERROR: OPENAI_API_KEY is not set in environment.", file=sys.stderr)
        sys.exit(2)

    input_path = Path(args.input)
    output_path = Path(args.output)
    cache_path = Path(args.cache)

    df = pd.read_csv(input_path, encoding=args.encoding)

    if args.legacy_tags_col not in df.columns:
        df[args.legacy_tags_col] = ""
    if args.new_tags_col not in df.columns:
        df[args.new_tags_col] = ""

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    total = len(df)

    start = max(0, args.start)
    end = args.end if args.end >= 0 else total - 1
    end = min(end, total - 1)
    if start > end:
        print(f"ERROR: invalid range start={start} end={end} total={total}", file=sys.stderr)
        sys.exit(2)

    cache = load_cache(cache_path)
    cfg = OpenAIConfig(api_key=api_key, model=args.model)

    indices = list(range(start, end + 1))
    iterator = tqdm(indices, desc="Auto-tagging", unit="row") if (tqdm is not None) else indices

    for i in iterator:
        row = df.iloc[i]

        entity_col = args.entity_col
        et_value = row.get(entity_col, "") if entity_col in df.columns else ""

        if args.dry_run:
            # Freeze legacy tags (do NOT modify legacy col)
            legacy_existing = str(row.get(args.legacy_tags_col, "") or "")
            raw_list = ensure_json_array(legacy_existing)
            norm = normalize_tags(raw_list, max_tags=999)

            # Also enforce: always include hard entity tag (useful even in dry-run migration)
            forced_tags = force_tags_by_entity_type(et_value)
            merged = []
            seen = set()
            for t in forced_tags + norm:
                t2 = normalize_one_tag(t) if t else None
                if t2 and t2 not in seen:
                    merged.append(t2); seen.add(t2)

            merged = enforce_entity_rules(merged, str(et_value))
            merged = merged[:args.max_tags]

            # Write to new column only
            df.at[i, args.new_tags_col] = json.dumps(merged, ensure_ascii=False)
            continue

        content = build_content(row, fields=fields)
        if not content:
            continue

        h = stable_hash(content)
        if h in cache:
            raw_tags = cache[h]
        else:
            raw_tags = call_openai_tags(cfg, content=content, max_tags=args.raw_max_tags, entity_type=str(et_value))
            cache[h] = raw_tags
            if args.sleep > 0:
                time.sleep(args.sleep)

        forced_tags = force_tags_by_entity_type(et_value)
        norm_tags = normalize_tags(raw_tags, max_tags=args.max_tags)

        merged = []
        seen = set()
        for t in forced_tags + norm_tags:
            if t and t not in seen:
                merged.append(t)
                seen.add(t)

        # Enforce procedure rules and cap
        merged = enforce_entity_rules(merged, str(et_value))
        merged = merged[:args.max_tags]

        # Write tags_v2 only (freeze legacy)
        new_existing = str(row.get(args.new_tags_col, "") or "")
        df.at[i, args.new_tags_col] = upsert_tags(new_existing, merged, mode=args.mode)

        if (i - start + 1) % 50 == 0:
            save_cache(cache_path, cache)

    save_cache(cache_path, cache)
    df.to_csv(output_path, index=False, encoding=args.encoding)
    print(f"Done. Wrote: {output_path}  (rows processed: {start}..{end}, total={total})")
    print(f"Cache: {cache_path} (entries: {len(cache)})")


if __name__ == "__main__":
    main()
