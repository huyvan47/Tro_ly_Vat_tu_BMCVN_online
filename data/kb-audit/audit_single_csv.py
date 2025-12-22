import pandas as pd
import re
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

CSV_PATH = args.input
OUT_DIR = Path(args.out)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[audit] input={CSV_PATH}")
print(f"[audit] out={OUT_DIR.resolve()}")

ALLOWED_ENTITY_TYPES = {
    "registry", "product", "disease", "procedure", "pest", "weed", "general", "skill"
}

# Heuristic keywords (bạn có thể tinh chỉnh theo domain của bạn)
KW_DISEASE_Q = re.compile(r"\b(bệnh|triệu chứng|nấm bệnh|thối rễ|vàng lá|đốm lá|sẹo|cháy lá)\b", re.I)
KW_PEST_Q    = re.compile(r"\b(sâu|nhện|bọ|rầy|ruồi|tuyến trùng|côn trùng)\b", re.I)
KW_WEED_Q    = re.compile(r"\b(cỏ|diệt cỏ|cỏ dại)\b", re.I)

KW_REGISTRY_A = re.compile(r"\b(đơn vị đăng ký|đăng ký|tt:|tên thương phẩm|hoạt chất|min\s*\d+%|hàm lượng)\b", re.I)
KW_PRODUCT_A  = re.compile(r"\b(thành phần|hoạt chất|hàm lượng|dạng|quy cách|nhãn|nhà sản xuất)\b", re.I)

def is_empty(x):
    return pd.isna(x) or str(x).strip() == ""

def split_pipe(s):
    if is_empty(s): 
        return []
    return [x.strip() for x in str(s).split("|") if x.strip()]

def audit():
    df = pd.read_csv(CSV_PATH)

    issues = []
    summary = {}

    # ---------- A) Integrity ----------
    summary["rows"] = int(len(df))
    summary["columns"] = list(df.columns)

    # Null/empty counts
    nulls = df.isna().sum().to_dict()
    empties = {c: int((df[c].fillna("").astype(str).str.strip() == "").sum()) for c in df.columns}
    summary["nulls"] = {k: int(v) for k, v in nulls.items()}
    summary["empties"] = empties

    # ID issues
    if "id" in df.columns:
        dup_mask = df["id"].duplicated(keep=False)
        dup_count = int(dup_mask.sum())
        summary["duplicate_id_rows"] = dup_count

        # Record duplicate ID issues
        for idx, row in df[dup_mask].iterrows():
            issues.append({
                "row_index": int(idx),
                "id": row.get("id", ""),
                "issue_type": "DUPLICATE_ID",
                "severity": "HIGH",
                "detail": "id bị trùng"
            })

        # Null/empty id
        empty_id_mask = df["id"].isna() | (df["id"].astype(str).str.strip() == "")
        for idx, row in df[empty_id_mask].iterrows():
            issues.append({
                "row_index": int(idx),
                "id": row.get("id", ""),
                "issue_type": "EMPTY_ID",
                "severity": "HIGH",
                "detail": "id rỗng"
            })

    # ---------- B) Metadata hygiene ----------
    # entity_type whitelist
    df["entity_type_norm"] = df["entity_type"].astype(str).str.strip().str.lower()
    bad_et = ~df["entity_type_norm"].isin(ALLOWED_ENTITY_TYPES)
    summary["invalid_entity_type_rows"] = int(bad_et.sum())

    for idx, row in df[bad_et].iterrows():
        issues.append({
            "row_index": int(idx),
            "id": row.get("id", ""),
            "issue_type": "INVALID_ENTITY_TYPE",
            "severity": "HIGH",
            "detail": f"entity_type='{row.get('entity_type')}' không nằm trong whitelist"
        })

    # tags format checks (pipe-delimited, tokens are slug-ish)
    # Bạn đang dùng underscore nhiều, nên chỉ check “có token rỗng / token có khoảng trắng”
    if "tags" in df.columns:
        for idx, row in df.iterrows():
            tags = split_pipe(row.get("tags", ""))
            if tags:
                # token chứa space => dễ gây lỗi filter/normalize
                if any(" " in t for t in tags):
                    issues.append({
                        "row_index": int(idx),
                        "id": row.get("id", ""),
                        "issue_type": "TAG_HAS_SPACE",
                        "severity": "MED",
                        "detail": f"tags có token chứa space: {tags}"
                    })
                # token rỗng do '||' hoặc '|' ở cuối
                if str(row.get("tags", "")).strip().endswith("|") or "||" in str(row.get("tags", "")):
                    issues.append({
                        "row_index": int(idx),
                        "id": row.get("id", ""),
                        "issue_type": "TAG_DELIMITER_PROBLEM",
                        "severity": "LOW",
                        "detail": f"tags có dấu phân tách bất thường: '{row.get('tags')}'"
                    })

    # alt_questions delimiter check
    if "alt_questions" in df.columns:
        # Chỉ cảnh báo nếu có dấu xuống dòng/ dấu phân tách lạ
        for idx, row in df.iterrows():
            aq = row.get("alt_questions", "")
            if not is_empty(aq):
                s = str(aq)
                if "\n" in s:
                    issues.append({
                        "row_index": int(idx),
                        "id": row.get("id", ""),
                        "issue_type": "ALTQ_HAS_NEWLINE",
                        "severity": "LOW",
                        "detail": "alt_questions có newline (cần chuẩn hoá delimiter)"
                    })

    # ---------- C) entity_type semantic consistency (heuristic) ----------
    for idx, row in df.iterrows():
        et = row["entity_type_norm"]
        q = str(row.get("question", ""))
        a = str(row.get("answer", ""))

        # Nếu là product/registry mà question hỏi bệnh/sâu/cỏ -> nghi sai loại
        if et in {"product", "registry"}:
            if KW_DISEASE_Q.search(q) or KW_PEST_Q.search(q) or KW_WEED_Q.search(q):
                issues.append({
                    "row_index": int(idx),
                    "id": row.get("id", ""),
                    "issue_type": "ET_MISMATCH_Q",
                    "severity": "HIGH",
                    "detail": f"entity_type={et} nhưng question có dấu hiệu bệnh/sâu/cỏ"
                })

        # Nếu là disease/pest/weed mà answer mang dáng registry/product -> nghi sai loại hoặc answer nhiễm data đăng ký
        if et in {"disease", "pest", "weed"}:
            if KW_REGISTRY_A.search(a) or KW_PRODUCT_A.search(a):
                issues.append({
                    "row_index": int(idx),
                    "id": row.get("id", ""),
                    "issue_type": "ET_MISMATCH_A",
                    "severity": "HIGH",
                    "detail": f"entity_type={et} nhưng answer giống registry/product"
                })

        # procedure mà answer giống registry
        if et == "procedure" and KW_REGISTRY_A.search(a):
            issues.append({
                "row_index": int(idx),
                "id": row.get("id", ""),
                "issue_type": "PROCEDURE_LOOKS_LIKE_REGISTRY",
                "severity": "MED",
                "detail": "procedure nhưng nội dung giống dữ liệu đăng ký"
            })

    # ---------- D) Coverage / distribution ----------
    et_stats = (
        df.groupby("entity_type_norm")
        .agg(
            rows=("entity_type_norm", "size"),
            missing_tags=("tags", lambda s: int(s.isna().sum())),
            missing_category=("category", lambda s: int(s.isna().sum())),
            missing_img=("img_keys", lambda s: int(s.isna().sum()))
        )
        .reset_index()
        .rename(columns={"entity_type_norm": "entity_type"})
        .sort_values("rows", ascending=False)
    )

    # Outputs
    out_dir = OUT_DIR

    pd.DataFrame(issues).to_csv(
        out_dir / "audit_issues.csv",
        index=False,
        encoding="utf-8-sig"
    )

    (out_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    et_stats.to_csv(
        out_dir / "audit_entity_type_stats.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Simple HTML report
    issues_df = pd.DataFrame(issues)
    html = []
    html.append("<html><head><meta charset='utf-8'><title>RAG KB Audit Report</title></head><body>")
    html.append("<h1>RAG KB Audit Report</h1>")
    html.append("<h2>Summary</h2>")
    html.append("<pre>" + json.dumps(summary, ensure_ascii=False, indent=2) + "</pre>")
    html.append("<h2>Entity type stats</h2>")
    html.append(et_stats.to_html(index=False))
    html.append("<h2>Top issues (first 300)</h2>")
    if len(issues_df) == 0:
        html.append("<p>No issues found by current rules.</p>")
    else:
        html.append(issues_df.head(300).to_html(index=False))
    html.append("</body></html>")

    (out_dir / "audit_report.html").write_text(
        "\n".join(html),
        encoding="utf-8"
    )

    print(f"Wrote audit outputs to: {out_dir.resolve()}")

if __name__ == "__main__":
    audit()
