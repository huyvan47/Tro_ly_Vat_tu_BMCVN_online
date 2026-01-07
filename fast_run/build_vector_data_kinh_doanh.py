import pandas as pd
import numpy as np
from openai import OpenAI
from pathlib import Path
import re
import json

# ==============================
#        CONFIG
# ==============================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/kb-audit/check-backbone/data-kd-1-4-chuan.csv"
OUT_FILE = "data-kd-1-4-chuan-v2.npz"

client = OpenAI(api_key="...")

# ==============================
#        LOAD CSV
# ==============================

df = pd.read_csv(DATA, encoding="utf-8")

required_cols = ["id", "question", "answer", "category", "tags", "alt_questions", "img_keys", "entity_type", "tags_v2"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Thiếu cột trong CSV: {missing}")

df["question"] = df["question"].astype(str)
df["answer"] = df["answer"].astype(str)
df["category"] = df["category"].astype(str)
df["tags"] = df["tags"].fillna("").astype(str)
df["alt_questions"] = df["alt_questions"].fillna("").astype(str)
df["img_keys"] = df["img_keys"].fillna("").astype(str)

# NEW
df["entity_type"] = df["entity_type"].fillna("").astype(str)
df["tags_v2"] = df["tags_v2"].fillna("").astype(str)

def tags_v2_to_pipe(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                arr = [str(x).strip() for x in arr if str(x).strip()]
                return "|".join(arr)
        except Exception:
            pass
    if "|" in s:
        return "|".join([p.strip() for p in s.split("|") if p.strip()])
    return s

df["tags_v2"] = df["tags_v2"].apply(tags_v2_to_pipe)

# ==============================
#   BUILD INPUT TEXTS FOR EMBED
# ==============================

EMBED_MODE = "Q_PLUS_A_FULL"
ANSWER_HEAD_CHARS = 800

def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

inputs = []
for _, row in df.iterrows():
    q = clean_text(row["question"])
    a = clean_text(row["answer"])
    alt = clean_text(row["alt_questions"])

    if EMBED_MODE == "Q_ONLY":
        text = f"Q: {q}"
        if alt:
            text += f"\nALT: {alt}"

    elif EMBED_MODE == "Q_PLUS_A_BRIEF":
        a_brief = a[:ANSWER_HEAD_CHARS]
        text = f"Q: {q}"
        if alt:
            text += f"\nALT: {alt}"
        if a_brief:
            text += f"\nA_BRIEF: {a_brief}"

    elif EMBED_MODE == "Q_PLUS_A_FULL":
        text = f"Q: {q}"
        if alt:
            text += f"\nALT: {alt}"
        if a:
            text += f"\nA: {a}"

    else:
        raise ValueError(f"EMBED_MODE không hợp lệ: {EMBED_MODE}")

    inputs.append(text)

print(f"🔢 Tổng số dòng cần embed: {len(inputs)}")

# ==============================
#       EMBEDDING (BATCH)
# ==============================

print("🚀 Bắt đầu embedding theo batch ...")

BATCH_SIZE = 200
all_embs = []

for start in range(0, len(inputs), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(inputs))
    batch = inputs[start:end]

    print(f"➡ Embedding batch {start} → {end - 1} (số lượng: {len(batch)})")

    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=batch,
    )

    all_embs.extend([item.embedding for item in resp.data])

embs = np.array(all_embs, dtype=np.float32)
assert embs.shape[0] == len(df), f"Mismatch: {embs.shape[0]} embeddings nhưng {len(df)} dòng CSV"

norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
embs = embs / norms

print("🔥 Embedding xong. Tổng số vector:", len(embs))

# ==============================
#        SAVE NPZ
# ==============================


OUT_PATH = Path(OUT_FILE)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

np.savez(
    OUT_FILE,
    embeddings=embs,
    questions=df["question"].to_numpy(dtype=object),
    answers=df["answer"].to_numpy(dtype=object),
    alt_questions=df["alt_questions"].to_numpy(dtype=object),
    category=df["category"].to_numpy(dtype=object),
    tags=df["tags"].to_numpy(dtype=object),
    img_keys=df["img_keys"].to_numpy(dtype=object),
    ids=df["id"].astype(str).to_numpy(dtype=object),

    # NEW for filter
    entity_type=df["entity_type"].to_numpy(dtype=object),
    tags_v2=df["tags_v2"].to_numpy(dtype=object),
)

print(f"✅ ĐÃ BUILD XONG VECTOR FILE → {OUT_FILE}")
