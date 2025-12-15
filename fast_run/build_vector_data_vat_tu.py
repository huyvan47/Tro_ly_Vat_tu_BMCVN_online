import pandas as pd
import numpy as np
from openai import OpenAI
from pathlib import Path
import re

# ==============================
#        CONFIG
# ==============================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/data-kinh-doanh/data-vat-tu-full-merge-overlap.csv"  # đổi đúng tên file mới của bạn
OUT_FILE = "data-vat-tu-full-merge-overlap.npz"

client = OpenAI(api_key="...")

# ==============================
#        LOAD CSV
# ==============================

df = pd.read_csv(DATA, encoding="utf-8")

# Đảm bảo các cột tồn tại đúng tên
required_cols = ["id", "question", "answer", "category", "tags", "alt_questions", "img_keys"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Thiếu cột trong CSV: {missing}")

# Ép kiểu về string để tránh NaN gây lỗi
df["question"] = df["question"].astype(str)
df["answer"] = df["answer"].astype(str)
df["category"] = df["category"].astype(str)
df["tags"] = df["tags"].fillna("").astype(str)
df["alt_questions"] = df["alt_questions"].fillna("").astype(str)
df["img_keys"] = df["img_keys"].fillna("").astype(str)

# ==============================
#   BUILD INPUT TEXTS FOR EMBED
# ==============================

EMBED_MODE = "Q_PLUS_A_FULL"   # chọn: "Q_ONLY", "Q_PLUS_A_BRIEF", "Q_PLUS_A_FULL"
ANSWER_HEAD_CHARS = 800        # chỉ dùng cho Q_PLUS_A_BRIEF (600–1200 là hợp lý)

def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

inputs = []
for _, row in df.iterrows():
    q = clean_text(row["question"])
    a = clean_text(row["answer"])
    alt = clean_text(row["alt_questions"])

    # Nếu alt_questions của bạn là nhiều câu, có thể giữ nguyên hoặc rút gọn
    # alt = alt[:800]  # tùy chọn

    if EMBED_MODE == "Q_ONLY":
        # Neo vào câu hỏi để tăng recall theo question/alt
        text = f"Q: {q}"
        if alt:
            text += f"\nALT: {alt}"

    elif EMBED_MODE == "Q_PLUS_A_BRIEF":
        # Neo vào Q nhưng vẫn có tín hiệu nội dung
        a_brief = a[:ANSWER_HEAD_CHARS]
        text = f"Q: {q}"
        if alt:
            text += f"\nALT: {alt}"
        if a_brief:
            text += f"\nA_BRIEF: {a_brief}"

    elif EMBED_MODE == "Q_PLUS_A_FULL":
        # Dùng full answer như hiện tại, nhưng format rõ ràng hơn
        text = f"Q: {q}"
        if alt:
            text += f"\nALT: {alt}"
        if a:
            text += f"\nA: {a}"

    else:
        raise ValueError(f"EMBED_MODE không hợp lệ: {EMBED_MODE}")

    inputs.append(text)

# ==============================
#       EMBEDDING
# ==============================

resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=inputs,
)

embs = np.array([item.embedding for item in resp.data], dtype=np.float32)

# Chuẩn hoá vector đơn vị
norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
embs = embs / norms

# ==============================
#        SAVE NPZ
# ==============================

np.savez(
    OUT_FILE,
    embeddings=embs,
    questions=df["question"].to_numpy(dtype=object),
    answers=df["answer"].to_numpy(dtype=object),
    alt_questions=df["alt_questions"].to_numpy(dtype=object),
    category=df["category"].to_numpy(dtype=object),
    tags=df["tags"].to_numpy(dtype=object),
    img_keys=df["img_keys"].to_numpy(dtype=object),
    ids=df["id"].astype(str).to_numpy(dtype=object),  # dùng string cho an toàn
)

print(f"ĐÃ BUILD XONG VECTOR FILE → {OUT_FILE}")
