import pandas as pd
import numpy as np
from openai import OpenAI
from pathlib import Path
import re

# ==============================
#        CONFIG
# ==============================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/data-vat-tu/data-vat-tu-full-fixed.csv"  # đổi đúng tên file mới của bạn
OUT_FILE = "tong-hop-data-phong-vat-tu-fix-12-12-1.npz"

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

inputs = []
for _, row in df.iterrows():
    q = row["question"]
    a = row["answer"]
    alt = row["alt_questions"]

    # Làm sạch alt cho chắc (tránh 'nan', khoảng trắng)
    alt = alt.strip()
    if alt:
        text = f"Hỏi: {q}. Cách hỏi khác: {alt}. Trả lời: {a}."
    else:
        text = f"Hỏi: {q}. Trả lời: {a}."

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
