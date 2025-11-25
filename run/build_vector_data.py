import pandas as pd
import numpy as np
from openai import OpenAI
from pathlib import Path
import re

# ==============================
#        CONFIG
# ==============================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/data-vat-tu-full-done-enriched-25-11-PARA-CLEAN.csv"
OUT_FILE = "tong-hop-data-phong-vat-tu-fix-25-11-PARA-CLEAN-QAAL.npz"

client = OpenAI(
    api_key="..."   # hoặc bỏ dòng này nếu bạn dùng biến môi trường
)

# ==============================
#        LOAD CSV
# ==============================

df = pd.read_csv(DATA, encoding="utf-8")

df["question"] = df["question"].astype(str)
df["answer"] = df["answer"].astype(str)

if "alt_questions_clean" not in df.columns:
    df["alt_questions_clean"] = ""

# ==============================
#   BUILD INPUT TEXTS FOR EMBED
# ==============================

inputs = []
for _, row in df.iterrows():
    q = row["question"]
    a = row["answer"]
    alt = row["alt_questions_clean"]

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
    alt_questions=df["alt_questions_clean"].to_numpy(dtype=object),
    category=df["category"].to_numpy(dtype=object),
    tags=df["tags"].to_numpy(dtype=object),
    ids=df["id"].to_numpy(dtype=int),
)

print(f"ĐÃ BUILD XONG VECTOR FILE → {OUT_FILE}")
