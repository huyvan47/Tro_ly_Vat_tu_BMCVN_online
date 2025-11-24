import pandas as pd
import numpy as np
from openai import OpenAI
from pathlib import Path

# ==============================
#        CONFIG
# ==============================

ROOT = Path(__file__).resolve().parent.parent 
DATA = ROOT / "data/data-vat-tu-full-done.csv"
OUT_FILE = "tong-hop-data-phong-vat-tu-fix-24-11.npz"

client = OpenAI(api_key="...")

# ==============================
#        LOAD CSV
# ==============================

df = pd.read_csv(DATA, encoding="utf-8")

questions = df["question"].astype(str).tolist()
answers   = df["answer"].astype(str).tolist()

# ==============================
#  BUILD INPUT TEXTS FOR EMBEDS
#  GỘP CẢ Q + A THÀNH 1 ĐOẠN
# ==============================

inputs = [
    f"Q: {q}\nA: {a}"
    for q, a in zip(questions, answers)
]

# ==============================
#       EMBEDDING
# ==============================

resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=inputs,
)

embs = np.array([item.embedding for item in resp.data], dtype=np.float32)

# Chuẩn hoá
norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
embs = embs / norms

# ==============================
#        SAVE NPZ
# ==============================

np.savez(
    OUT_FILE,
    embeddings=embs,
    questions=np.array(questions, dtype=object),
    answers=np.array(answers, dtype=object),
)

print(f"Đã build xong index (Q+A) → {OUT_FILE}")
