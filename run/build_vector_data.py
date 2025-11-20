import pandas as pd
import numpy as np
from openai import OpenAI
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent 
DATA = ROOT / "data/tong-hop-data-phong-vat-tu-fix.csv"

client = OpenAI(api_key="...")

df = pd.read_csv(DATA, encoding="utf-8")  # có cột 'question', 'answer'

questions = df["question"].tolist()
answers   = df["answer"].tolist()

resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=questions,
)

embs = np.array([item.embedding for item in resp.data], dtype=np.float32)
# chuẩn hoá cho cosine similarity
norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
embs = embs / norms

np.savez("tong-hop-data-phong-vat-tu.npz", embeddings=embs, questions=np.array(questions, dtype=object), answers=np.array(answers, dtype=object))
print("Đã build xong index.")