import numpy as np
from openai import OpenAI

client = OpenAI(api_key="...")

data = np.load("qa_index.npz", allow_pickle=True)
EMBS = data["embeddings"]          # (N, d)
QUESTIONS = data["questions"]      # (N,)
ANSWERS = data["answers"]          # (N,)

def embed_query(text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v

def search(query: str, top_k: int = 5):
    vq = embed_query(query)  # (d,)
    sims = EMBS @ vq         # (N,) – cosine similarity
    idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in idx:
        results.append({
            "question": str(QUESTIONS[i]),
            "answer": str(ANSWERS[i]),
            "score": float(sims[i]),
        })
    return results

if __name__ == "__main__":
    q = "pet tròng trắng"
    hits = search(q, top_k=5)
    for h in hits:
        print(f"{h['score']:.4f} | {h['question']} -> {h['answer'][:50]}...")
