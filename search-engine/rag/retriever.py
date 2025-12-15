import numpy as np
from rag.config import RAGConfig
from rag.reranker import llm_rerank

def embed_query(client, text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v

def search(client, kb, norm_query: str, top_k):
    EMBS, QUESTIONS, ANSWERS, ALT_QUESTIONS, CATEGORY, TAGS = kb

    vq = embed_query(client, norm_query)
    sims = EMBS @ vq

    idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in idx:
        item = {
            "question": str(QUESTIONS[i]),
            "alt_question": str(ALT_QUESTIONS[i]),
            "answer": str(ANSWERS[i]),
            "score": float(sims[i]),
        }
        if CATEGORY is not None:
            item["category"] = str(CATEGORY[i])
        if TAGS is not None:
            item["tags"] = str(TAGS[i])
        results.append(item)

    if RAGConfig.use_llm_rerank and len(results) > 1:
        results = llm_rerank(client, norm_query, results, RAGConfig.top_k_rerank)

    return results
