import numpy as np

def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)

    EMBS = data["embeddings"]
    QUESTIONS = data.get("questions", None)
    ANSWERS = data["answers"]
    ALT_QUESTIONS = data.get("alt_questions", None)
    CATEGORY = data.get("category", None)
    TAGS = data.get("tags", None)

    IDS = data.get("ids", data.get("id", None))
    if IDS is None:
        raise ValueError("NPZ missing 'ids' (or 'id') - required for VERBATIM mode.")
    
    TAGS_V2 = data.get("TAGS_V2", data.get("tags_v2", None))
    ENTITY_TYPE = data.get("ENTITY_TYPE", data.get("entity_type", None))

    return (EMBS, QUESTIONS, ANSWERS, ALT_QUESTIONS, CATEGORY, TAGS, IDS, TAGS_V2, ENTITY_TYPE)
