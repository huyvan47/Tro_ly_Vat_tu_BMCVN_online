import numpy as np

def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    embs = data["embeddings"]
    questions = data["questions"]
    answers = data["answers"]
    alt_questions = data["alt_questions"]
    category = data.get("category", None)
    tags = data.get("tags", None)

    # validate tối thiểu
    n = len(questions)
    assert len(answers) == n and len(alt_questions) == n and embs.shape[0] == n

    return embs, questions, answers, alt_questions, category, tags
