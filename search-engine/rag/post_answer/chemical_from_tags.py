from typing import List

def extract_chemicals_from_matched_tags(
    any_tags: List[str],
    must_tags: List[str],
    limit: int = 2,
) -> List[str]:
    """
    Trích hoạt chất từ TAG ANALYSIS (nguồn chuẩn KB)
    Ví dụ tag: 'chemical:acetochlor'
    """
    chems = []

    for t in (any_tags or []) + (must_tags or []):
        if not isinstance(t, str):
            continue
        if t.lower().startswith("chemical:"):
            c = t.split(":", 1)[1].strip().lower()
            if c and c not in chems:
                chems.append(c)
        if len(chems) >= limit:
            break

    return chems
