import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "chemical_tags_from_kb.json"

CHEMICALS = []

def load_chemicals():
    global CHEMICALS

    if KB_PATH.exists():
        with open(KB_PATH, "r", encoding="utf-8") as f:
            CHEMICALS = json.load(f)
    else:
        CHEMICALS = []

# Load ngay khi module được import
load_chemicals()

# Tạo regex động cho toàn bộ hoạt chất
if CHEMICALS:
    CHEMICAL_REGEX = re.compile(
        r"\b(" + "|".join(re.escape(c.lower()) for c in CHEMICALS) + r")\b"
    )
else:
    CHEMICAL_REGEX = re.compile(r"$^")   # không match gì nếu file rỗng