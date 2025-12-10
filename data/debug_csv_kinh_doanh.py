from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data-kinh-doanh_FIXED-3.csv"

with open(DATA, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        if i == 1016:
            print(f"=== LINE {i} ===")
            print(line)
            break
