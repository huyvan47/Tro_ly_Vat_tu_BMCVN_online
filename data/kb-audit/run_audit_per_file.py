from pathlib import Path
import subprocess
import sys

# ===== CẤU HÌNH =====
DATA_DIR = Path("check-backbone")      # thư mục chứa CSV
AUDIT_SCRIPT = Path("audit_single_csv.py")
OUTPUT_ROOT = Path("audit_results")
PYTHON = sys.executable                # dùng đúng python đang chạy

# ====================
OUTPUT_ROOT.mkdir(exist_ok=True)

csv_files = list(DATA_DIR.rglob("*.csv"))

print(f"🔍 Found {len(csv_files)} CSV files\n")

for csv in csv_files:
    rel = csv.relative_to(DATA_DIR)
    out_dir = OUTPUT_ROOT / rel.parent / csv.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"▶ Auditing: {csv}")

    result = subprocess.run(
        [
            PYTHON,
            str(AUDIT_SCRIPT),
            "--input", str(csv),
            "--out", str(out_dir),
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ ERROR")
        print(result.stderr)
    else:
        print("✅ Done\n")
