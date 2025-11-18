# Trợ lý Vật tư BMCVN — Template **All-Online** (File Search)

Kiến trúc: **giảm tải backend** — *không cần vector DB nội bộ.*  
OpenAI **File Search** lo toàn bộ **index + retrieval**; bạn chỉ **upload file** rồi gọi model (có thể là model fine-tune).

## Cấu trúc
```
Tro_ly_Vat_tu_BMCVN_online/
├── data/                     # Tài liệu nội bộ (.pdf/.docx/.txt/.xlsx/.csv...)
├── images/                   # Ảnh (chai/nhãn/bao bì...)
└── scripts/
    ├── config.py
    ├── upload_files_online.py  # Upload hàng loạt (100+ file) có retry + resume
    └── answer_online.py        # Hỏi → retrieval online (File Search) → trả lời
```

## Cài đặt
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Mở .env và điền OPENAI_API_KEY, (tùy chọn) ASSISTANT_ID
```

## Sử dụng nhanh
1) **Đặt file** vào `data/` và `images/`  
2) **Upload (100+ file ok, có retry & resume):**
```bash
python scripts/upload_files_online.py
```
3) **Hỏi (retrieval online):**
```bash
python scripts/answer_online.py "Phân biệt chai PET, HDPE và nhôm?"
```

### Ghi chú
- Uploader có **resume**: không upload lại file đã có `file_id` (dựa trên đường dẫn + mtime + size).  
- Có **retry + exponential backoff** và **rate-limit pacing** (mặc định 0.25s/lượt).  
- Bạn có thể **chia nhóm** bằng cách chạy uploader cho từng thư mục con (hoặc chỉnh `ALLOWED_EXTS`).  
- Nếu danh sách file rất lớn, `answer_online.py` sẽ tự **chunk** `file_ids` theo lô để gắn vào thread.

— Generated on 2025-11-12
