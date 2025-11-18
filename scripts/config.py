import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model chat mặc định (có thể dùng FT model nếu có)
CHAT_MODEL = os.getenv("FT_MODEL") or "gpt-4o-mini"

# (tùy chọn) dùng 1 assistant cố định để giữ cấu hình/tools
ASSISTANT_ID = os.getenv("ASSISTANT_ID", "")

# Mặc định attach tối đa bao nhiêu file_id vào 1 message
MAX_FILE_IDS_PER_MESSAGE = int(os.getenv("MAX_FILE_IDS_PER_MESSAGE", "100"))

# Nhịp nghỉ giữa các lần upload để tránh rate limit
UPLOAD_PACING_SECONDS = float(os.getenv("UPLOAD_PACING_SECONDS", "0.25"))

# Số lần retry tối đa khi upload lỗi
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
