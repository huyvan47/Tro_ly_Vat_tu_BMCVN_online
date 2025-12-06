import textwrap
import csv
from pathlib import Path

from openai import OpenAI

# ==============================
# CẤU HÌNH
# ==============================

MODEL_NAME = "gpt-4.1-mini"  # hoặc model bạn đang dùng
CATEGORY = "ic-top_co-che-tac-dung"
CROP_NAME = "ic-top281ssc"
ID_PREFIX = "IC_Top_Co_Che_Tac_Dung"

client = OpenAI(api_key="...")

# ==============================
# ĐOẠN SOP THÔ – BẠN THAY BẰNG TEXT CỦA BẠN
# ==============================
raw_text = """
• Tác dụng lâu bền ;
Đây là số liệu nghiên cứu dư lượng trên cây trồng (lá cây cam) của IC Top so với các thuốc gốc đồng khác. IC Top cho thấy tính bám dính tốt, lượng bám dính nhiều hơn so với các thuốc gốc đồng khác, cả khi lượng mưa lớn hơn 250mm thì lượng bám dính vẫn nhiều hơn 1 µg/cm2 (Lượng tối thiểu cần đế xác định là có hiệu quả phòng ngừa dịch bệnh), cho thấy khả năng chịu mưa tốt. Chính điều này mang lại hiệu quả khử khuẩn đế bảo vệ cây trồng trong thời gian dài.
Có thể nói, ưu điểm lớn nhất của IC Top chính là "Tác dụng lâu bền".
Đó là do không chỉ có chứa vôi, mà có vôi phản ứng và tồn tại kết hợp cùng với đồng. Nhờ thế mà chất khử khuẩn là đồng không bị mưa, tuyết rửa trôi quá mức.
Chính vì vậy, IC Top duy trì tác dụng khử khuẩn trong thời gian dài hơn, hiệu quả phòng ngừa dịch bệnh ổn định hơn so với thuốc gốc đồng thông thường. Ngoài ra, đồng không bị rửa trôi quá mức cũng có nghĩa ít gây hại cho cây trồng.
Xem xét mối tương quan giữa lượng đồng bám dính với hiệu quả khử khuẩn, ta thấy lượng đồng bám dính càng nhiều thì hiệu quả khử khuẩn càng cao.
Số liệu ghi chép sự thay đổi lượng đồng bám dính cho thấy IC Top có khả năng bám dính lâu nhất trên bề mặt cây trồng.
Trong bảng là kết quả khảo sát nồng độ đồng bị tan ra trong dung dịch phun tưới.
IC Top có lượng đồng tan trong dung dịch thấp nhất và vì thế mà đồng không bị rửa trôi quá mức.
Đối tượng thuốc nghiên cứu: IC Top 	  Ngày phun: 16/10/2020     Cây trồng: Cam         Lượng mưa lũy kế: 200mm
Không gây hại cho cây trồng. Đây là giống cam rất hay nhiễm bệnh đốm lá, nhưng ở đây thì không thấy phát sinh. Và dù phun tưới dung dịch pha loãng 200 lần thì sau 88 ngày vẫn còn thấy ICTop bám dính trên bề mặt. 
"""

# ==============================
# PROMPT MẪU CHO LLM
# ==============================
system_prompt = """
Bạn là chuyên gia nông nghiệp và data engineer. 
Nhiệm vụ: Chuyển đoạn văn bản thành các dòng CSV theo schema:
CHUYỂN ĐOẠN TÀI LIỆU KỸ THUẬT / QUY TRÌNH CANH TÁC / HƯỚNG DẪN SỬ DỤNG THUỐC
cho đối tượng: {CROP_NAME}
thành các dòng CSV theo schema:
id,question,answer,category,tags,alt_questions_v2

YÊU CẦU:
- id: tạo dạng {ID_PREFIX} + số thứ tự
- question: câu hỏi tự nhiên, rõ ràng, tiếng Việt.
- answer: giải thích chi tiết, giữ LIỀU LƯỢNG, BƯỚC, LƯU Ý QUAN TRỌNG, nhưng viết lại cho mạch lạc, dễ hiểu.
- category: luôn là "{category}".
- tags: 3–6 từ khóa tiếng Việt không dấu hoặc có dấu, phân tách bằng dấu |, ví dụ: "sau_rieng|lam_hoa|ham_nuoc".
- alt_questions_v2: 2–5 biến thể câu hỏi, tiếng Việt, phân tách bằng dấu |.

PHÂN LOẠI:
- Tạo 2–4 dòng dạng CHUNK lớn: mô tả các khối chính (bón lân + hãm nước; phun phân hóa mầm hoa; vô nước + kéo hoa + tỉa hoa; già lá + phòng bệnh).
- Tạo 2–4 dòng dạng Q&A GIẢI THÍCH: hỏi “vì sao”, “lưu ý”, “tại sao”, “ý nghĩa” (ví dụ: vì sao phải hãm nước, vì sao tỉa hoa 2 lần, vì sao không tưới nước khi xổ nhuỵ, vì sao thụ phấn thủ công…).

ĐỊNH DẠNG OUTPUT:
- CHỈ xuất ra CSV TEXT, có header:
  id,question,answer,category,tags,alt_questions_v2
- Mỗi dòng là 1 bản ghi.
- Escape dấu ngoặc kép đúng chuẩn CSV (bọc chuỗi bằng "..." nếu có dấu phẩy).
- Không giải thích thêm gì ngoài CSV.
"""

def build_prompt(raw: str) -> str:
    # Wrap text nếu muốn, ở đây gửi thẳng cũng được
    return f"Đây là đoạn quy trình gốc:\n\n{raw}\n\nHãy sinh các dòng CSV như yêu cầu."

def call_model(raw: str) -> str:
    prompt = build_prompt(raw)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt.format(
                CROP_NAME=CROP_NAME,
                ID_PREFIX=ID_PREFIX,
                category=CATEGORY
            )},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    csv_text = response.choices[0].message.content
    return csv_text

if __name__ == "__main__":
    csv_output = call_model(raw_text)

    # In ra màn hình để bạn xem trước
    print("=== CSV OUTPUT ===")
    print(csv_output)

    # Nếu muốn lưu ra file
    out_path = Path("process_data_kd.csv")
    out_path.write_text(csv_output, encoding="utf-8")
    print(f"\nĐã lưu CSV vào: {out_path.resolve()}")
