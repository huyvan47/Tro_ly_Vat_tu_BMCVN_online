from pathlib import Path
from openai import OpenAI

# ==============================
# CẤU HÌNH
# ==============================

MODEL_NAME = "gpt-4.1-mini"  # hoặc model bạn đang dùng

# Khuyến nghị set OPENAI_API_KEY trong biến môi trường:
# export OPENAI_API_KEY="sk-xxxx"
client = OpenAI(api_key="...")

# ==============================
# ĐOẠN SOP THÔ – BẠN THAY BẰNG TEXT CỦA BẠN
# ==============================

raw_text = """
Lời nói đầu
Bệnh hại cây trồng tiếp tục gây thiệt hại mùa màng đáng kể ở Việt Nam và các khu
vực khác có khí hậu nhiệt đới ở Đông Nam Á. Bệnh dịch vàng lùn và lùn xoắn
lá trên lúa ở miền Nam Việt Nam gần đây đánh dấu những tác động đáng kể của
bệnh cây đối với kinh tế xã hội ở cấp quốc gia.
Sự bùng phát dịch bệnh trên các cây trồng có giá trị kinh tế có thể tác động lớn
đến từng hộ nông dân tại những địa phương có ít cây trồng thay thế phù hợp –
phức hợp bệnh héo trên gừng ở Quảng Nam là một ví dụ.
Việc chẩn đoán chính xác tác nhân gây bệnh là yếu tố quan trọng quyết định sự
thành công của các biện pháp phòng trừ. Tuy nhiên, nhiều bệnh hại có các triệu
chứng giống nhau, khiến cho việc chẩn đoán tại chỗ gặp nhiều khó khăn, đôi khi
không thể thực hiện được. Vì vậy, các phòng thí nghiệm chẩn đoán là một thành
phần không thể thiếu trong mạng lưới bảo vệ thực vật. Cán bộ nhận trách nhiệm
làm công việc chẩn đoán bệnh cây cần phải trải qua quá trình đào tạo bài bản ở
trình độ đại học và sau đại học về kỹ năng nghiên cứu trong phòng thí nghiệm và
ngoài đồng ruộng, ngoài ra còn phải nắm vững những khái niệm cơ bản về bệnh
cây và quản lý bệnh hại tổng hợp.
Việc chẩn đoán chính xác tác nhân gây bệnh cũng vô cùng cần thiết cho việc xây
dựng và phát triển một cơ sở dữ liệu bệnh cây quốc gia một cách khoa học. Cơ sở
dữ liệu về bệnh cây ở Việt Nam sẽ là một phần then chốt cho sự thành công của
công tác kiểm dịch thực vật. Hơn nữa, cơ sở dữ liệu quốc gia là một phần quan
trọng của các biện pháp an ninh sinh học liên quan tới vấn đề trao đổi thương mại
hàng nông sản, đặc biệt đối với những quốc gia thành viên của Tổ chức Thương
mại Thế giới.
Cuốn cẩm nang này được biên soạn nhằm giúp các nhà nghiên cứu bệnh cây phát
triển những kỹ năng cơ bản trong việc chẩn đoán tác nhân gây bệnh, chủ yếu là các
bệnh do nấm ở rễ và thân cây. Những bệnh này thường ẩn, không biểu hiện triệu
chứng ngay nhưng gây ra những tổn thất đáng kể về mặt kinh tế xã hội ở Việt Nam.
"""

# ==============================
# PROMPT MẪU CHO LLM (TỰ SUY LUẬN TOÀN BỘ)
# ==============================

system_prompt = """
Bạn là chuyên gia nông nghiệp và data engineer. 

NHIỆM VỤ:
- Tự suy luận:
  - Đối tượng cây trồng / sản phẩm
  - Category phù hợp
  - ID_PREFIX phù hợp theo nội dung
- Sau đó chuyển đoạn văn bản thành các dòng CSV theo schema:

id,question,answer,category,tags,alt_questions

YÊU CẦU:
- id:
  - Tự tạo ID_PREFIX theo nội dung (ví dụ: ic_top, citrus_scab, rose_disease…)
  - Sau đó đánh số tăng dần: _chunk_01, _chunk_02, _qa_01…
- question: câu hỏi tự nhiên, rõ ràng, tiếng Việt.
- answer: giải thích chi tiết, giữ LIỀU LƯỢNG, CƠ CHẾ, LƯU Ý AN TOÀN nếu có.
- category:
  - Tự chọn category phù hợp theo nội dung (ví dụ: thuoc_benh_citrus, benh_cay_hong, co_che_tac_dong…)
- tags: 3–6 từ khóa, phân tách bằng dấu |
- alt_questions_v2: 2–5 biến thể câu hỏi, phân tách bằng dấu |

PHÂN LOẠI:
- Tự tạo:
  - 2–4 dòng CHUNK kiến thức
  - 2–4 dòng Q&A giải thích “vì sao – khi nào – lưu ý – an toàn – cơ chế”

ĐỊNH DẠNG OUTPUT:
- CHỈ xuất CSV TEXT
- Có header
- Escape đúng chuẩn CSV
- KHÔNG giải thích thêm bất kỳ dòng nào ngoài CSV
"""

def build_prompt(raw: str) -> str:
    return f"""
Đây là nội dung gốc cần xử lý:

{raw}

Hãy sinh dữ liệu CSV đúng yêu cầu.
"""

def call_model(raw: str) -> str:
    prompt = build_prompt(raw)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    csv_output = call_model(raw_text)

    # In ra để kiểm tra
    print("=== CSV OUTPUT ===")
    print(csv_output)

    # Lưu file
    out_path = Path("process_data_kd.csv")
    out_path.write_text(csv_output, encoding="utf-8")
    print(f"\n✅ Đã lưu CSV vào: {out_path.resolve()}")
