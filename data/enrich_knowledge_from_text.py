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
"Chính sách bán hàng 1. Giá bán: đã bao gồm VATCó 3 khung giá : * Giá 1 và giá 2: dành cho khách mua lẻ, ít* Giá 3 ( giá net thấp nhất) : dành cho khách hàng mua lần đầu, Khách hàng mua số lượng lớn, KH làm nhãn riêng , KH ủng hộ thường xuyên,… Ví dụ: - Giá 1: Đơn hàng 1 thùng- Giá 2: Đơn hàng 3 thùng - Giá 3: Đơn hàng 20 thùng => Có thể linh động giữa các giá không nhất thiết theo khung giá công ty đã áp , tuy nhiên không được bán thấp hơn Giá net. 2. Chiết khấu và khuyến mãi. Tùy theo từng tháng mà có chương trình khuyến mãi khác nhau ( Có thể chiết khấu 3%, 5%, 15%, 20%, 50% ,… tùy sản phẩm hoặc chạy cắt lô đạt sản lượng được trừ lại tiền)Có 2 hình thức áp dụng :* Chiết khấu hoặc trừ tiền trực tiếp trên đơn hàng * Chiết khấu hoặc trừ tiền sau khi đạt đủ sản lượng đăng ký 3. Hình thức thanh toán * Thanh toán bằng tiền mặt hoặc Chuyển khoản * Thời hạn thanh toán tối đa: 7 ngày Đối với khách hàng thân thiết , KH lớn, uy tín, KH mới mua lần đầu muốn trải nghiệm sản phẩm có thể linh động thời hạn thanh toán lâu hơn. 4. Giao hàng và vận chuyển. Các hình thức giao hàng:* Xe vào công ty bốc hàng hoặc giao ra chành xe : Công ty hổ trợ cước vận chuyển lại cho khách tùy theo khu vực và số lượng hàng * Xe công ty : giao hàng tận nơi không tốn thêm cước. 5.Chính sách trả hàng. Điều kiện trả hàng về* Hàng trả về còn nguyên vẹn, còn hạn sử dụng* Hàng đang trong thời hạn sử dụng nhưng bị lỗi kỹ thuật hoặc chất lượng của nhà sản xuấtHàng trả về sẽ có tính phí tái chế nếu date từ 18-24 tháng:* Loại đóng gói có quy cách ≤ 100 gram ( hoặc ml) phí tái chế 30% trên giá trị lô hàng.* Loại đóng gói có quy cách trên 100 gram ( hoặc ml) phí tái chế 20% giá trị lô hàng.Hàng trả về sẽ trừ tiền theo thời điểm xuất bán và thu hồi toàn bộ Chương trình khuyến mãi. 6.chinh_sach_ban_hang_chunk_01, 'đóng gói có quy cách ≤ 100 gram ( hoặc ml) phí tái chế 30% trên giá trị lô hàng.* Loại đóng gói có quy cách trên 100 gram ( hoặc ml) phí tái chế 20% giá trị lô hàng. Hàng trả về sẽ trừ tiền theo thời điểm xuất bán và thu hồi toàn bộ Chương trình khuyến mãi'6. Chăm sóc khách hàng sau mua ''* Hỗ trợ tư vấn kỹ thuật, giải đáp các thắc mắc về sản phẩm * Giới thiệu thêm các Chương trình, các sản phẩm khác cho khách* Cho phép đổi trả hàng nếu sản phẩm bị lỗi hoặc bán không được"""

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
