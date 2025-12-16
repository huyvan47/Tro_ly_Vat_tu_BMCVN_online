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
chinh_sach_va_gia_ban_chunk_00,"CHÍNH SÁCH BÁN HÀNG VÀ GIÁ BÁN 1. Giá bán Giá bán đã bao gồm VAT Có 3 khung giá : * Giá 1 và giá 2: dành cho khách mua lẻ, ít * Giá 3 ( giá net thấp nhất) : dành cho khách hàng mua lần đầu, Khách hàng mua số lượng lớn, KH làm nhãn riêng , KH ủng hộ thường xuyên,… Ví dụ: - Giá 1: Đơn hàng 1 thùng - Giá 2: Đơn hàng 3 thùng - Giá 3: Đơn hàng 20 thùng => Có thể linh động giữa các giá không nhất thiết theo khung giá công ty đã áp , tuy nhiên không được bán thấp hơn Giá net 2. Chiết khấu và khuyến mãi ""Tùy theo từng tháng mà có chương trình khuyến mãi khác nhau ( Có thể chiết khấu 3%, 5%, 15%, 20%, 50% ,… tùy sản phẩm hoặc chạy cắt lô đạt sản lượng được trừ lại tiền) Có 2 hình thức áp dụng : * Chiết khấu hoặc trừ tiền trực tiếp trên đơn hàng * Chiết khấu hoặc trừ tiền sau khi đạt đủ sản lượng đăng ký"" 3. Hình thức thanh toán ""* Thanh toán bằng tiền mặt hoặc Chuyển khoản * Thời hạn thanh toán tối đa: 7 ngày Đối với khách hàng thân thiết , KH lớn, uy tín, KH mới mua lần đầu muốn trải nghiệm sản phẩm có thể linh động thời hạn thanh toán lâu hơn "" 4. Giao hàng và vận chuyển ""Các hình thức giao hàng: * Xe vào công ty bốc hàng hoặc giao ra chành xe : Công ty hổ trợ cước vận chuyển lại cho khách tùy theo khu vực và số lượng hàng * Xe công ty : giao hàng tận nơi không tốn thêm cước"" 5. Chính sách trả hàng ""Điều kiện trả hàng về * Hàng trả về còn nguyên vẹn, còn hạn sử dụng * Hàng đang trong thời hạn sử dụng nhưng bị lỗi kỹ thuật hoặc chất lượng của nhà sản xuất Hàng trả về sẽ có tính phí tái chế nếu date từ 18-24 tháng: * Loại đóng gói có quy cách ≤ 100 gram ( hoặc ml) phí tái chế 30% trên giá trị lô hàng. * Loại đóng gói có quy cách trên 100 gram ( hoặc ml) phí tái chế 20% giá trị lô hàng. Hàng trả về sẽ trừ tiền theo thời điểm xuất bán và thu hồi toàn bộ Chương trình khuyến mãi"" 6"
chinh_sach_va_gia_ban_chunk_01,"đóng gói có quy cách ≤ 100 gram ( hoặc ml) phí tái chế 30% trên giá trị lô hàng. * Loại đóng gói có quy cách trên 100 gram ( hoặc ml) phí tái chế 20% giá trị lô hàng. Hàng trả về sẽ trừ tiền theo thời điểm xuất bán và thu hồi toàn bộ Chương trình khuyến mãi"" 6. Chăm sóc khách hàng sau mua ""* Hỗ trợ tư vấn kỹ thuật, giải đáp các thắc mắc về sản phẩm * Giới thiệu thêm các Chương trình , các sản phẩm khác cho khách * Cho phép đổi trả hàng nếu sản phẩm bị lỗi hoặc bán không được"" CHÍNH SÁCH BÁN HÀNG CỐ ĐỊNH CHÍNH SÁCH CHIẾT KHẤU CUỐI VỤ THUỐC BVTV 1. Chính sách chiết khấu thanh toán 8% ( hưởng giá tiền mặt). *Điều kiện hưởng chiết khấu 8% (hưởng giá tiền mặt): Thanh toán tiền hàng trong vòng 10 ngày kể từ ngày xuất hàng. Lưu ý: Nếu khách hàng thanh toán sau 10 ngày Công ty sẽ tự động chuyển lên giá tiền nợ 2. Chính sách thưởng cuối vụ *Điều kiện hưởng chính sách thưởng cuối vụ: Khách hàng thanh toán hết 100% công nợ trong vụ. ""Lưu ý: + Chính sách này không áp dụng với các sản phẩm trong bảng giá bán NÉT và không áp dụng với các đơn hàng tính giá NÉT hoặc đã được hưởng chính sách chiết khấu theo đơn hàng. +Các khách hàng bán hàng sang các tỉnh khác sẽ không được chi trả chiết khấu cuối vụ. +Doanh số hoàn thành được tính trên giá có VAT, thưởng cuối vụ được tính trên giá không VAT."" -Doanh thu đạt 200 triệu trở lên: Thưởng Cuối vụ 3%/doanh thu thực tế không VAT. -Doanh thu đạt 300 triệu trở lên: Thưởng Cuối vụ 4%/doanh thu thực tế không VAT. -Doanh thu đạt 700 triệu trở lên: Thưởng Cuối vụ 5%/doanh thu thực tế không VAT. -Doanh thu đạt 1 Tỷ trở lên: Thưởng Cuối vụ 6%/doanh thu thực tế không VAT. CHÍNH SÁCH BÁN THUỐC BVTV CHIẾT KHẤU THEO ĐƠN HÀNG 1. Chính sách chiết khấu thanh toán 8% ( hưởng giá tiền mặt). *Điều kiện hưởng chiết khấu 8% (hưởng giá tiền mặt)"
chinh_sach_va_gia_ban_chunk_02,"thực tế không VAT. -Doanh thu đạt 1 Tỷ trở lên: Thưởng Cuối vụ 6%/doanh thu thực tế không VAT. CHÍNH SÁCH BÁN THUỐC BVTV CHIẾT KHẤU THEO ĐƠN HÀNG 1. Chính sách chiết khấu thanh toán 8% ( hưởng giá tiền mặt). *Điều kiện hưởng chiết khấu 8% (hưởng giá tiền mặt): Thanh toán tiền hàng trong vòng 10 ngày kể từ ngày xuất hàng. Lưu ý: Nếu khách hàng thanh toán sau 10 ngày Công ty sẽ tự động chuyển lên giá tiền nợ 2. Chính sách chiết khấu theo đơn hàng: *Điều kiện hưởng chiết khấu theo đơn hàng: Thanh toán trong vòng 10 ngày kể từ ngày xuất hàng. ""Lưu ý: Chính sách này không áp dụng với các sản phẩm trong bảng giá Nét và không được cộng dồn đơn hàng."" -Chiết khấu 1%: áp dụng với đơn hàng có tổng cộng 5 thùng trở lên. -Chiết khấu 2%: áp dụng với đơn hàng có tổng cộng 10 thùng trở lên. -Chiết khấu 3%: áp dụng với đơn hàng có tổng cộng 20 thùng trở lên. -Chiết khấu 4%: Áp dụng với đơn hàng có tổng cộng 40 thùng trở lên."
"""

# ==============================
# PROMPT MẪU CHO LLM (TỰ SUY LUẬN TOÀN BỘ)
# ==============================

system_prompt = """
SYSTEM PROMPT

Bạn là hệ thống chuyển đổi dữ liệu (data transformer) cho pipeline RAG. Nhiệm vụ của bạn là tạo bản ghi CSV theo schema cố định, nhưng TUYỆT ĐỐI KHÔNG được chỉnh sửa nội dung gốc.

Quy tắc bắt buộc:

Giữ nguyên 100% nội dung của từng chunk (nguyên văn, không sửa chính tả, không thêm bớt, không diễn giải).

Mỗi chunk tạo đúng 1 dòng CSV.

Cột answer phải chứa nguyên văn chunk tương ứng.

Không tạo Q&A suy diễn. Không tổng hợp. Không rút gọn.

Không thêm bất kỳ dòng giải thích nào ngoài CSV.

USER PROMPT

NHIỆM VỤ

Bạn nhận vào N chunk văn bản thô (N có thể là 1, 7 hoặc nhiều hơn).

Hãy xuất ra CSV với MỖI CHUNK = 1 DÒNG.

Giữ nguyên 100% nội dung của chunk.

SCHEMA CSV (BẮT BUỘC)
id,question,answer,category,tags,alt_questions,img_keys

QUY ĐỊNH TỪNG CỘT

id

Dùng đúng id đã có sẵn ở đầu mỗi chunk (ví dụ: chinh_sach_ban_hang_chunk_00, chinh_sach_ban_hang_chunk_01, ...).

Không tự đổi tên id.

question

Phản ánh về giá cả và dịch vụ công ty

answer

Dán nguyên văn nội dung chunk tương ứng (giữ nguyên 100%).

Không chỉnh sửa, không chuẩn hóa, không thay ký tự, không xóa xuống dòng.

Giữ nguyên dấu câu, ký tự đặc biệt, bullet, mũi tên, dấu ngoặc, khoảng trắng, xuống dòng.

category

Lấy phần tiền tố trước "chunk" trong id.

Ví dụ: id = "chinh_sach_ban_hang_chunk_03" => category = "chinh_sach_ban_hang"

tags

Tự tạo sao cho phù hợp.

alt_questions

Tự tạo các câu người dùng có thể hỏi.

img_keys

Để trống.

RÀNG BUỘC ĐỊNH DẠNG CSV

Trả về duy nhất CSV text có header.

Không Markdown. Không bọc trong ``` .

Mỗi dòng dữ liệu phải hợp lệ CSV:

Nếu answer có dấu phẩy hoặc xuống dòng, PHẢI đặt answer trong dấu ngoặc kép ".

Bên trong answer, nếu có ký tự " thì escape thành "" (double quote).

Số dòng output = số chunk input.

INPUT CHUNKS
{{chunks_text}}

OUTPUT CSV ONLY.
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
        temperature=0.2,
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
