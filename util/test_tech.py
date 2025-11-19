from openai import OpenAI
import json
client = OpenAI(api_key="...")

SYSTEM_PROMPT = """
Bạn là Entity Resolver của hệ thống “mục lục vật tư phụ”.

Nhiệm vụ:
- Từ câu hỏi người dùng, xác định họ muốn hỏi ENTITY nào.
- ENTITY có trong danh sách dưới đây (đúng y tên gốc):

ENTITY_LIST:
'MỤC LỤC VẬT TƯ PHỤ', 'PHÂN BIỆT CẤU TẠO CHAI', 'BỘ CHAI', 'Bộ chai dùng chung', 'BỘ 3 LỚP', 'CHA30-02', 'CHA100-04', 'CHA240-TEEN', 'CHA240-ASMIL', 'CHA240-TEEN', 'CHA240-ASMIL', 'CHAI 240-TEEN VÀ 240-ASMIL', 'BỘ PET VÀNG', 'CHA100-14', 'CHA240-06', 'CHA480-04', 'CHA1000-09', 'CHA240-HELP', 'BỘ PET TRÒN TRẮNG', 'CHA100-03', 'CHA240-04', 'CHA480-01', 'CHA480-09', 'CHA480-10', 'CHA1000-03', 'CHA480-12', 'MẪU PET TRÒN TRẮNG 480 NẮP TRẮNG (480-09) BẢN CHẤT GIỐNG MẪU 480-04 PET TRÒN VÀNG (CHỈ KHÁC NHAU MÀU CHAI)', 'BỘ PET VUÔNG', 'CHA100-13', 'CHA240-05', 'CHA450-02', 'CHA450-04', 'CHA450-05', 'CHA450-06', 'CHA450-08', 'CHA450-09', 'CHA1000-02LOGO', 'BỘ PET ĐÓNG CỎ', 'CHA100480-01', 'CHA1000-04', 'CHA1000-15', 'CHA1000-20', 'SEAL LOGO CHAI PET ĐÓNG CỎ', 'NẮP LOGO CHAI PET ĐÓNG CỎ', 'BỘ HD XÁM (CHAI HD KHÔNG ĐÓNG ĐƯỢC LIỆU EC)', 'CHA240-08', 'CHA480-03', 'CHA1000-06', 'NẮP LOGO CHAI HD', 'SEAL LOGO CHAI HD', 'CHA1000-ZEROANVIL', 'THU1000-ZEROANVIL', 'MỘT SỐ MÃ CHAI LẺ KHÁC', 'CHA480-08 (CHAI PET TRÒN ĐEN)', 'CHA1000-07 (CHAI VUÔNG ĐỤC HDPE)', 'CHA1000-18 (CHAI BASTAR 1LIT HDPE)', 'CHA05L-01 (CAN XANH NHẠT, NẮP XANH)', 'CHAG-5000-0002 (CAN XANH ĐẬM, NẮP ĐỎ 5L)', 'CHA240-BASAGA (CHA240 VÀNG NHẠT HDPE NẮP VÀNG)', 'CHA240-09', 'NẮP LOGO CHAI LẺ KHÁC', 'SEAL LOGO CHAI LẺ KHÁC', 'LƯU Ý: CHAI 1000-07 HD VUÔNG 1 LÍT: NẮP LOGO BMC VÀ ĐÁY CHAI CÓ LOGO BMC', 'CHAI 1000-18 – NẮP LOGO + SEAL LOGO 10.0\\n3.2025', 'BỘ CHAI ĐẶT RIÊNG CHO KHÁCH > CHUYỂN BỘ DÙNG CHUNG', 'CHA240-BASP (BỘ CHAI BASP)', 'CHA480-BASP (BỘ CHAI BASP)', 'CHA240-BAYE (BỘ CHAI BAYE)', 'CHA480-BAYE (BỘ CHAI BAYE)', 'CHA480-11', 'BỘ CHAI ĐẶT RIÊNG CHO KHÁCH (1 LÍT)', 'CHAG-240-0002', 'CHAG-480-0002', 'CHAG-1000-0002', 'CHAG-100-0015', 'CHAG-240-0007', 'CHAG-240-0009', 'CHAG-480-0005', 'CHAG-480-0006', 'CHAG-1000-0005', 'CHAG-1000-0008', 'BỘ CHAI ĐẶT RIÊNG CHO KHÁCH', 'BỘ CHAI KHÁCH HÀNG MANG TỚI', 'THÙNG', 'THÙNG OFFSET', 'THAY ĐỔI TRÊN THÙNG', 'THÙNG Q7 900ML BMC', 'THÙNG Q7 900ML PHÚC THỊNH', 'THÙNG Q10 900ML DELTA', 'THÙNG TRÂU RỪNG MỚI 900ML DELTA', 'THÙNG GUSSI NHÃN ĐỎ 900ML DELTA', 'THÙNG VOI RỪNG 900ML AGRISHOP', 'THÙNG PHƯỢNG HOÀNG LỬA 900ML DELTA', 'THÙNG Q7 \\n4.5L BMC', 'THÙNG Q7 \\n4.5L PHÚC THỊNH', 'THÙNG Q10 \\n4.5L DELTA', 'THÙNG THÙNG TRÂU RỪNG MỚI \\n4.5L DELTA', 'TEM CHỐNG GIẢ', 'THAY ĐỔI TRÊN NHÃN CHAI', 'NHÃN + MÀNG + TÚI', 'NỘI DUNG CẦN PHẢI BIẾT'
Trả về đúng JSON:
{
  "entity": "<entity-name or null>",
  "confidence": 0.0-1.0
}

Không giải thích thêm.
"""

def llm_normalize(user_query: str):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        # response_format có cũng được, không có cũng được,
        # nhưng DÙ SAO vẫn phải lấy từ message.content
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_query},
        ],
    )

    content = resp.choices[0].message.content  # <-- đây là string JSON
    # In ra thử để debug nếu cần
    # print("RAW:", content)

    data = json.loads(content)
    return data


if __name__ == "__main__":
    result = llm_normalize("CHA240-ASM")
    print(result)
    # Ví dụ: {'entity': 'THÙNG', 'confidence': 0.97}