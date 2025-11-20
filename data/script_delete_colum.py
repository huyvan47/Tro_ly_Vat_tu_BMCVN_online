import pandas as pd

input_file = "tong-hop-data-phong-vat-tu.csv"       # đổi thành file của bạn
output_file = "tong-hop-data-phong-vat-tu-fix.csv"  # file sau khi xóa cột

# Đọc file CSV
df = pd.read_csv(input_file, encoding="utf-8")

# Lấy cột đầu tiên
col0 = df.columns[0]

# Xóa dấu " và chuyển thành chữ thường
df[col0] = (
    df[col0]
    .astype(str)         # đảm bảo kiểu dữ liệu là chuỗi
    .str.replace('"', '', regex=False)
    .str.lower()
    .str.strip()         # xóa khoảng trắng thừa
)

# Lưu lại file mới
df.to_csv(output_file, index=False, encoding="utf-8")

print("Đã xử lý xong. File mới:", output_file)