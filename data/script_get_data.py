import csv

file_path = "muc-luc-vat-tu-phu.csv"
first_column = []

with open(file_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if row:  # tránh dòng rỗng
            first_column.append(row[0])

print(first_column)
