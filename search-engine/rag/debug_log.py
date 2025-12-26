from datetime import datetime

DEBUG_LOG_FILE = "debug_flow.log"

def debug_log(*args):
    """
    Ghi log giống print(), nhưng ghi vào file (append)
    """
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
        for arg in args:
            f.write(str(arg) + "\n")
        f.write("\n")  # cách dòng cho dễ đọc
