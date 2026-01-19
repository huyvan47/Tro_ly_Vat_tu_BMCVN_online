from dataclasses import dataclass

@dataclass(frozen=True)
class RAGConfig:

    # ===== MULTI QUERY CONFIG =====
    use_multi_query: bool = True          # master switch
    enable_multi_query_log: bool = True
    enable_timing_log: bool = True
    max_sub_queries: int = 3               # tối đa số sub query LLM sinh ra
    multi_query_top_k: int = 80            # top_k cho mỗi sub-query
    rrf_k: int = 60                        # tham số k cho RRF
    rrf_top_n: int = 400                   # số doc tối đa sau fuse

    # Multi-hop
    use_multi_hop = True
    max_multi_hops = 3
    multi_hop_top_k = 60
    multi_hop_stop_threshold = 180

    min_score_main: float = 0.35
    """
    1️⃣ min_score_main: float = 0.35
        📌 Ý nghĩa

    Ngưỡng điểm tối thiểu để một document:

    được xem là “đủ tốt”

    được dùng cho main answer

    📍 Ở đâu dùng?

    Trong:

    scoring.py

    strategy.py

    🧠 Logic:
    Nếu fused_score >= min_score_main
    → có thể dùng làm nguồn chính
    Ngược lại → không đủ tin cậy
    """

    min_score_suggest: float = 0.40
    """
    2️⃣ min_score_suggest: float = 0.40
    📌 Ý nghĩa

    Ngưỡng để gợi ý thêm tài liệu liên quan, không dùng làm câu trả lời chính.

    🧠 Dùng khi:

    Query không đủ mạnh để trả lời chắc chắn

    Nhưng vẫn muốn hiển thị “Có thể bạn quan tâm…”

    📍 Thường dùng trong:

    suggestion list

    alternative docs

    💡 Nếu min_score_main < min_score_suggest → nghĩa là:

    “Muốn suggest thì phải chắc hơn trả lời chính”
    → cách làm này khá an toàn.
    """
    max_suggest: int = 0

    """
    3️⃣ max_suggest: int = 0
    📌 Ý nghĩa

    Số lượng gợi ý phụ được trả về.

    0 → không hiển thị gợi ý

    >0 → cho phép hiển thị thêm tài liệu liên quan

    📌 Với hệ thống của bạn:
    👉 đặt 0 là hợp lý vì đang tập trung vào 1 câu trả lời đúng, không phải search engine.
    """

    use_llm_rerank: bool = False

    """
        4️⃣ use_llm_rerank: bool = False
    📌 Ý nghĩa

    Có dùng LLM để rerank lại top documents hay không.

    Nếu = True:

    Lấy top_k

    Gửi nội dung vào LLM

    LLM đánh giá lại độ liên quan

    Nếu = False:

    Chỉ dùng embedding similarity
    """
    top_k_rerank: int = 30

    """
    5️⃣ top_k_rerank: int = 30
    📌 Ý nghĩa

    Nếu bật rerank → chỉ rerank top N document.

    📌 Không có tác dụng nếu use_llm_rerank = False
    """

    rerank_snippet_chars: int = 1200

    """
    6️⃣ rerank_snippet_chars: int = 1200
    📌 Ý nghĩa

    Giới hạn số ký tự mỗi document khi gửi cho LLM rerank.

    → Tránh vượt context
    → Tối ưu cost

    📌 Thường dùng 800–1500 là hợp lý.
    """

    debug_rerank: bool = True
    """
    7️⃣ debug_rerank: bool = True
    📌 Ý nghĩa

    In log chi tiết khi rerank:

    score

    lý do chọn

    ranking

    👉 Dùng khi tuning, tắt khi production.
    """

    topk_router = 20
    """
    8️⃣ topk_router = 20
    📌 Ý nghĩa

    Số document tối đa dùng để:

    phân tích

    quyết định route

    đánh giá độ tự tin

    ⚠️ Không phải số doc đưa vào LLM
    """

    max_source_chars_per_call = 12000
    """
    9️⃣ max_source_chars_per_call = 12000
    📌 Ý nghĩa

    Giới hạn tổng ký tự context đưa cho LLM trong 1 lần gọi.

    Vai trò cực quan trọng:

    Tránh vượt context window

    Tránh LLM “ngợp dữ liệu”

    Giữ latency ổn định

    📌 Với GPT-4/4o → 12k chars là an toàn.
    """
    max_ctx_strict: int = 32
    max_ctx_soft: int = 24

    # # GIẢM RẤT MẠNH
    # max_ctx_strict: int = 20
    # max_ctx_soft: int = 16

    # Dành riêng cho câu hỏi dạng listing
    max_ctx_listing: int = 12

    # Dành cho câu hỏi hỏi đáp cần reasoning
    max_ctx_reasoning: int = 24

    """
    🔟 max_ctx_strict: int = 16
    🔟 max_ctx_soft: int = 12
    📌 Ý nghĩa

    Số lượng document tối đa được đưa vào prompt:

    Mode	Số doc	Ý nghĩa
    STRICT	32	Tin dữ liệu, cần nhiều nguồn
    SOFT	24	Ưu tiên trả lời ngắn gọn

    📌 Điều này thể hiện bạn hiểu rõ RAG không phải càng nhiều context càng tốt.
    """

    code_boost_direct: bool = True

    """
    1️⃣1️⃣ code_boost_direct: bool = True
    📌 Ý nghĩa

    Nếu query chứa:

    mã sản phẩm

    mã thuốc

    ký hiệu kỹ thuật

    → ưu tiên DIRECT_DOC

    Tác dụng:

    Không cho LLM suy luận

    Trả thẳng tài liệu gốc

    Tránh bịa thông tin

    👉 Đây là best practice trong RAG cho dữ liệu kỹ thuật.
    """