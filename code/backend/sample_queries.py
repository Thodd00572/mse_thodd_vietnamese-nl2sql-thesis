"""
Sample Vietnamese queries for testing pipeline performance across different complexity levels
Based on Table 3.2: Sample Vietnamese Queries by Complexity
"""

SAMPLE_QUERIES = {
    "simple": [
        # Basic keyword matching
        {"query": "tìm áo thun", "challenge": "Basic keyword matching", "expected_sql_type": "SELECT with LIKE"},
        {"query": "giày thể thao nam", "challenge": "Compound word (thể thao)", "expected_sql_type": "SELECT with category filter"},
        {"query": "váy giá dưới 500k", "challenge": "Numeric value extraction", "expected_sql_type": "SELECT with price condition"},
        {"query": "tìm điện thoại", "challenge": "Basic keyword matching", "expected_sql_type": "SELECT with category filter"},
        {"query": "laptop Dell", "challenge": "Brand keyword matching", "expected_sql_type": "SELECT with brand filter"},
        {"query": "tai nghe Apple", "challenge": "Brand and category", "expected_sql_type": "SELECT with brand and category"},
        {"query": "tìm iPhone", "challenge": "Product name matching", "expected_sql_type": "SELECT with name LIKE"},
        {"query": "Samsung Galaxy", "challenge": "Brand and product series", "expected_sql_type": "SELECT with name LIKE"},
        {"query": "máy tính bảng", "challenge": "Compound product name", "expected_sql_type": "SELECT with category filter"},
        {"query": "chuột máy tính", "challenge": "Compound accessory name", "expected_sql_type": "SELECT with category filter"},
        {"query": "bàn phím cơ", "challenge": "Product type with modifier", "expected_sql_type": "SELECT with name LIKE"},
        {"query": "ốp lưng điện thoại", "challenge": "Accessory with device type", "expected_sql_type": "SELECT with category filter"},
        {"query": "sạc dự phòng", "challenge": "Compound accessory name", "expected_sql_type": "SELECT with name LIKE"},
        {"query": "loa bluetooth", "challenge": "Technology specification", "expected_sql_type": "SELECT with name LIKE"},
        {"query": "đồng hồ thông minh", "challenge": "Smart device category", "expected_sql_type": "SELECT with category filter"},
        {"query": "camera hành trình", "challenge": "Specific device type", "expected_sql_type": "SELECT with name LIKE"},
        {"query": "pin sạc dự phòng", "challenge": "Extended compound name", "expected_sql_type": "SELECT with name LIKE"},
        {"query": "kính cường lực", "challenge": "Material specification", "expected_sql_type": "SELECT with name LIKE"},
        {"query": "giá rẻ", "challenge": "Price adjective", "expected_sql_type": "SELECT with price ORDER BY"},
        {"query": "hàng mới", "challenge": "Condition adjective", "expected_sql_type": "SELECT with date filter"}
    ],
    
    "medium": [
        # Multiple conditions and logical operations
        {"query": "áo khoác nam màu đen có hơn 100 lượt đánh giá", "challenge": "Multiple conditions (AND), compound word", "expected_sql_type": "SELECT with AND conditions"},
        {"query": "tìm áo sơ mi hoặc quần jean cho nam", "challenge": "Logical OR condition", "expected_sql_type": "SELECT with OR conditions"},
        {"query": "điện thoại Samsung giá từ 5 triệu đến 15 triệu", "challenge": "Range condition with brand", "expected_sql_type": "SELECT with BETWEEN and brand"},
        {"query": "laptop có RAM 8GB và SSD 256GB", "challenge": "Multiple technical specifications", "expected_sql_type": "SELECT with multiple LIKE conditions"},
        {"query": "tai nghe không dây có đánh giá trên 4 sao", "challenge": "Technology type with rating condition", "expected_sql_type": "SELECT with rating filter"},
        {"query": "tìm giày nam size 42 màu đen hoặc nâu", "challenge": "Size specification with color options", "expected_sql_type": "SELECT with size and color OR"},
        {"query": "áo thun nữ giá dưới 200k có nhiều màu", "challenge": "Gender, price, and variety condition", "expected_sql_type": "SELECT with multiple conditions"},
        {"query": "điện thoại Apple hoặc Samsung có camera tốt", "challenge": "Brand OR with feature condition", "expected_sql_type": "SELECT with brand OR and feature"},
        {"query": "laptop gaming có card đồ họa rời", "challenge": "Category with technical specification", "expected_sql_type": "SELECT with category and specs"},
        {"query": "đồng hồ nam dây da giá từ 1 triệu", "challenge": "Gender, material, and price range", "expected_sql_type": "SELECT with multiple filters"},
        {"query": "túi xách nữ thương hiệu nổi tiếng", "challenge": "Gender with brand reputation", "expected_sql_type": "SELECT with brand filter"},
        {"query": "giày thể thao chạy bộ có đệm khí", "challenge": "Activity type with technology feature", "expected_sql_type": "SELECT with activity and feature"},
        {"query": "máy tính bảng Android hoặc iOS", "challenge": "Device with OS options", "expected_sql_type": "SELECT with OS OR condition"},
        {"query": "camera DSLR Canon hoặc Nikon dưới 20 triệu", "challenge": "Brand OR with price limit", "expected_sql_type": "SELECT with brand OR and price"},
        {"query": "loa di động chống nước có bluetooth", "challenge": "Portability with multiple features", "expected_sql_type": "SELECT with multiple feature conditions"},
        {"query": "balo laptop có ngăn chống sốc", "challenge": "Purpose-specific with protection feature", "expected_sql_type": "SELECT with specific features"},
        {"query": "kem chống nắng SPF 50 cho da nhạy cảm", "challenge": "Specification with skin type", "expected_sql_type": "SELECT with SPF and skin type"},
        {"query": "nước hoa nam mùi gỗ hoặc mùi tươi mát", "challenge": "Gender with scent type options", "expected_sql_type": "SELECT with scent OR condition"},
        {"query": "sách tiếng Anh cho trẻ em dưới 10 tuổi", "challenge": "Language with age range", "expected_sql_type": "SELECT with language and age"},
        {"query": "đèn LED để bàn có thể điều chỉnh độ sáng", "challenge": "Product type with adjustable feature", "expected_sql_type": "SELECT with adjustable feature"}
    ],
    
    "complex": [
        # Aggregation, JOIN, ORDER BY, LIMIT, nested logic
        {"query": "sản phẩm nào được giảm giá nhiều nhất", "challenge": "Aggregation (MAX), calculated field", "expected_sql_type": "SELECT with MAX aggregation"},
        {"query": "tìm 5 áo thun nam có đánh giá cao nhất trong danh mục Thời Trang Nam", "challenge": "JOIN, ORDER BY, LIMIT, nested logic", "expected_sql_type": "SELECT with JOIN, ORDER BY, LIMIT"},
        {"query": "có bao nhiều sản phẩm của nữ có giá từ 200k đến 500k", "challenge": "COUNT aggregation with range condition", "expected_sql_type": "SELECT COUNT with BETWEEN"},
        {"query": "hiển thị các loại áo có giá trung bình trên 300k", "challenge": "GROUP BY, HAVING with aggregation (AVG)", "expected_sql_type": "SELECT with GROUP BY, HAVING"},
        {"query": "top 10 thương hiệu có nhiều sản phẩm nhất", "challenge": "GROUP BY, COUNT, ORDER BY, LIMIT", "expected_sql_type": "SELECT with GROUP BY, COUNT, ORDER BY"},
        {"query": "sản phẩm nào có số lượng đánh giá cao nhất trong từng danh mục", "challenge": "Window function or subquery with MAX", "expected_sql_type": "SELECT with subquery or window function"},
        {"query": "tìm áo khoác nam có giá gần với giá trung bình của danh mục", "challenge": "Subquery with AVG calculation", "expected_sql_type": "SELECT with subquery AVG"},
        {"query": "danh sách 3 sản phẩm rẻ nhất và 3 sản phẩm đắt nhất", "challenge": "UNION with ORDER BY and LIMIT", "expected_sql_type": "SELECT with UNION"},
        {"query": "thương hiệu nào có tỷ lệ sản phẩm giảm giá cao nhất", "challenge": "Calculated percentage with GROUP BY", "expected_sql_type": "SELECT with calculated percentage"},
        {"query": "sản phẩm có rating cao hơn trung bình của cùng danh mục", "challenge": "Correlated subquery with AVG", "expected_sql_type": "SELECT with correlated subquery"},
        {"query": "tìm các cặp sản phẩm cùng thương hiệu có giá chênh lệch lớn nhất", "challenge": "Self-join with MAX difference", "expected_sql_type": "SELECT with self-join"},
        {"query": "danh mục nào có sự đa dạng giá cao nhất (độ lệch chuẩn)", "challenge": "Statistical function (STDDEV)", "expected_sql_type": "SELECT with statistical aggregation"},
        {"query": "sản phẩm bán chạy nhất theo từng khoảng giá 0-1tr, 1-5tr, 5tr+", "challenge": "CASE WHEN with GROUP BY", "expected_sql_type": "SELECT with CASE WHEN grouping"},
        {"query": "tỷ lệ sản phẩm có đánh giá trên 4 sao của mỗi thương hiệu", "challenge": "Conditional aggregation with percentage", "expected_sql_type": "SELECT with conditional COUNT"},
        {"query": "sản phẩm có xu hướng giá tăng theo thời gian trong 6 tháng qua", "challenge": "Time series analysis with trend", "expected_sql_type": "SELECT with date functions and trend"},
        {"query": "top 5 combo sản phẩm thường được mua cùng nhau", "challenge": "Association analysis", "expected_sql_type": "Complex query with product associations"},
        {"query": "thương hiệu có mức độ hài lòng khách hàng cao nhất dựa trên rating và review", "challenge": "Weighted scoring with multiple factors", "expected_sql_type": "SELECT with weighted calculation"},
        {"query": "sản phẩm nào có hiệu suất bán hàng tốt nhất theo tỷ lệ rating/giá", "challenge": "Performance ratio calculation", "expected_sql_type": "SELECT with ratio calculation"},
        {"query": "phân tích xu hướng giá của iPhone qua các thế hệ", "challenge": "Time series with product evolution", "expected_sql_type": "SELECT with time series analysis"},
        {"query": "so sánh hiệu suất bán hàng giữa sản phẩm có giảm giá và không giảm giá", "challenge": "Comparative analysis with grouping", "expected_sql_type": "SELECT with comparative grouping"}
    ]
}

def get_all_queries():
    """Get all sample queries organized by complexity"""
    return SAMPLE_QUERIES

def get_queries_by_complexity(complexity: str):
    """Get queries for a specific complexity level"""
    return SAMPLE_QUERIES.get(complexity, [])

def get_total_query_count():
    """Get total number of sample queries"""
    return sum(len(queries) for queries in SAMPLE_QUERIES.values())
