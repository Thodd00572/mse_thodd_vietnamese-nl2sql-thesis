#!/usr/bin/env python3
"""
Vietnamese Test Queries for NL2SQL Pipeline Testing
Based on complexity levels: Simple, Medium, Complex
"""

# SIMPLE QUERIES - Basic keyword matching, compound words, numeric values
SIMPLE_QUERIES = [
    # Basic keyword matching
    "tìm áo thun",
    "tìm giày",
    "tìm balo",
    "tìm túi xách",
    "tìm ví",
    "tìm dép",
    "tìm sandal",
    "tìm sneaker",
    "tìm boot",
    "tìm phụ kiện",
    
    # Compound words (thể thao)
    "giày thể thao nam",
    "áo thể thao nữ",
    "túi thể thao",
    "balo thể thao",
    "dép thể thao",
    "giày thể thao trắng",
    "áo thể thao đen",
    "phụ kiện thể thao",
    "giày thể thao Adidas",
    "túi thể thao Nike",
    
    # Numeric value extraction
    "váy giá dưới 500k",
    "giày giá dưới 300k",
    "túi giá dưới 200k",
    "balo giá dưới 150k",
    "áo giá dưới 100k",
    "giày giá dưới 1 triệu",
    "túi giá dưới 500 nghìn",
    "phụ kiện giá dưới 50k",
    "dép giá dưới 200 nghìn",
    "ví giá dưới 100 nghìn",
    
    # Basic color and material
    "giày màu đen",
    "túi màu nâu",
    "áo màu trắng",
    "balo màu xanh",
    "dép màu đỏ",
    "giày da",
    "túi da",
    "áo cotton",
    "balo vải",
    "dép nhựa",
    
    # Basic brand queries
    "giày Nike",
    "túi Gucci",
    "áo Adidas",
    "balo Jansport",
    "giày Converse",
    "túi Louis Vuitton",
    "áo Uniqlo",
    "giày Vans",
    "balo Herschel",
    "dép Havaianas"
]

# MEDIUM QUERIES - Multiple conditions (AND), compound words, logical OR
MEDIUM_QUERIES = [
    # Multiple conditions (AND) with compound words
    "áo khoác nam màu đen có hơn 100 lượt đánh giá",
    "giày thể thao nữ màu trắng giá dưới 500k",
    "túi xách nữ da thật có đánh giá trên 4 sao",
    "balo laptop nam chống nước giá dưới 1 triệu",
    "giày cao gót nữ màu đỏ có hơn 50 lượt bán",
    "áo sơ mi nam trắng có đánh giá cao",
    "túi đeo chéo nữ da bò giá hợp lý",
    "giày sneaker nam trắng có nhiều size",
    "áo hoodie nữ màu xám có hood",
    "balo du lịch lớn chống thấm nước",
    
    # Logical OR conditions
    "tìm áo sơ mi hoặc quần jean cho nam",
    "giày thể thao Nike hoặc Adidas cho nữ",
    "túi xách Gucci hoặc Louis Vuitton",
    "balo laptop hoặc túi đựng máy tính",
    "giày cao gót hoặc giày búp bê nữ",
    "áo khoác denim hoặc áo khoác da",
    "dép sandal hoặc dép tông nam",
    "túi đeo vai hoặc túi đeo chéo",
    "giày boot hoặc giày oxford nam",
    "áo thun hoặc áo polo nam",
    
    # Price range conditions
    "giày nam giá từ 500k đến 1 triệu",
    "túi nữ giá từ 200k đến 800k",
    "áo khoác giá từ 300k đến 1.5 triệu",
    "balo giá từ 150k đến 600k",
    "giày cao gót giá từ 400k đến 1.2 triệu",
    "phụ kiện thời trang giá từ 50k đến 300k",
    "giày thể thao giá từ 800k đến 2 triệu",
    "túi xách da giá từ 1 triệu đến 3 triệu",
    "áo sơ mi giá từ 200k đến 700k",
    "dép sandal giá từ 100k đến 500k",
    
    # Rating and review conditions
    "giày có đánh giá trên 4.5 sao",
    "túi có hơn 200 lượt đánh giá",
    "áo có rating cao nhất",
    "balo được yêu thích nhiều nhất",
    "giày có nhiều đánh giá tích cực",
    "túi có số sao cao",
    "áo được khách hàng ưa chuộng",
    "phụ kiện có đánh giá tốt",
    "giày có feedback tích cực",
    "balo có rating trên 4 sao",
    
    # Seller and brand combinations
    "giày Nike từ seller uy tín",
    "túi Gucci chính hãng",
    "áo Adidas giá tốt",
    "balo Jansport chất lượng",
    "giày Converse màu đen",
    "túi Coach da thật",
    "áo Uniqlo cotton",
    "giày Vans classic",
    "balo Herschel vintage",
    "dép Birkenstock chính hãng"
]

# COMPLEX QUERIES - Aggregation (MAX, COUNT), JOIN operations, GROUP BY, HAVING
COMPLEX_QUERIES = [
    # Aggregation (MAX) with calculated fields
    "sản phẩm nào được giảm giá nhiều nhất",
    "túi nào có giá cao nhất trong danh mục nữ",
    "giày nào có số lượng bán chạy nhất",
    "áo nào có đánh giá cao nhất",
    "balo nào có nhiều hình ảnh nhất",
    "phụ kiện nào được yêu thích nhiều nhất",
    "sản phẩm nào có cashback cao nhất",
    "giày nào có review nhiều nhất",
    "túi nào có rating trung bình cao nhất",
    "áo nào có giá gốc cao nhất",
    
    # JOIN, ORDER BY, LIMIT with nested logic
    "tìm 5 áo thun nam có đánh giá cao nhất trong danh mục Thời Trang Nam",
    "top 10 giày nữ bán chạy nhất có giá dưới 1 triệu",
    "5 túi xách nữ đắt nhất có đánh giá trên 4 sao",
    "10 balo laptop tốt nhất theo đánh giá khách hàng",
    "top 3 giày thể thao nam Nike có nhiều size nhất",
    "5 áo khoác nữ được yêu thích nhất mùa đông này",
    "top 7 phụ kiện thời trang trending nhất",
    "10 sản phẩm giảm giá sâu nhất hôm nay",
    "5 giày cao gót nữ sang trọng nhất",
    "top 8 túi đeo chéo nam phong cách nhất",
    
    # COUNT aggregation with range conditions
    "có bao nhiều sản phẩm của nữ có giá từ 200k đến 500k",
    "đếm số lượng giày thể thao nam có đánh giá trên 4 sao",
    "có bao nhiều túi xách da thật giá dưới 1 triệu",
    "tổng số áo khoác nữ có trong kho",
    "đếm số sản phẩm Nike có giá trên 500k",
    "có bao nhiều balo laptop chống nước",
    "số lượng giày cao gót có size 37",
    "đếm phụ kiện thời trang có giá dưới 100k",
    "có bao nhiều sản phẩm Adidas đang sale",
    "tổng số túi xách nữ màu đen",
    
    # GROUP BY, HAVING with aggregation (AVG)
    "hiển thị các loại áo có giá trung bình trên 300k",
    "thống kê số lượng sản phẩm theo từng thương hiệu",
    "các danh mục nào có giá trung bình cao nhất",
    "thương hiệu nào có nhiều sản phẩm nhất",
    "loại giày nào có đánh giá trung bình tốt nhất",
    "danh mục nào có sản phẩm bán chạy nhất",
    "seller nào có nhiều sản phẩm chất lượng nhất",
    "thống kê giá trung bình theo từng loại túi",
    "các thương hiệu có rating trung bình trên 4.5",
    "danh mục nào có sản phẩm đắt nhất",
    
    # Complex filtering with multiple aggregations
    "sản phẩm nào có tỷ lệ giảm giá cao nhất so với giá gốc",
    "thương hiệu nào có sản phẩm với cashback trung bình cao nhất",
    "danh mục nào có sản phẩm được review nhiều nhất",
    "seller nào có sản phẩm với rating trung bình cao nhất",
    "loại sản phẩm nào có số lượng hình ảnh trung bình nhiều nhất",
    "thống kê sản phẩm có video theo từng danh mục",
    "các sản phẩm có tỷ lệ yêu thích cao nhất",
    "thương hiệu nào có sản phẩm bán chạy nhất tháng này",
    "danh mục nào có sản phẩm với fulfillment type tốt nhất",
    "phân tích xu hướng giá theo từng loại sản phẩm"
]

def print_all_queries():
    """Print all queries organized by complexity"""
    print("=== SIMPLE QUERIES (50) ===")
    for i, query in enumerate(SIMPLE_QUERIES, 1):
        print(f"{i:2d}. {query}")
    
    print("\n=== MEDIUM QUERIES (50) ===")
    for i, query in enumerate(MEDIUM_QUERIES, 1):
        print(f"{i:2d}. {query}")
    
    print("\n=== COMPLEX QUERIES (50) ===")
    for i, query in enumerate(COMPLEX_QUERIES, 1):
        print(f"{i:2d}. {query}")
    
    print(f"\nTotal queries: {len(SIMPLE_QUERIES) + len(MEDIUM_QUERIES) + len(COMPLEX_QUERIES)}")

if __name__ == "__main__":
    print_all_queries()
