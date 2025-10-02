#!/usr/bin/env python3
"""
Generate complete Vietnamese NL2SQL sample queries dataset
Based on the normalized Tiki database schema with 300 queries (100 per complexity)
"""

import json
from datetime import datetime

def generate_simple_queries():
    """Generate 100 simple queries - basic product searches"""
    queries = []
    
    # Base products for simple searches
    products = [
        'áo thun', 'giày', 'túi xách', 'balo', 'vali', 'ví', 'dép', 'nón', 'thắt lưng', 'kính',
        'đồng hồ', 'vớ', 'khăn', 'găng tay', 'áo khoác', 'quần', 'váy', 'áo sơ mi', 'giày thể thao',
        'giày cao gót', 'boot', 'túi đeo chéo', 'cặp sách', 'túi laptop', 'phụ kiện', 'trang sức',
        'mũ lưỡi trai', 'mũ beret', 'áo len', 'hoodie', 'quần jean', 'quần short', 'áo polo',
        'tank top', 'đầm', 'chân váy', 'blazer', 'vest', 'đồ lót', 'đồ ngủ', 'đồ bơi',
        'áo cardigan', 'áo croptop', 'áo bomber', 'áo denim', 'áo flannel', 'áo kimono',
        'quần tây', 'quần baggy', 'quần cargo', 'quần legging', 'quần culottes', 'quần palazzo',
        'váy midi', 'váy mini', 'váy maxi', 'váy bodycon', 'váy swing', 'váy wrap',
        'giày oxford', 'giày loafer', 'giày moccasin', 'giày slip-on', 'giày combat', 'giày chelsea',
        'túi clutch', 'túi tote', 'túi bucket', 'túi saddle', 'túi hobo', 'túi messenger',
        'mũ fedora', 'mũ bucket', 'mũ panama', 'mũ newsboy', 'mũ trucker', 'mũ beanie',
        'kính cận', 'kính mát', 'kính gọng tròn', 'kính aviator', 'kính wayfare', 'kính cat-eye',
        'đồng hồ thông minh', 'đồng hồ cơ', 'đồng hồ điện tử', 'đồng hồ thể thao', 'đồng hồ nữ trang',
        'thắt lưng da', 'thắt lưng vải', 'thắt lưng xích', 'thắt lưng kim loại'
    ]
    
    # Query variations with different verbs
    verbs = [
        ('Tìm', 'Find', 'SELECT * FROM products WHERE name LIKE'),
        ('Hiển thị', 'Show', 'SELECT * FROM products WHERE name LIKE'),
        ('Xem', 'View', 'SELECT * FROM products WHERE name LIKE'),
        ('Liệt kê', 'List', 'SELECT name FROM products WHERE name LIKE'),
        ('Tìm kiếm', 'Search', 'SELECT * FROM products WHERE name LIKE'),
    ]
    
    # Generate product search queries
    query_id = 1
    for product in products[:80]:  # Use first 80 products
        verb_vn, verb_en, sql_start = verbs[(query_id - 1) % len(verbs)]
        queries.append({
            "id": f"simple_{query_id:03d}",
            "vietnamese": f"{verb_vn} {product}",
            "english": f"{verb_en} {product}",
            "sql": f"{sql_start} '%{product}%' LIMIT 10;",
            "complexity": "simple"
        })
        query_id += 1
    
    # Add basic database queries
    basic_queries = [
        ('Tìm tất cả sản phẩm', 'Find all products', 'SELECT * FROM products LIMIT 10;'),
        ('Hiển thị tên sản phẩm', 'Show product names', 'SELECT name FROM products LIMIT 10;'),
        ('Đếm số sản phẩm', 'Count products', 'SELECT COUNT(*) as total_products FROM products;'),
        ('Xem danh sách thương hiệu', 'View brand list', 'SELECT DISTINCT brand_name FROM brands;'),
        ('Hiển thị danh mục', 'Show categories', 'SELECT category_name FROM categories;'),
        ('Tìm sản phẩm đầu tiên', 'Find first product', 'SELECT * FROM products ORDER BY product_id LIMIT 1;'),
        ('Xem 5 sản phẩm', 'View 5 products', 'SELECT * FROM products LIMIT 5;'),
        ('Hiển thị tên và giá', 'Show name and price', 'SELECT p.name, pr.current_price FROM products p JOIN product_pricing pr ON p.product_id = pr.product_id LIMIT 10;'),
        ('Đếm thương hiệu', 'Count brands', 'SELECT COUNT(*) as total_brands FROM brands;'),
        ('Xem người bán', 'View sellers', 'SELECT seller_name FROM sellers LIMIT 10;'),
        ('Xem 10 sản phẩm', 'View 10 products', 'SELECT * FROM products LIMIT 10;'),
        ('Xem 20 sản phẩm', 'View 20 products', 'SELECT * FROM products LIMIT 20;'),
        ('Sản phẩm mới nhất', 'Latest products', 'SELECT * FROM products ORDER BY product_id DESC LIMIT 10;'),
        ('Sản phẩm cũ nhất', 'Oldest products', 'SELECT * FROM products ORDER BY product_id ASC LIMIT 10;'),
        ('Danh sách tất cả danh mục', 'List all categories', 'SELECT * FROM categories;'),
        ('Danh sách tất cả thương hiệu', 'List all brands', 'SELECT * FROM brands;'),
        ('Danh sách người bán', 'List sellers', 'SELECT * FROM sellers LIMIT 10;'),
        ('Tổng số danh mục', 'Total categories', 'SELECT COUNT(*) as total FROM categories;'),
        ('Tổng số người bán', 'Total sellers', 'SELECT COUNT(*) as total FROM sellers;'),
        ('Sản phẩm có hình ảnh', 'Products with images', 'SELECT * FROM products WHERE number_of_images > 0 LIMIT 10;')
    ]
    
    for vn, en, sql in basic_queries:
        if query_id <= 100:
            queries.append({
                "id": f"simple_{query_id:03d}",
                "vietnamese": vn,
                "english": en,
                "sql": sql,
                "complexity": "simple"
            })
            query_id += 1
    
    return queries[:100]

def generate_medium_queries():
    """Generate 100 medium queries - JOINs, price filters, brand searches"""
    queries = []
    
    # Base medium queries
    base_queries = [
        ('Sản phẩm theo thương hiệu', 'Products by brand', 'SELECT b.brand_name, COUNT(p.product_id) as product_count FROM brands b JOIN products p ON b.brand_id = p.brand_id GROUP BY b.brand_name ORDER BY product_count DESC;'),
        ('Giá trung bình theo danh mục', 'Average price by category', 'SELECT c.category_name, AVG(pr.current_price) as avg_price FROM categories c JOIN products p ON c.category_id = p.category_id JOIN product_pricing pr ON p.product_id = pr.product_id GROUP BY c.category_name;'),
        ('Sản phẩm có đánh giá cao', 'High rated products', 'SELECT p.name, rv.rating_average FROM products p JOIN product_reviews rv ON p.product_id = rv.product_id WHERE rv.rating_average >= 4.0 ORDER BY rv.rating_average DESC LIMIT 20;'),
        ('Thương hiệu Nike', 'Nike brand products', "SELECT p.name, pr.current_price FROM products p JOIN brands b ON p.brand_id = b.brand_id JOIN product_pricing pr ON p.product_id = pr.product_id WHERE b.brand_name LIKE '%Nike%' LIMIT 10;"),
        ('Sản phẩm giá dưới 500k', 'Products under 500k', 'SELECT p.name, pr.current_price FROM products p JOIN product_pricing pr ON p.product_id = pr.product_id WHERE pr.current_price < 500000 ORDER BY pr.current_price LIMIT 20;'),
    ]
    
    query_id = 1
    for vn, en, sql in base_queries:
        queries.append({
            "id": f"medium_{query_id:03d}",
            "vietnamese": vn,
            "english": en,
            "sql": sql,
            "complexity": "medium"
        })
        query_id += 1
    
    # Price range variations
    price_ranges = [100000, 200000, 300000, 500000, 1000000, 2000000]
    for price in price_ranges:
        queries.extend([
            {
                "id": f"medium_{query_id:03d}",
                "vietnamese": f'Sản phẩm giá dưới {price//1000}k',
                "english": f'Products under {price//1000}k',
                "sql": f'SELECT p.name, pr.current_price FROM products p JOIN product_pricing pr ON p.product_id = pr.product_id WHERE pr.current_price < {price} ORDER BY pr.current_price LIMIT 20;',
                "complexity": "medium"
            },
            {
                "id": f"medium_{query_id+1:03d}",
                "vietnamese": f'Sản phẩm giá trên {price//1000}k',
                "english": f'Products over {price//1000}k',
                "sql": f'SELECT p.name, pr.current_price FROM products p JOIN product_pricing pr ON p.product_id = pr.product_id WHERE pr.current_price > {price} ORDER BY pr.current_price DESC LIMIT 20;',
                "complexity": "medium"
            }
        ])
        query_id += 2
    
    # Brand variations
    brands = ['Nike', 'Adidas', 'Samsung', 'Apple', 'Louis Vuitton', 'Gucci', 'Uniqlo', 'Zara']
    for brand in brands:
        if query_id <= 100:
            queries.append({
                "id": f"medium_{query_id:03d}",
                "vietnamese": f'Sản phẩm thương hiệu {brand}',
                "english": f'{brand} brand products',
                "sql": f"SELECT p.name, pr.current_price FROM products p JOIN brands b ON p.brand_id = b.brand_id JOIN product_pricing pr ON p.product_id = pr.product_id WHERE b.brand_name LIKE '%{brand}%' LIMIT 10;",
                "complexity": "medium"
            })
            query_id += 1
    
    # Rating variations
    while query_id <= 100:
        rating = 3.5 + (query_id % 10) * 0.1
        queries.append({
            "id": f"medium_{query_id:03d}",
            "vietnamese": f'Sản phẩm có đánh giá trên {rating:.1f} sao',
            "english": f'Products with rating above {rating:.1f} stars',
            "sql": f'SELECT p.name, rv.rating_average FROM products p JOIN product_reviews rv ON p.product_id = rv.product_id WHERE rv.rating_average >= {rating:.1f} ORDER BY rv.rating_average DESC LIMIT 15;',
            "complexity": "medium"
        })
        query_id += 1
    
    return queries[:100]

def generate_complex_queries():
    """Generate 100 complex queries - aggregations, rankings, market analysis"""
    queries = []
    
    # Base complex queries
    base_queries = [
        {
            "vietnamese": "Top 10 sản phẩm đánh giá cao nhất có giá dưới 1 triệu",
            "english": "Top 10 highest rated products under 1 million",
            "sql": """SELECT p.name, b.brand_name, c.category_name, pr.current_price, rv.rating_average, rv.review_count 
FROM products p 
JOIN brands b ON p.brand_id = b.brand_id 
JOIN categories c ON p.category_id = c.category_id 
JOIN product_pricing pr ON p.product_id = pr.product_id 
JOIN product_reviews rv ON p.product_id = rv.product_id 
WHERE pr.current_price < 1000000 AND rv.rating_average >= 4.0 
ORDER BY rv.rating_average DESC, rv.review_count DESC 
LIMIT 10;"""
        },
        {
            "vietnamese": "Phân tích thị phần thương hiệu",
            "english": "Brand market share analysis",
            "sql": """SELECT b.brand_name, 
COUNT(p.product_id) as product_count,
ROUND(COUNT(p.product_id) * 100.0 / (SELECT COUNT(*) FROM products), 2) as market_share_percent,
AVG(pr.current_price) as avg_price,
AVG(rv.rating_average) as avg_rating
FROM brands b 
JOIN products p ON b.brand_id = p.brand_id 
JOIN product_pricing pr ON p.product_id = pr.product_id 
JOIN product_reviews rv ON p.product_id = rv.product_id 
GROUP BY b.brand_id, b.brand_name 
HAVING COUNT(p.product_id) >= 10
ORDER BY market_share_percent DESC 
LIMIT 20;"""
        }
    ]
    
    query_id = 1
    for query_data in base_queries:
        queries.append({
            "id": f"complex_{query_id:03d}",
            "vietnamese": query_data["vietnamese"],
            "english": query_data["english"],
            "sql": query_data["sql"],
            "complexity": "complex"
        })
        query_id += 1
    
    # Complex template variations
    categories = ['Phụ kiện thời trang', 'Giày dép nam', 'Túi nam', 'Balo & Vali']
    brands_pairs = [('Nike', 'Adidas'), ('Samsung', 'Apple'), ('Louis Vuitton', 'Gucci')]
    
    # Top N products in category
    for i in range(30):
        n = 5 + (i % 10)
        category = categories[i % len(categories)]
        queries.append({
            "id": f"complex_{query_id:03d}",
            "vietnamese": f"Top {n} sản phẩm bán chạy nhất trong danh mục {category}",
            "english": f"Top {n} best selling products in {category} category",
            "sql": f"""SELECT p.name, b.brand_name, pr.current_price, rv.rating_average, pr.quantity_sold
FROM products p 
JOIN brands b ON p.brand_id = b.brand_id 
JOIN categories c ON p.category_id = c.category_id 
JOIN product_pricing pr ON p.product_id = pr.product_id 
JOIN product_reviews rv ON p.product_id = rv.product_id 
WHERE c.category_name = '{category}'
ORDER BY pr.quantity_sold DESC, rv.rating_average DESC 
LIMIT {n};""",
            "complexity": "complex"
        })
        query_id += 1
    
    # Brand comparison queries
    for i in range(30):
        brand1, brand2 = brands_pairs[i % len(brands_pairs)]
        rating = 4.0 + (i % 5) * 0.1
        queries.append({
            "id": f"complex_{query_id:03d}",
            "vietnamese": f"Sản phẩm {brand1} hoặc {brand2} có đánh giá trên {rating} sao",
            "english": f"{brand1} or {brand2} products with rating above {rating} stars",
            "sql": f"""SELECT p.name, b.brand_name, c.category_name, pr.current_price, rv.rating_average
FROM products p 
JOIN brands b ON p.brand_id = b.brand_id 
JOIN categories c ON p.category_id = c.category_id 
JOIN product_pricing pr ON p.product_id = pr.product_id 
JOIN product_reviews rv ON p.product_id = rv.product_id 
WHERE (b.brand_name LIKE '%{brand1}%' OR b.brand_name LIKE '%{brand2}%') 
AND rv.rating_average >= {rating}
ORDER BY rv.rating_average DESC, pr.current_price ASC 
LIMIT 15;""",
            "complexity": "complex"
        })
        query_id += 1
    
    # Price analysis queries
    for i in range(36):
        min_price = 100 + (i % 10) * 100
        max_price = min_price + 500
        queries.append({
            "id": f"complex_{query_id:03d}",
            "vietnamese": f"Phân tích giá theo khoảng từ {min_price}k đến {max_price}k",
            "english": f"Price analysis from {min_price}k to {max_price}k range",
            "sql": f"""SELECT 
CASE 
    WHEN pr.current_price < {min_price * 1000} THEN 'Under {min_price}K'
    WHEN pr.current_price BETWEEN {min_price * 1000} AND {max_price * 1000} THEN '{min_price}K-{max_price}K'
    ELSE 'Over {max_price}K'
END as price_range,
COUNT(*) as product_count,
AVG(rv.rating_average) as avg_rating,
AVG(pr.current_price) as avg_price
FROM product_pricing pr 
JOIN product_reviews rv ON pr.product_id = rv.product_id 
GROUP BY price_range 
ORDER BY avg_price;""",
            "complexity": "complex"
        })
        query_id += 1
    
    return queries[:100]

def main():
    """Generate complete dataset and save to JSON"""
    
    print("Generating Vietnamese NL2SQL Complete Dataset...")
    
    # Generate all queries
    simple_queries = generate_simple_queries()
    medium_queries = generate_medium_queries()
    complex_queries = generate_complex_queries()
    
    # Create complete dataset
    dataset = {
        "metadata": {
            "title": "Vietnamese NL2SQL Sample Queries Dataset - Complete",
            "description": "Complete structured sample queries for Vietnamese to SQL translation testing based on normalized Tiki database schema",
            "version": "2.0.0",
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "total_queries": 300,
            "categories": {
                "simple": 100,
                "medium": 100,
                "complex": 100
            },
            "database_schema": {
                "tables": ["products", "brands", "categories", "sellers", "product_pricing", "product_reviews"],
                "relationships": "Normalized schema with proper foreign key relationships for complex JOIN operations"
            },
            "structure": {
                "simple": "Basic product searches, single table queries, basic filtering",
                "medium": "Brand/category searches with JOINs, price filtering, rating filters",
                "complex": "Multi-table JOINs, aggregations, rankings, market analysis, statistical queries"
            }
        },
        "simple": simple_queries,
        "medium": medium_queries,
        "complex": complex_queries
    }
    
    # Save to JSON file
    output_file = "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/frontend/public/data/sample_queries_complete_300.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Complete dataset generated successfully!")
    print(f"📁 File: {output_file}")
    print(f"📊 Total queries: {len(simple_queries) + len(medium_queries) + len(complex_queries)}")
    print(f"   - Simple: {len(simple_queries)}")
    print(f"   - Medium: {len(medium_queries)}")
    print(f"   - Complex: {len(complex_queries)}")

if __name__ == "__main__":
    main()
