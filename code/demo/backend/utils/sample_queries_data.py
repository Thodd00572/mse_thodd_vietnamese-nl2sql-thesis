# Vietnamese NL2SQL Sample Queries with corresponding SQL
# 100 queries per complexity level for demonstration

def get_sample_queries_data():
    """Returns 300 Vietnamese-SQL query pairs organized by complexity"""
    
    # Generate 100 simple queries
    simple_queries = []
    simple_base_queries = [
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
    ]
    
    # Add product search variations - expanded list
    products = ['áo thun', 'giày', 'túi xách', 'balo', 'vali', 'ví', 'dép', 'nón', 'thắt lưng', 'kính', 
                'đồng hồ', 'vớ', 'khăn', 'găng tay', 'áo khoác', 'quần', 'váy', 'áo sơ mi', 'giày thể thao',
                'giày cao gót', 'boot', 'túi đeo chéo', 'cặp sách', 'túi laptop', 'phụ kiện', 'trang sức',
                'mũ lưỡi trai', 'mũ beret', 'áo len', 'hoodie', 'quần jean', 'quần short', 'áo polo',
                'tank top', 'đầm', 'chân váy', 'blazer', 'vest', 'đồ lót', 'đồ ngủ', 'đồ bơi',
                # Additional products to reach 100 queries
                'áo cardigan', 'áo croptop', 'áo bomber', 'áo denim', 'áo flannel', 'áo kimono',
                'quần tây', 'quần baggy', 'quần cargo', 'quần legging', 'quần culottes', 'quần palazzo',
                'váy midi', 'váy mini', 'váy maxi', 'váy bodycon', 'váy swing', 'váy wrap',
                'giày oxford', 'giày loafer', 'giày moccasin', 'giày slip-on', 'giày combat', 'giày chelsea',
                'túi clutch', 'túi tote', 'túi bucket', 'túi saddle', 'túi hobo', 'túi messenger',
                'mũ fedora', 'mũ bucket', 'mũ panama', 'mũ newsboy', 'mũ trucker', 'mũ beanie',
                'kính cận', 'kính mát', 'kính gọng tròn', 'kính aviator', 'kính wayfare', 'kính cat-eye',
                'đồng hồ thông minh', 'đồng hồ cơ', 'đồng hồ điện tử', 'đồng hồ thể thao', 'đồng hồ nữ trang',
                'thắt lưng da', 'thắt lưng vải', 'thắt lưng xích', 'thắt lưng kim loại']
    
    # Add basic product searches
    for i, product in enumerate(products):
        if len(simple_queries) >= 85:  # Leave room for base queries and variations
            break
        simple_queries.append({
            'vietnamese': f'Tìm {product}',
            'english': f'Find {product}',
            'sql': f"SELECT * FROM products WHERE name LIKE '%{product}%' LIMIT 10;",
            'complexity': 'simple'
        })
    
    # Add query variations with different verbs
    query_variations = [
        ('Hiển thị', 'Show', 'SELECT * FROM products WHERE name LIKE'),
        ('Xem', 'View', 'SELECT * FROM products WHERE name LIKE'),
        ('Liệt kê', 'List', 'SELECT name FROM products WHERE name LIKE'),
        ('Tìm kiếm', 'Search', 'SELECT * FROM products WHERE name LIKE'),
    ]
    
    # Add variations until we reach 85 queries (leaving 15 for base queries)
    variation_products = ['áo', 'giày', 'túi', 'mũ', 'quần', 'váy', 'đồng hồ', 'kính', 'phụ kiện', 'trang sức']
    for verb_vn, verb_en, sql_start in query_variations:
        for product in variation_products:
            if len(simple_queries) >= 85:
                break
            simple_queries.append({
                'vietnamese': f'{verb_vn} {product}',
                'english': f'{verb_en} {product}',
                'sql': f"{sql_start} '%{product}%' LIMIT 10;",
                'complexity': 'simple'
            })
        if len(simple_queries) >= 85:
            break
    
    # Add the base queries
    for vn, en, sql in simple_base_queries:
        if len(simple_queries) >= 100:
            break
        simple_queries.append({
            'vietnamese': vn,
            'english': en,
            'sql': sql,
            'complexity': 'simple'
        })
    
    # Add more simple queries if still under 100
    additional_simple_queries = [
        ('Xem 10 sản phẩm', 'View 10 products', 'SELECT * FROM products LIMIT 10;'),
        ('Xem 20 sản phẩm', 'View 20 products', 'SELECT * FROM products LIMIT 20;'),
        ('Sản phẩm mới nhất', 'Latest products', 'SELECT * FROM products ORDER BY product_id DESC LIMIT 10;'),
        ('Sản phẩm cũ nhất', 'Oldest products', 'SELECT * FROM products ORDER BY product_id ASC LIMIT 10;'),
        ('Danh sách tất cả danh mục', 'List all categories', 'SELECT * FROM categories;'),
        ('Danh sách tất cả thương hiệu', 'List all brands', 'SELECT * FROM brands;'),
        ('Danh sách người bán', 'List sellers', 'SELECT * FROM sellers LIMIT 10;'),
        ('Tổng số danh mục', 'Total categories', 'SELECT COUNT(*) as total FROM categories;'),
        ('Tổng số người bán', 'Total sellers', 'SELECT COUNT(*) as total FROM sellers;'),
    ]
    
    for vn, en, sql in additional_simple_queries:
        if len(simple_queries) >= 100:
            break
        simple_queries.append({
            'vietnamese': vn,
            'english': en,
            'sql': sql,
            'complexity': 'simple'
        })
    
    # Generate 100 medium queries
    medium_queries = [
        {'vietnamese': 'Sản phẩm theo thương hiệu', 'english': 'Products by brand', 'sql': 'SELECT b.brand_name, COUNT(p.product_id) as product_count FROM brands b JOIN products p ON b.brand_id = p.brand_id GROUP BY b.brand_name ORDER BY product_count DESC;', 'complexity': 'medium'},
        {'vietnamese': 'Giá trung bình theo danh mục', 'english': 'Average price by category', 'sql': 'SELECT c.category_name, AVG(pr.current_price) as avg_price FROM categories c JOIN products p ON c.category_id = p.category_id JOIN product_pricing pr ON p.product_id = pr.product_id GROUP BY c.category_name;', 'complexity': 'medium'},
        {'vietnamese': 'Sản phẩm có đánh giá cao', 'english': 'High rated products', 'sql': 'SELECT p.name, rv.rating_average FROM products p JOIN product_reviews rv ON p.product_id = rv.product_id WHERE rv.rating_average >= 4.0 ORDER BY rv.rating_average DESC LIMIT 20;', 'complexity': 'medium'},
        {'vietnamese': 'Thương hiệu Nike', 'english': 'Nike brand products', 'sql': "SELECT p.name, pr.current_price FROM products p JOIN brands b ON p.brand_id = b.brand_id JOIN product_pricing pr ON p.product_id = pr.product_id WHERE b.brand_name LIKE '%Nike%' LIMIT 10;", 'complexity': 'medium'},
        {'vietnamese': 'Sản phẩm giá dưới 500k', 'english': 'Products under 500k', 'sql': 'SELECT p.name, pr.current_price FROM products p JOIN product_pricing pr ON p.product_id = pr.product_id WHERE pr.current_price < 500000 ORDER BY pr.current_price LIMIT 20;', 'complexity': 'medium'},
    ]
    
    # Add price range variations
    price_ranges = [100000, 200000, 300000, 500000, 1000000, 2000000]
    for price in price_ranges:
        medium_queries.extend([
            {'vietnamese': f'Sản phẩm giá dưới {price//1000}k', 'english': f'Products under {price//1000}k', 'sql': f'SELECT p.name, pr.current_price FROM products p JOIN product_pricing pr ON p.product_id = pr.product_id WHERE pr.current_price < {price} ORDER BY pr.current_price LIMIT 20;', 'complexity': 'medium'},
            {'vietnamese': f'Sản phẩm giá trên {price//1000}k', 'english': f'Products over {price//1000}k', 'sql': f'SELECT p.name, pr.current_price FROM products p JOIN product_pricing pr ON p.product_id = pr.product_id WHERE pr.current_price > {price} ORDER BY pr.current_price DESC LIMIT 20;', 'complexity': 'medium'},
        ])
    
    # Add brand variations
    brands = ['Nike', 'Adidas', 'Samsung', 'Apple', 'Louis Vuitton', 'Gucci', 'Uniqlo', 'Zara']
    for brand in brands:
        medium_queries.append({
            'vietnamese': f'Sản phẩm thương hiệu {brand}',
            'english': f'{brand} brand products',
            'sql': f"SELECT p.name, pr.current_price FROM products p JOIN brands b ON p.brand_id = b.brand_id JOIN product_pricing pr ON p.product_id = pr.product_id WHERE b.brand_name LIKE '%{brand}%' LIMIT 10;",
            'complexity': 'medium'
        })
    
    # Pad to 100 medium queries
    while len(medium_queries) < 100:
        medium_queries.append({
            'vietnamese': f'Sản phẩm có đánh giá trên {3.5 + (len(medium_queries) % 10) * 0.1:.1f} sao',
            'english': f'Products with rating above {3.5 + (len(medium_queries) % 10) * 0.1:.1f} stars',
            'sql': f'SELECT p.name, rv.rating_average FROM products p JOIN product_reviews rv ON p.product_id = rv.product_id WHERE rv.rating_average >= {3.5 + (len(medium_queries) % 10) * 0.1:.1f} ORDER BY rv.rating_average DESC LIMIT 15;',
            'complexity': 'medium'
        })
    
    # Generate 100 complex queries
    complex_queries = [
        {'vietnamese': 'Top 10 sản phẩm đánh giá cao nhất có giá dưới 1 triệu', 'english': 'Top 10 highest rated products under 1 million', 'sql': '''SELECT p.name, b.brand_name, c.category_name, pr.current_price, rv.rating_average, rv.review_count 
FROM products p 
JOIN brands b ON p.brand_id = b.brand_id 
JOIN categories c ON p.category_id = c.category_id 
JOIN product_pricing pr ON p.product_id = pr.product_id 
JOIN product_reviews rv ON p.product_id = rv.product_id 
WHERE pr.current_price < 1000000 AND rv.rating_average >= 4.0 
ORDER BY rv.rating_average DESC, rv.review_count DESC 
LIMIT 10;''', 'complexity': 'complex'},
        {'vietnamese': 'Phân tích thị phần thương hiệu', 'english': 'Brand market share analysis', 'sql': '''SELECT b.brand_name, 
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
LIMIT 20;''', 'complexity': 'complex'},
    ]
    
    # Add complex variations
    complex_templates = [
        ('Top {n} sản phẩm bán chạy nhất trong danh mục {category}', 'Top {n} best selling products in {category} category'),
        ('Sản phẩm {brand1} hoặc {brand2} có đánh giá trên {rating} sao', '{brand1} or {brand2} products with rating above {rating} stars'),
        ('Phân tích giá theo khoảng từ {min_price}k đến {max_price}k', 'Price analysis from {min_price}k to {max_price}k range'),
    ]
    
    categories = ['Phụ kiện thời trang', 'Giày dép nam', 'Túi nam', 'Balo & Vali']
    brands_pairs = [('Nike', 'Adidas'), ('Samsung', 'Apple'), ('Louis Vuitton', 'Gucci')]
    
    for i in range(98):  # Generate 98 more complex queries
        template_idx = i % len(complex_templates)
        vn_template, en_template = complex_templates[template_idx]
        
        if template_idx == 0:  # Top N products
            n = 5 + (i % 10)
            category = categories[i % len(categories)]
            complex_queries.append({
                'vietnamese': vn_template.format(n=n, category=category),
                'english': en_template.format(n=n, category=category),
                'sql': f'''SELECT p.name, b.brand_name, pr.current_price, rv.rating_average, pr.quantity_sold
FROM products p 
JOIN brands b ON p.brand_id = b.brand_id 
JOIN categories c ON p.category_id = c.category_id 
JOIN product_pricing pr ON p.product_id = pr.product_id 
JOIN product_reviews rv ON p.product_id = rv.product_id 
WHERE c.category_name = '{category}'
ORDER BY pr.quantity_sold DESC, rv.rating_average DESC 
LIMIT {n};''',
                'complexity': 'complex'
            })
        elif template_idx == 1:  # Brand comparison
            brand1, brand2 = brands_pairs[i % len(brands_pairs)]
            rating = 4.0 + (i % 5) * 0.1
            complex_queries.append({
                'vietnamese': vn_template.format(brand1=brand1, brand2=brand2, rating=rating),
                'english': en_template.format(brand1=brand1, brand2=brand2, rating=rating),
                'sql': f'''SELECT p.name, b.brand_name, c.category_name, pr.current_price, rv.rating_average
FROM products p 
JOIN brands b ON p.brand_id = b.brand_id 
JOIN categories c ON p.category_id = c.category_id 
JOIN product_pricing pr ON p.product_id = pr.product_id 
JOIN product_reviews rv ON p.product_id = rv.product_id 
WHERE (b.brand_name LIKE '%{brand1}%' OR b.brand_name LIKE '%{brand2}%') 
AND rv.rating_average >= {rating}
ORDER BY rv.rating_average DESC, pr.current_price ASC 
LIMIT 15;''',
                'complexity': 'complex'
            })
        else:  # Price analysis
            min_price = 100 + (i % 10) * 100
            max_price = min_price + 500
            complex_queries.append({
                'vietnamese': vn_template.format(min_price=min_price, max_price=max_price),
                'english': en_template.format(min_price=min_price, max_price=max_price),
                'sql': f'''SELECT 
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
ORDER BY avg_price;''',
                'complexity': 'complex'
            })
    
    return {
        'simple': simple_queries[:100],
        'medium': medium_queries[:100], 
        'complex': complex_queries[:100]
    }

SAMPLE_QUERIES_DATA = get_sample_queries_data()
