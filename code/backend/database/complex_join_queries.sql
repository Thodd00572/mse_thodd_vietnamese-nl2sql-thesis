-- Complex JOIN Queries for Vietnamese NL2SQL Demonstration
-- These queries showcase multi-table relationships and complex business logic

-- 1. BRAND PERFORMANCE ANALYSIS
-- Vietnamese: "Thương hiệu nào có sản phẩm bán chạy nhất với rating cao?"
SELECT 
    b.brand_name,
    COUNT(p.product_id) as total_products,
    AVG(pr.rating_average) as avg_rating,
    SUM(pp.quantity_sold) as total_sold,
    AVG(pp.current_price) as avg_price
FROM brands b
JOIN products p ON b.brand_id = p.brand_id
JOIN product_reviews pr ON p.product_id = pr.product_id
JOIN product_pricing pp ON p.product_id = pp.product_id
WHERE pr.rating_average >= 4.0
GROUP BY b.brand_id, b.brand_name
HAVING total_sold > 100
ORDER BY total_sold DESC, avg_rating DESC
LIMIT 10;

-- 2. CATEGORY-SELLER PERFORMANCE MATRIX
-- Vietnamese: "Danh mục nào có nhiều seller tham gia nhất và doanh thu cao?"
SELECT 
    c.category_name,
    COUNT(DISTINCT s.seller_id) as unique_sellers,
    COUNT(p.product_id) as total_products,
    SUM(pp.current_price * pp.quantity_sold) as total_revenue,
    AVG(pr.rating_average) as category_avg_rating
FROM categories c
JOIN products p ON c.category_id = p.category_id
JOIN sellers s ON p.seller_id = s.seller_id
JOIN product_pricing pp ON p.product_id = pp.product_id
JOIN product_reviews pr ON p.product_id = pr.product_id
GROUP BY c.category_id, c.category_name
HAVING unique_sellers >= 5
ORDER BY total_revenue DESC;

-- 3. TOP SELLERS BY CATEGORY WITH BRAND DIVERSITY
-- Vietnamese: "Seller nào bán nhiều thương hiệu khác nhau trong từng danh mục?"
SELECT 
    s.seller_name,
    c.category_name,
    COUNT(DISTINCT b.brand_id) as brand_diversity,
    COUNT(p.product_id) as products_in_category,
    AVG(pp.current_price) as avg_price_in_category,
    SUM(pp.quantity_sold) as total_sold_in_category
FROM sellers s
JOIN products p ON s.seller_id = p.seller_id
JOIN brands b ON p.brand_id = b.brand_id
JOIN categories c ON p.category_id = c.category_id
JOIN product_pricing pp ON p.product_id = pp.product_id
GROUP BY s.seller_id, s.seller_name, c.category_id, c.category_name
HAVING brand_diversity >= 3
ORDER BY brand_diversity DESC, total_sold_in_category DESC;

-- 4. PRICE COMPETITIVENESS ANALYSIS
-- Vietnamese: "So sánh giá sản phẩm giữa các thương hiệu trong cùng danh mục"
SELECT 
    c.category_name,
    b.brand_name,
    COUNT(p.product_id) as product_count,
    MIN(pp.current_price) as min_price,
    MAX(pp.current_price) as max_price,
    AVG(pp.current_price) as avg_price,
    AVG(pr.rating_average) as avg_rating,
    SUM(pp.quantity_sold) as total_sales
FROM categories c
JOIN products p ON c.category_id = p.category_id
JOIN brands b ON p.brand_id = b.brand_id
JOIN product_pricing pp ON p.product_id = pp.product_id
JOIN product_reviews pr ON p.product_id = pr.product_id
WHERE c.category_name IN ('Balo nữ', 'Giày dép nam', 'Túi nữ')
GROUP BY c.category_id, c.category_name, b.brand_id, b.brand_name
HAVING product_count >= 2
ORDER BY c.category_name, avg_price DESC;

-- 5. CUSTOMER ENGAGEMENT METRICS
-- Vietnamese: "Sản phẩm nào có tỷ lệ yêu thích/đánh giá cao nhất?"
SELECT 
    p.name,
    b.brand_name,
    c.category_name,
    s.seller_name,
    pr.rating_average,
    pr.review_count,
    pr.favourite_count,
    pp.current_price,
    pp.quantity_sold,
    CASE 
        WHEN pr.review_count > 0 THEN CAST(pr.favourite_count AS REAL) / pr.review_count 
        ELSE 0 
    END as favourite_to_review_ratio
FROM products p
JOIN brands b ON p.brand_id = b.brand_id
JOIN categories c ON p.category_id = c.category_id
JOIN sellers s ON p.seller_id = s.seller_id
JOIN product_pricing pp ON p.product_id = pp.product_id
JOIN product_reviews pr ON p.product_id = pr.product_id
WHERE pr.review_count >= 10 AND pr.rating_average >= 4.0
ORDER BY favourite_to_review_ratio DESC, pr.rating_average DESC
LIMIT 20;

-- 6. SELLER SPECIALIZATION ANALYSIS
-- Vietnamese: "Seller nào chuyên về thương hiệu cao cấp và có doanh thu tốt?"
SELECT 
    s.seller_name,
    COUNT(DISTINCT b.brand_id) as brand_count,
    COUNT(DISTINCT c.category_id) as category_count,
    COUNT(p.product_id) as total_products,
    AVG(pp.current_price) as avg_product_price,
    SUM(pp.current_price * pp.quantity_sold) as total_revenue,
    AVG(pr.rating_average) as avg_seller_rating,
    STRING_AGG(DISTINCT b.brand_name, ', ') as brands_sold
FROM sellers s
JOIN products p ON s.seller_id = p.seller_id
JOIN brands b ON p.brand_id = b.brand_id
JOIN categories c ON p.category_id = c.category_id
JOIN product_pricing pp ON p.product_id = pp.product_id
JOIN product_reviews pr ON p.product_id = pr.product_id
GROUP BY s.seller_id, s.seller_name
HAVING avg_product_price > 100000 AND total_products >= 5
ORDER BY total_revenue DESC, avg_seller_rating DESC;

-- 7. MARKET PENETRATION BY BRAND-CATEGORY
-- Vietnamese: "Thương hiệu nào có mặt trong nhiều danh mục nhất?"
SELECT 
    b.brand_name,
    COUNT(DISTINCT c.category_id) as category_penetration,
    COUNT(p.product_id) as total_products,
    STRING_AGG(DISTINCT c.category_name, ' | ') as categories,
    AVG(pp.current_price) as avg_price_across_categories,
    SUM(pp.quantity_sold) as total_units_sold
FROM brands b
JOIN products p ON b.brand_id = p.brand_id
JOIN categories c ON p.category_id = c.category_id
JOIN product_pricing pp ON p.product_id = pp.product_id
GROUP BY b.brand_id, b.brand_name
HAVING category_penetration >= 2
ORDER BY category_penetration DESC, total_units_sold DESC;
