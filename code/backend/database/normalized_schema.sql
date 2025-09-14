-- Normalized Database Schema for Vietnamese Tiki Products
-- Purpose: Enable complex JOIN queries for NL2SQL demonstration

-- 1. BRANDS TABLE
CREATE TABLE brands (
    brand_id INTEGER PRIMARY KEY AUTOINCREMENT,
    brand_name TEXT UNIQUE NOT NULL,
    brand_type TEXT DEFAULT 'OEM', -- OEM, Premium, Local, International
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. CATEGORIES TABLE
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL,
    parent_category TEXT,
    category_level INTEGER DEFAULT 1, -- 1=Root, 2=Subcategory
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. SELLERS TABLE
CREATE TABLE sellers (
    seller_id INTEGER PRIMARY KEY AUTOINCREMENT,
    seller_name TEXT UNIQUE NOT NULL,
    seller_type TEXT DEFAULT 'dropship', -- dropship, official, marketplace
    join_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_products INTEGER DEFAULT 0
);

-- 4. PRODUCTS TABLE (Core product information)
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    brand_id INTEGER,
    category_id INTEGER,
    seller_id INTEGER,
    date_created INTEGER,
    number_of_images INTEGER DEFAULT 0,
    has_video BOOLEAN DEFAULT FALSE,
    source_file TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (brand_id) REFERENCES brands(brand_id),
    FOREIGN KEY (category_id) REFERENCES categories(category_id),
    FOREIGN KEY (seller_id) REFERENCES sellers(seller_id)
);

-- 5. PRODUCT_PRICING TABLE (Price and financial information)
CREATE TABLE product_pricing (
    pricing_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    original_price INTEGER,
    current_price INTEGER,
    fulfillment_type TEXT DEFAULT 'dropship',
    pay_later BOOLEAN DEFAULT FALSE,
    vnd_cashback INTEGER DEFAULT 0,
    quantity_sold INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- 6. PRODUCT_REVIEWS TABLE (Review and rating information)
CREATE TABLE product_reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    review_count INTEGER DEFAULT 0,
    rating_average REAL DEFAULT 0.0,
    favourite_count INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- INDEXES for better JOIN performance
CREATE INDEX idx_products_brand ON products(brand_id);
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_seller ON products(seller_id);
CREATE INDEX idx_pricing_product ON product_pricing(product_id);
CREATE INDEX idx_reviews_product ON product_reviews(product_id);
CREATE INDEX idx_brands_name ON brands(brand_name);
CREATE INDEX idx_categories_name ON categories(category_name);
CREATE INDEX idx_sellers_name ON sellers(seller_name);
