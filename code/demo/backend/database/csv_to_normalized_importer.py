#!/usr/bin/env python3
"""
CSV to Normalized Database Importer
Migrates Vietnamese Tiki product data from CSV files to normalized SQLite database
with data enrichment and missing field generation.
"""

import sqlite3
import pandas as pd
import os
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TikiDataImporter:
    """Import and normalize Tiki product data from CSV files"""
    
    def __init__(self, db_path: str = "data/tiki_products_normalized.db"):
        self.db_path = db_path
        self.conn = None
        
        # Data enrichment mappings
        self.brand_enrichment = {
            'OEM': ['Generic', 'No Brand', 'Unbranded', 'Private Label'],
            'Premium': ['Apple', 'Samsung', 'Nike', 'Adidas', 'Gucci', 'Louis Vuitton'],
            'Local': ['Biti\'s', 'Vina Giày', 'Thủy Tiên', 'Hoàng Long', 'Việt Tiến'],
            'International': ['Uniqlo', 'H&M', 'Zara', 'Timberland', 'Converse', 'Vans']
        }
        
        self.category_mapping = {
            'Balo nữ': 'Balo & Vali',
            'Root': 'Phụ kiện thời trang',
            'Cài Áo': 'Phụ kiện thời trang',
            'Giày nam': 'Giày dép nam', 
            'Giày nữ': 'Giày dép nữ',
            'Túi nam': 'Túi nam',
            'Túi nữ': 'Túi nữ'
        }
        
        self.seller_types = ['dropship', 'official', 'marketplace']
        
    def connect_db(self):
        """Create database connection and ensure directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        
    def create_normalized_schema(self):
        """Create normalized database schema"""
        schema_sql = """
        -- Drop existing tables if they exist
        DROP TABLE IF EXISTS product_reviews;
        DROP TABLE IF EXISTS product_pricing;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS sellers;
        DROP TABLE IF EXISTS categories;
        DROP TABLE IF EXISTS brands;
        
        -- 1. BRANDS TABLE
        CREATE TABLE brands (
            brand_id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand_name TEXT UNIQUE NOT NULL,
            brand_type TEXT DEFAULT 'OEM',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 2. CATEGORIES TABLE
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL,
            parent_category TEXT,
            category_level INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 3. SELLERS TABLE
        CREATE TABLE sellers (
            seller_id INTEGER PRIMARY KEY AUTOINCREMENT,
            seller_name TEXT UNIQUE NOT NULL,
            seller_type TEXT DEFAULT 'dropship',
            join_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_products INTEGER DEFAULT 0
        );
        
        -- 4. PRODUCTS TABLE
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
        
        -- 5. PRODUCT_PRICING TABLE
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
        
        -- 6. PRODUCT_REVIEWS TABLE
        CREATE TABLE product_reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            review_count INTEGER DEFAULT 0,
            rating_average REAL DEFAULT 0.0,
            favourite_count INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );
        
        -- INDEXES for better performance
        CREATE INDEX idx_products_brand ON products(brand_id);
        CREATE INDEX idx_products_category ON products(category_id);
        CREATE INDEX idx_products_seller ON products(seller_id);
        CREATE INDEX idx_pricing_product ON product_pricing(product_id);
        CREATE INDEX idx_reviews_product ON product_reviews(product_id);
        """
        
        # Execute schema creation
        for statement in schema_sql.split(';'):
            if statement.strip():
                self.conn.execute(statement)
        self.conn.commit()
        logger.info("Normalized schema created successfully")
    
    def enrich_brand_name(self, original_brand: str) -> Tuple[str, str]:
        """Enrich brand names and determine brand type"""
        if not original_brand or original_brand.strip() == '' or original_brand == 'OEM':
            # Generate realistic brand name for OEM products
            brand_type = 'OEM'
            brand_name = random.choice(self.brand_enrichment['OEM'])
        else:
            brand_name = original_brand.strip()
            # Determine brand type based on name patterns
            if any(premium in brand_name for premium in self.brand_enrichment['Premium']):
                brand_type = 'Premium'
            elif any(local in brand_name for local in self.brand_enrichment['Local']):
                brand_type = 'Local'
            elif len(brand_name) > 15 or 'Shop' in brand_name or 'Store' in brand_name:
                brand_type = 'Local'
            else:
                brand_type = 'International'
        
        return brand_name, brand_type
    
    def normalize_category(self, original_category: str, source_file: str) -> str:
        """Normalize category names based on source file and existing category"""
        if original_category in self.category_mapping:
            return self.category_mapping[original_category]
        
        # Infer category from source file name
        if 'backpack' in source_file or 'suitcase' in source_file:
            return 'Balo & Vali'
        elif 'men_bags' in source_file:
            return 'Túi nam'
        elif 'women_bags' in source_file:
            return 'Túi nữ'
        elif 'men_shoes' in source_file:
            return 'Giày dép nam'
        elif 'women_shoes' in source_file:
            return 'Giày dép nữ'
        elif 'fashion_accessories' in source_file:
            return 'Phụ kiện thời trang'
        else:
            return original_category if original_category != 'Root' else 'Phụ kiện thời trang'
    
    def generate_missing_data(self, row: pd.Series) -> Dict:
        """Generate realistic missing data for products"""
        enriched_data = {}
        
        # Generate rating if missing
        if row['rating_average'] == 0.0 and row['review_count'] == 0:
            # 70% chance of having reviews
            if random.random() < 0.7:
                enriched_data['review_count'] = random.randint(1, 150)
                enriched_data['rating_average'] = round(random.uniform(3.5, 5.0), 1)
            else:
                enriched_data['review_count'] = 0
                enriched_data['rating_average'] = 0.0
        else:
            enriched_data['review_count'] = row['review_count']
            enriched_data['rating_average'] = row['rating_average']
        
        # Generate quantity sold if missing
        if row['quantity_sold'] == 0:
            # Base on price range - cheaper items sell more
            price = row['price']
            if price < 50000:
                enriched_data['quantity_sold'] = random.randint(50, 500)
            elif price < 200000:
                enriched_data['quantity_sold'] = random.randint(10, 200)
            else:
                enriched_data['quantity_sold'] = random.randint(1, 50)
        else:
            enriched_data['quantity_sold'] = row['quantity_sold']
        
        # Generate favourite count based on reviews
        if enriched_data['review_count'] > 0:
            enriched_data['favourite_count'] = max(0, enriched_data['review_count'] // 3 + random.randint(-5, 10))
        else:
            enriched_data['favourite_count'] = row['favourite_count']
        
        # Generate cashback (5-15% of price for some products)
        if random.random() < 0.3:  # 30% of products have cashback
            enriched_data['vnd_cashback'] = int(row['price'] * random.uniform(0.05, 0.15))
        else:
            enriched_data['vnd_cashback'] = row['vnd_cashback']
        
        return enriched_data
    
    def insert_or_get_brand(self, brand_name: str, brand_type: str) -> int:
        """Insert brand or get existing brand ID"""
        cursor = self.conn.cursor()
        
        # Check if brand exists
        cursor.execute("SELECT brand_id FROM brands WHERE brand_name = ?", (brand_name,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
        # Insert new brand
        cursor.execute("""
            INSERT INTO brands (brand_name, brand_type) 
            VALUES (?, ?)
        """, (brand_name, brand_type))
        
        return cursor.lastrowid
    
    def insert_or_get_category(self, category_name: str) -> int:
        """Insert category or get existing category ID"""
        cursor = self.conn.cursor()
        
        # Check if category exists
        cursor.execute("SELECT category_id FROM categories WHERE category_name = ?", (category_name,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
        # Insert new category
        cursor.execute("""
            INSERT INTO categories (category_name, category_level) 
            VALUES (?, ?)
        """, (category_name, 1))
        
        return cursor.lastrowid
    
    def insert_or_get_seller(self, seller_name: str, fulfillment_type: str) -> int:
        """Insert seller or get existing seller ID"""
        cursor = self.conn.cursor()
        
        # Check if seller exists
        cursor.execute("SELECT seller_id FROM sellers WHERE seller_name = ?", (seller_name,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
        # Determine seller type
        seller_type = fulfillment_type if fulfillment_type in self.seller_types else 'dropship'
        
        # Insert new seller
        cursor.execute("""
            INSERT INTO sellers (seller_name, seller_type) 
            VALUES (?, ?)
        """, (seller_name, seller_type))
        
        return cursor.lastrowid
    
    def import_csv_file(self, csv_path: str):
        """Import single CSV file into normalized database"""
        logger.info(f"Importing {csv_path}")
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        source_file = os.path.basename(csv_path)
        
        # Clean data
        df = df.fillna('')
        df = df.replace('nan', '')
        
        imported_count = 0
        
        for _, row in df.iterrows():
            try:
                # Enrich and normalize data
                brand_name, brand_type = self.enrich_brand_name(row['brand'])
                category_name = self.normalize_category(row['category'], source_file)
                enriched_data = self.generate_missing_data(row)
                
                # Get or create foreign key IDs
                brand_id = self.insert_or_get_brand(brand_name, brand_type)
                category_id = self.insert_or_get_category(category_name)
                seller_id = self.insert_or_get_seller(row['current_seller'], row['fulfillment_type'])
                
                # Insert product
                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO products 
                    (product_id, name, description, brand_id, category_id, seller_id, 
                     date_created, number_of_images, has_video, source_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['id'], row['name'], row['description'], brand_id, category_id, seller_id,
                    row['date_created'], row['number_of_images'], bool(row['has_video']), source_file
                ))
                
                # Insert pricing
                cursor.execute("""
                    INSERT OR REPLACE INTO product_pricing 
                    (product_id, original_price, current_price, fulfillment_type, 
                     pay_later, vnd_cashback, quantity_sold)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['id'], row['original_price'], row['price'], row['fulfillment_type'],
                    bool(row['pay_later']), enriched_data['vnd_cashback'], enriched_data['quantity_sold']
                ))
                
                # Insert reviews
                cursor.execute("""
                    INSERT OR REPLACE INTO product_reviews 
                    (product_id, review_count, rating_average, favourite_count)
                    VALUES (?, ?, ?, ?)
                """, (
                    row['id'], enriched_data['review_count'], 
                    enriched_data['rating_average'], enriched_data['favourite_count']
                ))
                
                imported_count += 1
                
            except Exception as e:
                logger.error(f"Error importing row {row['id']}: {e}")
                continue
        
        self.conn.commit()
        logger.info(f"Imported {imported_count} products from {source_file}")
        return imported_count
    
    def update_seller_product_counts(self):
        """Update total_products count for each seller"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE sellers 
            SET total_products = (
                SELECT COUNT(*) 
                FROM products 
                WHERE products.seller_id = sellers.seller_id
            )
        """)
        self.conn.commit()
        logger.info("Updated seller product counts")
    
    def import_all_csv_files(self, csv_directory: str):
        """Import all CSV files from directory"""
        csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
        
        if not csv_files:
            logger.error(f"No CSV files found in {csv_directory}")
            return
        
        total_imported = 0
        
        for csv_file in csv_files:
            csv_path = os.path.join(csv_directory, csv_file)
            count = self.import_csv_file(csv_path)
            total_imported += count
        
        # Update seller statistics
        self.update_seller_product_counts()
        
        logger.info(f"Import completed! Total products imported: {total_imported}")
        
        # Print summary statistics
        self.print_import_summary()
    
    def print_import_summary(self):
        """Print import summary statistics"""
        cursor = self.conn.cursor()
        
        # Get counts
        cursor.execute("SELECT COUNT(*) FROM products")
        product_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM brands")
        brand_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM categories")
        category_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sellers")
        seller_count = cursor.fetchone()[0]
        
        # Get top categories
        cursor.execute("""
            SELECT c.category_name, COUNT(p.product_id) as count
            FROM categories c
            LEFT JOIN products p ON c.category_id = p.category_id
            GROUP BY c.category_id
            ORDER BY count DESC
            LIMIT 5
        """)
        top_categories = cursor.fetchall()
        
        logger.info("=== IMPORT SUMMARY ===")
        logger.info(f"Products: {product_count}")
        logger.info(f"Brands: {brand_count}")
        logger.info(f"Categories: {category_count}")
        logger.info(f"Sellers: {seller_count}")
        logger.info("Top Categories:")
        for cat_name, count in top_categories:
            logger.info(f"  {cat_name}: {count} products")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

def main():
    """Main import function"""
    # Configuration
    csv_directory = "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/Sample_Tiki_dataset"
    db_path = "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/backend/data/tiki_products_normalized.db"
    
    # Initialize importer
    importer = TikiDataImporter(db_path)
    
    try:
        # Connect and create schema
        importer.connect_db()
        importer.create_normalized_schema()
        
        # Import all CSV files
        importer.import_all_csv_files(csv_directory)
        
        logger.info("Data import completed successfully!")
        
    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise
    finally:
        importer.close()

if __name__ == "__main__":
    main()
