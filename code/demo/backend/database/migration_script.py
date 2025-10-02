#!/usr/bin/env python3
"""
Database Migration Script: CSV to Normalized Schema
Migrates existing single-table structure to normalized multi-table schema
for complex JOIN query demonstrations in Vietnamese NL2SQL pipeline.
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    def __init__(self, db_path: str, csv_directory: str):
        self.db_path = db_path
        self.csv_directory = csv_directory
        self.conn = None
        
    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        logger.info(f"Connected to database: {self.db_path}")
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def backup_existing_data(self):
        """Backup existing products table"""
        try:
            backup_query = """
            CREATE TABLE products_backup AS 
            SELECT * FROM products;
            """
            self.conn.execute(backup_query)
            self.conn.commit()
            logger.info("Existing products table backed up to products_backup")
        except sqlite3.Error as e:
            logger.warning(f"Backup failed or table doesn't exist: {e}")
    
    def create_normalized_schema(self):
        """Create normalized tables"""
        schema_file = Path(__file__).parent / "normalized_schema.sql"
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        # Execute schema creation
        statements = schema_sql.split(';')
        for statement in statements:
            if statement.strip():
                try:
                    self.conn.execute(statement)
                except sqlite3.Error as e:
                    logger.error(f"Error executing statement: {e}")
                    logger.error(f"Statement: {statement[:100]}...")
        
        self.conn.commit()
        logger.info("Normalized schema created successfully")
    
    def extract_unique_entities(self) -> Dict[str, Set[str]]:
        """Extract unique brands, categories, and sellers from CSV files"""
        brands = set()
        categories = set()
        sellers = set()
        
        csv_files = list(Path(self.csv_directory).glob("*.csv"))
        logger.info(f"Processing {len(csv_files)} CSV files for entity extraction")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Extract unique values
                brands.update(df['brand'].dropna().unique())
                categories.update(df['category'].dropna().unique())
                sellers.update(df['current_seller'].dropna().unique())
                
                logger.info(f"Processed {csv_file.name}: {len(df)} records")
                
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
        
        logger.info(f"Extracted entities - Brands: {len(brands)}, Categories: {len(categories)}, Sellers: {len(sellers)}")
        return {
            'brands': brands,
            'categories': categories,
            'sellers': sellers
        }
    
    def populate_reference_tables(self, entities: Dict[str, Set[str]]):
        """Populate brands, categories, and sellers tables"""
        
        # Populate brands
        brand_data = [(brand, 'OEM' if brand == 'OEM' else 'Branded') for brand in entities['brands']]
        self.conn.executemany(
            "INSERT OR IGNORE INTO brands (brand_name, brand_type) VALUES (?, ?)",
            brand_data
        )
        
        # Populate categories (detect parent-child relationships)
        category_data = []
        for category in entities['categories']:
            if category in ['Root', 'Balo nữ', 'Cài Áo', 'Giày dép nam', 'Túi nữ']:
                category_data.append((category, None, 1))
            else:
                category_data.append((category, 'Root', 2))
        
        self.conn.executemany(
            "INSERT OR IGNORE INTO categories (category_name, parent_category, category_level) VALUES (?, ?, ?)",
            category_data
        )
        
        # Populate sellers
        seller_data = [(seller, 'dropship') for seller in entities['sellers']]
        self.conn.executemany(
            "INSERT OR IGNORE INTO sellers (seller_name, seller_type) VALUES (?, ?)",
            seller_data
        )
        
        self.conn.commit()
        logger.info("Reference tables populated successfully")
    
    def migrate_product_data(self):
        """Migrate CSV data to normalized tables"""
        csv_files = list(Path(self.csv_directory).glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"Migrating data from {csv_file.name}: {len(df)} records")
                
                for _, row in df.iterrows():
                    # Get foreign key IDs
                    brand_id = self.get_or_create_brand_id(row.get('brand', 'OEM'))
                    category_id = self.get_or_create_category_id(row.get('category', 'Root'))
                    seller_id = self.get_or_create_seller_id(row.get('current_seller', 'Unknown'))
                    
                    # Insert into products table
                    product_data = (
                        int(row['id']),
                        row['name'],
                        row.get('description', ''),
                        brand_id,
                        category_id,
                        seller_id,
                        int(row.get('date_created', 0)),
                        int(row.get('number_of_images', 0)),
                        bool(row.get('has_video', False)),
                        csv_file.name
                    )
                    
                    self.conn.execute("""
                        INSERT OR REPLACE INTO products 
                        (product_id, name, description, brand_id, category_id, seller_id, 
                         date_created, number_of_images, has_video, source_file)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, product_data)
                    
                    # Insert into product_pricing table
                    pricing_data = (
                        int(row['id']),
                        int(row.get('original_price', 0)),
                        int(row.get('price', 0)),
                        row.get('fulfillment_type', 'dropship'),
                        bool(row.get('pay_later', False)),
                        int(row.get('vnd_cashback', 0)),
                        int(row.get('quantity_sold', 0))
                    )
                    
                    self.conn.execute("""
                        INSERT OR REPLACE INTO product_pricing
                        (product_id, original_price, current_price, fulfillment_type, 
                         pay_later, vnd_cashback, quantity_sold)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, pricing_data)
                    
                    # Insert into product_reviews table
                    reviews_data = (
                        int(row['id']),
                        int(row.get('review_count', 0)),
                        float(row.get('rating_average', 0.0)),
                        int(row.get('favourite_count', 0))
                    )
                    
                    self.conn.execute("""
                        INSERT OR REPLACE INTO product_reviews
                        (product_id, review_count, rating_average, favourite_count)
                        VALUES (?, ?, ?, ?)
                    """, reviews_data)
                
                self.conn.commit()
                logger.info(f"Successfully migrated {csv_file.name}")
                
            except Exception as e:
                logger.error(f"Error migrating {csv_file}: {e}")
                self.conn.rollback()
    
    def get_or_create_brand_id(self, brand_name: str) -> int:
        """Get brand_id or create if not exists"""
        cursor = self.conn.execute("SELECT brand_id FROM brands WHERE brand_name = ?", (brand_name,))
        result = cursor.fetchone()
        return result[0] if result else 1  # Default to brand_id 1 if not found
    
    def get_or_create_category_id(self, category_name: str) -> int:
        """Get category_id or create if not exists"""
        cursor = self.conn.execute("SELECT category_id FROM categories WHERE category_name = ?", (category_name,))
        result = cursor.fetchone()
        return result[0] if result else 1  # Default to category_id 1 if not found
    
    def get_or_create_seller_id(self, seller_name: str) -> int:
        """Get seller_id or create if not exists"""
        cursor = self.conn.execute("SELECT seller_id FROM sellers WHERE seller_name = ?", (seller_name,))
        result = cursor.fetchone()
        return result[0] if result else 1  # Default to seller_id 1 if not found
    
    def update_seller_stats(self):
        """Update seller statistics"""
        self.conn.execute("""
            UPDATE sellers 
            SET total_products = (
                SELECT COUNT(*) 
                FROM products 
                WHERE products.seller_id = sellers.seller_id
            )
        """)
        self.conn.commit()
        logger.info("Seller statistics updated")
    
    def verify_migration(self):
        """Verify migration results"""
        tables = ['brands', 'categories', 'sellers', 'products', 'product_pricing', 'product_reviews']
        
        for table in tables:
            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"Table {table}: {count} records")
        
        # Verify foreign key relationships
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM products p
            JOIN brands b ON p.brand_id = b.brand_id
            JOIN categories c ON p.category_id = c.category_id
            JOIN sellers s ON p.seller_id = s.seller_id
        """)
        valid_products = cursor.fetchone()[0]
        logger.info(f"Products with valid foreign keys: {valid_products}")

def main():
    """Main migration function"""
    # Configuration
    project_root = Path(__file__).parent.parent.parent.parent
    db_path = project_root / "data" / "tiki_products_normalized.db"
    csv_directory = project_root / "data" / "Sample_Tiki_dataset"
    
    logger.info("Starting database migration to normalized schema")
    logger.info(f"Database path: {db_path}")
    logger.info(f"CSV directory: {csv_directory}")
    
    # Initialize migrator
    migrator = DatabaseMigrator(str(db_path), str(csv_directory))
    
    try:
        migrator.connect()
        
        # Migration steps
        logger.info("Step 1: Creating normalized schema")
        migrator.create_normalized_schema()
        
        logger.info("Step 2: Extracting unique entities")
        entities = migrator.extract_unique_entities()
        
        logger.info("Step 3: Populating reference tables")
        migrator.populate_reference_tables(entities)
        
        logger.info("Step 4: Migrating product data")
        migrator.migrate_product_data()
        
        logger.info("Step 5: Updating statistics")
        migrator.update_seller_stats()
        
        logger.info("Step 6: Verifying migration")
        migrator.verify_migration()
        
        logger.info("Database migration completed successfully!")
        logger.info(f"New normalized database created: {db_path}")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        migrator.close()

if __name__ == "__main__":
    main()
