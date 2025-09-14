#!/usr/bin/env python3
"""
Tiki Dataset Import Script
Imports all CSV files from Sample_Tiki_dataset into SQLite database
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_database_schema(db_path):
    """Create SQLite database with appropriate schema for Tiki products"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create products table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        row_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        original_price INTEGER,
        price INTEGER,
        fulfillment_type TEXT,
        brand TEXT,
        review_count INTEGER DEFAULT 0,
        rating_average REAL DEFAULT 0.0,
        favourite_count INTEGER DEFAULT 0,
        pay_later BOOLEAN DEFAULT FALSE,
        current_seller TEXT,
        date_created INTEGER,
        number_of_images INTEGER DEFAULT 0,
        vnd_cashback INTEGER DEFAULT 0,
        has_video BOOLEAN DEFAULT FALSE,
        category TEXT,
        quantity_sold INTEGER DEFAULT 0,
        source_file TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create indexes for better query performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_product_id ON products(product_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON products(category)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_brand ON products(brand)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_price ON products(price)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rating ON products(rating_average)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_seller ON products(current_seller)')
    
    conn.commit()
    conn.close()
    logger.info(f"Database schema created successfully at {db_path}")

def clean_boolean_value(value):
    """Convert string boolean values to proper boolean"""
    if pd.isna(value) or value == '':
        return False
    if isinstance(value, str):
        return value.lower() in ['true', '1', 'yes']
    return bool(value)

def clean_numeric_value(value, default=0):
    """Clean numeric values, handle NaN and empty strings"""
    if pd.isna(value) or value == '':
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def clean_float_value(value, default=0.0):
    """Clean float values, handle NaN and empty strings"""
    if pd.isna(value) or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def import_csv_file(csv_path, db_path):
    """Import a single CSV file into the database"""
    logger.info(f"Importing {csv_path}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Clean the data
        df['product_id'] = df['id'].apply(lambda x: clean_numeric_value(x))
        df['name'] = df['name'].fillna('').astype(str)
        df['description'] = df['description'].fillna('').astype(str)
        df['original_price'] = df['original_price'].apply(lambda x: clean_numeric_value(x))
        df['price'] = df['price'].apply(lambda x: clean_numeric_value(x))
        df['fulfillment_type'] = df['fulfillment_type'].fillna('').astype(str)
        df['brand'] = df['brand'].fillna('').astype(str)
        df['review_count'] = df['review_count'].apply(lambda x: clean_numeric_value(x))
        df['rating_average'] = df['rating_average'].apply(lambda x: clean_float_value(x))
        df['favourite_count'] = df['favourite_count'].apply(lambda x: clean_numeric_value(x))
        df['pay_later'] = df['pay_later'].apply(clean_boolean_value)
        df['current_seller'] = df['current_seller'].fillna('').astype(str)
        df['date_created'] = df['date_created'].apply(lambda x: clean_numeric_value(x))
        df['number_of_images'] = df['number_of_images'].apply(lambda x: clean_numeric_value(x))
        df['vnd_cashback'] = df['vnd_cashback'].apply(lambda x: clean_numeric_value(x))
        df['has_video'] = df['has_video'].apply(clean_boolean_value)
        df['category'] = df['category'].fillna('').astype(str)
        df['quantity_sold'] = df['quantity_sold'].apply(lambda x: clean_numeric_value(x))
        df['source_file'] = os.path.basename(csv_path)
        
        # Connect to database and insert data
        conn = sqlite3.connect(db_path)
        
        # Prepare data for insertion (exclude the unnamed index column and original 'id')
        insert_columns = [
            'product_id', 'name', 'description', 'original_price', 'price',
            'fulfillment_type', 'brand', 'review_count', 'rating_average',
            'favourite_count', 'pay_later', 'current_seller', 'date_created',
            'number_of_images', 'vnd_cashback', 'has_video', 'category',
            'quantity_sold', 'source_file'
        ]
        
        # Insert data
        df[insert_columns].to_sql('products', conn, if_exists='append', index=False)
        
        conn.close()
        logger.info(f"Successfully imported {len(df)} records from {os.path.basename(csv_path)}")
        return len(df)
        
    except Exception as e:
        logger.error(f"Error importing {csv_path}: {str(e)}")
        return 0

def main():
    """Main function to import all Tiki CSV files"""
    # Define paths
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / 'data' / 'Sample_Tiki_dataset'
    db_path = project_root / 'data' / 'tiki_products.db'
    
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Database path: {db_path}")
    
    # Create database schema
    create_database_schema(db_path)
    
    # Find all CSV files
    csv_files = list(dataset_path.glob('*.csv'))
    logger.info(f"Found {len(csv_files)} CSV files to import")
    
    total_records = 0
    successful_imports = 0
    
    # Import each CSV file
    for csv_file in csv_files:
        records_imported = import_csv_file(csv_file, db_path)
        if records_imported > 0:
            total_records += records_imported
            successful_imports += 1
    
    # Print summary
    logger.info(f"\n=== IMPORT SUMMARY ===")
    logger.info(f"Files processed: {len(csv_files)}")
    logger.info(f"Successful imports: {successful_imports}")
    logger.info(f"Total records imported: {total_records}")
    logger.info(f"Database location: {db_path}")
    
    # Verify data in database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM products")
    db_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT source_file, COUNT(*) FROM products GROUP BY source_file")
    file_counts = cursor.fetchall()
    
    logger.info(f"\n=== DATABASE VERIFICATION ===")
    logger.info(f"Total records in database: {db_count}")
    logger.info("Records per file:")
    for file_name, count in file_counts:
        logger.info(f"  {file_name}: {count} records")
    
    # Sample data check
    cursor.execute("SELECT product_id, name, price, category FROM products LIMIT 5")
    sample_data = cursor.fetchall()
    logger.info(f"\n=== SAMPLE DATA ===")
    for row in sample_data:
        logger.info(f"ID: {row[0]}, Name: {row[1][:50]}..., Price: {row[2]}, Category: {row[3]}")
    
    conn.close()

if __name__ == "__main__":
    main()
