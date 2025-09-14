#!/usr/bin/env python3
"""
Import all Tiki CSV files from Sample_Tiki_dataset into the database
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path

def import_csv_to_database():
    """Import all CSV files from Sample_Tiki_dataset into the database"""
    
    # Database path
    db_path = "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/backend/data/tiki_products.db"
    csv_folder = "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/Sample_Tiki_dataset"
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop existing products table and recreate with proper schema
    cursor.execute("DROP TABLE IF EXISTS products")
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            tiki_id INTEGER,
            name TEXT,
            description TEXT,
            original_price REAL,
            price REAL,
            fulfillment_type TEXT,
            brand TEXT,
            review_count INTEGER,
            rating_average REAL,
            favourite_count INTEGER,
            pay_later BOOLEAN,
            current_seller TEXT,
            date_created INTEGER,
            number_of_images INTEGER,
            vnd_cashback INTEGER,
            has_video BOOLEAN,
            category TEXT,
            quantity_sold INTEGER
        )
    """)
    
    # Get all CSV files
    csv_files = [
        "vietnamese_tiki_products_backpacks_suitcases.csv",
        "vietnamese_tiki_products_fashion_accessories.csv", 
        "vietnamese_tiki_products_men_bags.csv",
        "vietnamese_tiki_products_men_shoes.csv",
        "vietnamese_tiki_products_women_bags.csv",
        "vietnamese_tiki_products_women_shoes.csv"
    ]
    
    total_imported = 0
    
    for csv_file in csv_files:
        csv_path = os.path.join(csv_folder, csv_file)
        if os.path.exists(csv_path):
            print(f"Importing {csv_file}...")
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Clean and prepare data
            df = df.drop(columns=['Unnamed: 0'], errors='ignore')  # Remove index column
            df['tiki_id'] = df['id']  # Keep original Tiki ID
            df = df.drop(columns=['id'])  # Remove original id column
            
            # Convert boolean columns
            df['pay_later'] = df['pay_later'].astype(bool)
            df['has_video'] = df['has_video'].astype(bool)
            
            # Handle missing values
            df = df.fillna({
                'description': '',
                'brand': 'Unknown',
                'category': 'Uncategorized',
                'current_seller': 'Unknown'
            })
            
            # Insert data
            df.to_sql('products', conn, if_exists='append', index=False)
            
            imported_count = len(df)
            total_imported += imported_count
            print(f"  → Imported {imported_count} products from {csv_file}")
        else:
            print(f"File not found: {csv_path}")
    
    # Create indexes for better query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_name ON products(name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_price ON products(price)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand)")
    
    # Get final count
    cursor.execute("SELECT COUNT(*) FROM products")
    final_count = cursor.fetchone()[0]
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Import completed!")
    print(f"Total products imported: {total_imported}")
    print(f"Final database count: {final_count}")
    
    return final_count

if __name__ == "__main__":
    import_csv_to_database()
