#!/usr/bin/env python3
"""
Tiki Database Query Script
Provides useful queries to explore the imported Tiki product data
"""

import sqlite3
import pandas as pd
from pathlib import Path

def connect_db():
    """Connect to the Tiki products database"""
    db_path = Path(__file__).parent.parent / 'data' / 'tiki_products.db'
    return sqlite3.connect(db_path)

def basic_stats():
    """Display basic statistics about the dataset"""
    conn = connect_db()
    
    print("=== TIKI PRODUCTS DATABASE STATISTICS ===\n")
    
    # Total products
    total = pd.read_sql_query("SELECT COUNT(*) as total FROM products", conn).iloc[0]['total']
    print(f"Total Products: {total:,}")
    
    # Products by category
    print("\n--- Products by Category ---")
    category_stats = pd.read_sql_query("""
        SELECT category, COUNT(*) as count 
        FROM products 
        GROUP BY category 
        ORDER BY count DESC
    """, conn)
    print(category_stats.to_string(index=False))
    
    # Products by source file
    print("\n--- Products by Source File ---")
    file_stats = pd.read_sql_query("""
        SELECT source_file, COUNT(*) as count 
        FROM products 
        GROUP BY source_file 
        ORDER BY count DESC
    """, conn)
    print(file_stats.to_string(index=False))
    
    # Price statistics
    print("\n--- Price Statistics ---")
    price_stats = pd.read_sql_query("""
        SELECT 
            MIN(price) as min_price,
            MAX(price) as max_price,
            AVG(price) as avg_price,
            COUNT(CASE WHEN price > 0 THEN 1 END) as products_with_price
        FROM products
    """, conn)
    print(price_stats.to_string(index=False))
    
    # Rating statistics
    print("\n--- Rating Statistics ---")
    rating_stats = pd.read_sql_query("""
        SELECT 
            MIN(rating_average) as min_rating,
            MAX(rating_average) as max_rating,
            AVG(rating_average) as avg_rating,
            COUNT(CASE WHEN rating_average > 0 THEN 1 END) as products_with_ratings
        FROM products
    """, conn)
    print(rating_stats.to_string(index=False))
    
    # Top brands
    print("\n--- Top 10 Brands ---")
    brand_stats = pd.read_sql_query("""
        SELECT brand, COUNT(*) as count 
        FROM products 
        WHERE brand != '' AND brand != 'OEM'
        GROUP BY brand 
        ORDER BY count DESC 
        LIMIT 10
    """, conn)
    print(brand_stats.to_string(index=False))
    
    conn.close()

def sample_products():
    """Display sample products from each category"""
    conn = connect_db()
    
    print("\n=== SAMPLE PRODUCTS BY CATEGORY ===\n")
    
    categories = pd.read_sql_query("SELECT DISTINCT category FROM products WHERE category != ''", conn)
    
    for category in categories['category']:
        print(f"--- {category} ---")
        sample = pd.read_sql_query("""
            SELECT product_id, name, price, rating_average, review_count
            FROM products 
            WHERE category = ? 
            ORDER BY review_count DESC 
            LIMIT 3
        """, conn, params=(category,))
        
        for _, row in sample.iterrows():
            print(f"  ID: {row['product_id']}")
            print(f"  Name: {row['name'][:80]}...")
            print(f"  Price: {row['price']:,} VND")
            print(f"  Rating: {row['rating_average']}/5 ({row['review_count']} reviews)")
            print()
    
    conn.close()

def search_products(keyword):
    """Search for products by keyword"""
    conn = connect_db()
    
    print(f"\n=== SEARCH RESULTS FOR: '{keyword}' ===\n")
    
    results = pd.read_sql_query("""
        SELECT product_id, name, price, category, rating_average, review_count
        FROM products 
        WHERE name LIKE ? OR description LIKE ?
        ORDER BY review_count DESC 
        LIMIT 10
    """, conn, params=(f'%{keyword}%', f'%{keyword}%'))
    
    if len(results) == 0:
        print("No products found.")
    else:
        for _, row in results.iterrows():
            print(f"ID: {row['product_id']} | {row['name'][:60]}...")
            print(f"Price: {row['price']:,} VND | Category: {row['category']} | Rating: {row['rating_average']}/5")
            print()
    
    conn.close()

def main():
    """Main function with interactive menu"""
    while True:
        print("\n" + "="*50)
        print("TIKI PRODUCTS DATABASE EXPLORER")
        print("="*50)
        print("1. Basic Statistics")
        print("2. Sample Products by Category")
        print("3. Search Products")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            basic_stats()
        elif choice == '2':
            sample_products()
        elif choice == '3':
            keyword = input("Enter search keyword: ").strip()
            if keyword:
                search_products(keyword)
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    # Run basic stats by default
    basic_stats()
    
    # Uncomment the line below for interactive mode
    # main()
