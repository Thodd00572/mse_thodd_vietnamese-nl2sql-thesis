#!/usr/bin/env python3
"""
Simple Tiki Database Explorer
Quick queries for exploring the imported Tiki product data
"""

import sqlite3
import pandas as pd
from pathlib import Path

# Database connection
db_path = Path(__file__).parent.parent / 'data' / 'tiki_products.db'
conn = sqlite3.connect(db_path)

print("=== TIKI DATABASE READY FOR QUERIES ===")
print(f"Database: {db_path}")
print(f"Total products: {pd.read_sql_query('SELECT COUNT(*) as count FROM products', conn).iloc[0]['count']:,}")

# Example queries you can run:
print("\n=== EXAMPLE QUERIES ===")
print("# High-rated products (>4.5 stars)")
print("SELECT name, price, rating_average, review_count FROM products WHERE rating_average > 4.5 ORDER BY review_count DESC LIMIT 10;")

print("\n# Most expensive products")
print("SELECT name, price, category, brand FROM products ORDER BY price DESC LIMIT 10;")

print("\n# Products by price range")
print("SELECT COUNT(*) as count, AVG(price) as avg_price FROM products WHERE price BETWEEN 100000 AND 500000;")

print("\n# Popular brands")
print("SELECT brand, COUNT(*) as products, AVG(rating_average) as avg_rating FROM products WHERE brand != 'OEM' GROUP BY brand ORDER BY products DESC LIMIT 15;")

conn.close()
