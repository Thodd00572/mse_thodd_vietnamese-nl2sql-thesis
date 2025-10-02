#!/usr/bin/env python3
"""
Database Schema Verification Script
Tests both Colab and local databases to ensure they support all evaluation query types.
"""

import sqlite3
import sys
from pathlib import Path

def test_database(db_path, db_name):
    """Test a database for proper multi-table schema support"""
    print(f"\n🔍 Testing {db_name}: {db_path}")
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Test 1: Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = ['products', 'brands', 'categories', 'product_pricing', 'product_reviews']
        
        print(f"📋 Tables found: {', '.join(tables)}")
        
        missing_tables = [t for t in expected_tables if t not in tables]
        if missing_tables:
            print(f"❌ Missing tables: {', '.join(missing_tables)}")
            return False
        
        # Test 2: Check data counts
        for table in expected_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = cursor.fetchone()[0]
            print(f"   {table}: {count:,} records")
            if count == 0:
                print(f"⚠️  Warning: {table} is empty")
        
        # Test 3: Simple product search (should work)
        print("\n🧪 Testing Simple Queries:")
        cursor.execute("SELECT COUNT(*) FROM products WHERE name LIKE '%balo%';")
        balo_count = cursor.fetchone()[0]
        print(f"✅ Simple search: Found {balo_count} products with 'balo'")
        
        # Test 4: Brand JOIN (medium complexity)
        print("\n🧪 Testing Medium Queries (JOINs):")
        try:
            cursor.execute("""
                SELECT p.name, b.brand_name 
                FROM products p 
                JOIN brands b ON p.brand_id = b.brand_id 
                WHERE b.brand_name LIKE '%Nike%' 
                LIMIT 5;
            """)
            nike_products = cursor.fetchall()
            print(f"✅ Brand JOIN: Found {len(nike_products)} Nike products")
        except Exception as e:
            print(f"❌ Brand JOIN failed: {e}")
            return False
        
        # Test 5: Price filtering (medium complexity)
        try:
            cursor.execute("""
                SELECT p.name, pr.current_price 
                FROM products p 
                JOIN product_pricing pr ON p.product_id = pr.product_id 
                WHERE pr.current_price < 100000 
                LIMIT 5;
            """)
            cheap_products = cursor.fetchall()
            print(f"✅ Price JOIN: Found {len(cheap_products)} products under 100k")
        except Exception as e:
            print(f"❌ Price JOIN failed: {e}")
            return False
        
        # Test 6: Rating filtering (medium complexity)
        try:
            cursor.execute("""
                SELECT p.name, rv.rating_average 
                FROM products p 
                JOIN product_reviews rv ON p.product_id = rv.product_id 
                WHERE rv.rating_average >= 4.0 
                LIMIT 5;
            """)
            good_products = cursor.fetchall()
            print(f"✅ Rating JOIN: Found {len(good_products)} products with 4+ stars")
        except Exception as e:
            print(f"❌ Rating JOIN failed: {e}")
            return False
        
        # Test 7: Complex multi-table JOIN (complex queries)
        print("\n🧪 Testing Complex Queries (Multi-JOINs):")
        try:
            cursor.execute("""
                SELECT p.name, b.brand_name, c.category_name, pr.current_price, rv.rating_average
                FROM products p 
                JOIN brands b ON p.brand_id = b.brand_id 
                JOIN categories c ON p.category_id = c.category_id 
                JOIN product_pricing pr ON p.product_id = pr.product_id 
                JOIN product_reviews rv ON p.product_id = rv.product_id 
                WHERE pr.current_price < 1000000 AND rv.rating_average >= 4.0 
                ORDER BY rv.rating_average DESC 
                LIMIT 5;
            """)
            complex_results = cursor.fetchall()
            print(f"✅ Complex JOIN: Found {len(complex_results)} high-rated products under 1M")
            
            # Show sample result
            if complex_results:
                sample = complex_results[0]
                print(f"   Sample: {sample[0][:50]}... | {sample[1]} | {sample[2]} | {sample[3]:,}đ | {sample[4]}⭐")
        except Exception as e:
            print(f"❌ Complex JOIN failed: {e}")
            return False
        
        # Test 8: Aggregation queries
        try:
            cursor.execute("""
                SELECT b.brand_name, COUNT(p.product_id) as product_count
                FROM brands b 
                JOIN products p ON b.brand_id = p.brand_id 
                GROUP BY b.brand_name 
                ORDER BY product_count DESC 
                LIMIT 5;
            """)
            brand_stats = cursor.fetchall()
            print(f"✅ Aggregation: Top brands by product count")
            for brand, count in brand_stats:
                print(f"   {brand}: {count} products")
        except Exception as e:
            print(f"❌ Aggregation failed: {e}")
            return False
        
        # Test 9: Check products_with_price view (if exists)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='products_with_price';")
        if cursor.fetchone():
            try:
                cursor.execute("""
                    SELECT name, brand_name, category_name, current_price, rating_average 
                    FROM products_with_price 
                    WHERE current_price < 100000 AND rating_average > 4.0 
                    LIMIT 3;
                """)
                view_results = cursor.fetchall()
                print(f"✅ products_with_price view: Found {len(view_results)} results")
            except Exception as e:
                print(f"❌ products_with_price view failed: {e}")
        else:
            print("ℹ️  products_with_price view not found (optional)")
        
        conn.close()
        print(f"✅ {db_name} database verification PASSED")
        return True
        
    except Exception as e:
        print(f"❌ {db_name} database verification FAILED: {e}")
        return False

def main():
    """Test both databases"""
    print("🚀 Database Schema Verification for Vietnamese NL2SQL Evaluation")
    print("=" * 70)
    
    # Test Colab database
    colab_db = "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/ColabNotebook/db/tiki.sqlite"
    colab_ok = test_database(colab_db, "Colab Database")
    
    # Test local database  
    local_db = "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db"
    local_ok = test_database(local_db, "Local Database")
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS:")
    print(f"   Colab Database: {'✅ READY' if colab_ok else '❌ NEEDS FIXING'}")
    print(f"   Local Database: {'✅ READY' if local_ok else '❌ NEEDS FIXING'}")
    
    if colab_ok and local_ok:
        print("\n🎉 Both databases are ready for Vietnamese NL2SQL evaluation!")
        print("   - Simple queries: ✅ Supported")
        print("   - Medium queries (JOINs): ✅ Supported") 
        print("   - Complex queries (Multi-JOINs): ✅ Supported")
        print("   - Aggregation queries: ✅ Supported")
        return True
    else:
        print("\n⚠️  Some databases need fixing before evaluation can proceed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
