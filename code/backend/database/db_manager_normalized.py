import sqlite3
import os
import json
import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages normalized SQLite database with Tiki product dataset"""
    
    def __init__(self, db_path: str = "data/tiki_products_normalized.db"):
        self.db_path = db_path
        self.ensure_directory()
        self.init_database()
    
    def ensure_directory(self):
        """Ensure the database directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def init_database(self):
        """Initialize database with normalized schema - database should already exist from migration"""
        # Check if normalized database exists
        if not os.path.exists(self.db_path):
            logger.warning(f"Normalized database not found at {self.db_path}")
            logger.info("Run migration script to create normalized database")
            return
            
        # Verify normalized schema exists
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check for normalized tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['brands', 'categories', 'products', 'product_pricing', 'product_reviews', 'sellers']
        
        if not all(table in tables for table in expected_tables):
            logger.error(f"Missing normalized tables. Found: {tables}")
            logger.info("Run migration script to create normalized database")
        else:
            logger.info(f"Normalized database ready with tables: {tables}")
            
        conn.close()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results as list of dictionaries"""
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            return results
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            raise
        finally:
            conn.close()
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get comprehensive schema information for normalized database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        schema_info = {}
        
        try:
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                if table == 'sqlite_sequence':
                    continue
                    
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                schema_info[table] = {
                    "columns": [{"name": col[1], "type": col[2], "nullable": not col[3], "primary_key": bool(col[5])} for col in columns],
                    "row_count": row_count
                }
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            return {}
        finally:
            conn.close()
    
    def get_sample_data(self, table: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sample data from a specific table"""
        if table not in ['brands', 'categories', 'sellers', 'products', 'product_pricing', 'product_reviews']:
            raise ValueError(f"Invalid table name: {table}")
            
        query = f"SELECT * FROM {table} LIMIT ?"
        return self.execute_query(query, (limit,))
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # Total products
            cursor.execute("SELECT COUNT(*) FROM products")
            stats['total_products'] = cursor.fetchone()[0]
            
            # Total brands
            cursor.execute("SELECT COUNT(*) FROM brands")
            stats['total_brands'] = cursor.fetchone()[0]
            
            # Total categories
            cursor.execute("SELECT COUNT(*) FROM categories")
            stats['total_categories'] = cursor.fetchone()[0]
            
            # Total sellers
            cursor.execute("SELECT COUNT(*) FROM sellers")
            stats['total_sellers'] = cursor.fetchone()[0]
            
            # Average rating
            cursor.execute("SELECT AVG(rating_average) FROM product_reviews WHERE rating_average > 0")
            avg_rating = cursor.fetchone()[0]
            stats['average_rating'] = round(avg_rating, 2) if avg_rating else 0
            
            # Price range
            cursor.execute("SELECT MIN(current_price), MAX(current_price), AVG(current_price) FROM product_pricing WHERE current_price > 0")
            price_stats = cursor.fetchone()
            stats['price_range'] = {
                'min': price_stats[0] if price_stats[0] else 0,
                'max': price_stats[1] if price_stats[1] else 0,
                'avg': round(price_stats[2], 2) if price_stats[2] else 0
            }
            
            # Top categories by product count
            cursor.execute("""
                SELECT c.category_name, COUNT(p.product_id) as product_count
                FROM categories c
                LEFT JOIN products p ON c.category_id = p.category_id
                GROUP BY c.category_id, c.category_name
                ORDER BY product_count DESC
                LIMIT 5
            """)
            stats['top_categories'] = [{"name": row[0], "count": row[1]} for row in cursor.fetchall()]
            
            # Top brands by product count
            cursor.execute("""
                SELECT b.brand_name, COUNT(p.product_id) as product_count
                FROM brands b
                LEFT JOIN products p ON b.brand_id = p.brand_id
                GROUP BY b.brand_id, b.brand_name
                ORDER BY product_count DESC
                LIMIT 5
            """)
            stats['top_brands'] = [{"name": row[0], "count": row[1]} for row in cursor.fetchall()]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
        finally:
            conn.close()
    
    def search_products(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search products with JOIN across normalized tables"""
        search_query = """
            SELECT 
                p.product_id,
                p.name,
                p.description,
                b.brand_name,
                c.category_name,
                s.seller_name,
                pp.current_price,
                pp.original_price,
                pr.rating_average,
                pr.review_count
            FROM products p
            LEFT JOIN brands b ON p.brand_id = b.brand_id
            LEFT JOIN categories c ON p.category_id = c.category_id
            LEFT JOIN sellers s ON p.seller_id = s.seller_id
            LEFT JOIN product_pricing pp ON p.product_id = pp.product_id
            LEFT JOIN product_reviews pr ON p.product_id = pr.product_id
            WHERE p.name LIKE ? OR p.description LIKE ? OR b.brand_name LIKE ?
            ORDER BY pr.rating_average DESC, pp.current_price ASC
            LIMIT ?
        """
        
        search_term = f"%{query}%"
        return self.execute_query(search_query, (search_term, search_term, search_term, limit))
    
    def get_products_paginated(self, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """Get paginated products with JOIN data"""
        offset = (page - 1) * per_page
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM products"
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(count_query)
        total_count = cursor.fetchone()[0]
        conn.close()
        
        # Get paginated data
        products_query = """
            SELECT 
                p.product_id,
                p.name,
                p.description,
                b.brand_name,
                c.category_name,
                s.seller_name,
                pp.current_price,
                pp.original_price,
                pr.rating_average,
                pr.review_count,
                pp.quantity_sold
            FROM products p
            LEFT JOIN brands b ON p.brand_id = b.brand_id
            LEFT JOIN categories c ON p.category_id = c.category_id
            LEFT JOIN sellers s ON p.seller_id = s.seller_id
            LEFT JOIN product_pricing pp ON p.product_id = pp.product_id
            LEFT JOIN product_reviews pr ON p.product_id = pr.product_id
            ORDER BY p.product_id
            LIMIT ? OFFSET ?
        """
        
        products = self.execute_query(products_query, (per_page, offset))
        
        return {
            "products": products,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total_count,
                "pages": (total_count + per_page - 1) // per_page
            }
        }

# For backward compatibility
TikiDatabaseManager = DatabaseManager
