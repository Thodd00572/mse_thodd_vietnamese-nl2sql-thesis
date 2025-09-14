import sqlite3
import os
import json
import pandas as pd
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TikiDatabaseManager:
    """Manages SQLite database with Tiki product dataset"""
    
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
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_price ON products(price)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_rating ON products(rating_average)")
        
        # Insert sample data if tables are empty
        cursor.execute("SELECT COUNT(*) FROM categories")
        categories_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM products")
        products_count = cursor.fetchone()[0]
        
        if categories_count == 0 or products_count == 0:
            self._insert_sample_data(cursor)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def _insert_sample_data(self, cursor):
        """Insert comprehensive Tiki product sample data"""
        
        # Check if categories need to be inserted
        cursor.execute("SELECT COUNT(*) FROM categories")
        if cursor.fetchone()[0] == 0:
            # Categories data
            categories = [
                (1, "Điện thoại & Phụ kiện", None, 1),
                (2, "Laptop & Máy tính", None, 1),
                (3, "Thời trang", None, 1),
                (4, "Điện tử & Điện lạnh", None, 1),
                (5, "Nhà cửa & Đời sống", None, 1),
                (6, "Sách & Văn phòng phẩm", None, 1),
                (7, "Smartphone", 1, 2),
                (8, "Tai nghe", 1, 2),
                (9, "Ốp lưng & Bao da", 1, 2),
                (10, "Laptop", 2, 2),
                (11, "Chuột & Bàn phím", 2, 2),
                (12, "Màn hình", 2, 2),
                (13, "Áo nam", 3, 2),
                (14, "Giày dép", 3, 2),
                (15, "Tủ lạnh", 4, 2),
                (16, "Máy giặt", 4, 2),
            ]
            
            cursor.executemany("INSERT INTO categories (id, name, parent_id, level) VALUES (?, ?, ?, ?)", categories)
        
        # Check if products need to be inserted
        cursor.execute("SELECT COUNT(*) FROM products")
        if cursor.fetchone()[0] == 0:
            # Products data - comprehensive Tiki-style dataset
            products = [
            # Smartphones
            (1, "iPhone 15 Pro Max 256GB Titan Tự Nhiên", 29990000, 7, "Apple", "iPhone 15 Pro Max với chip A17 Pro, camera 48MP, màn hình Super Retina XDR 6.7 inch", 4.8, 1250, True, 5, 31490000, "iphone15.jpg", "Apple Store", "2024-01-15"),
            (2, "Samsung Galaxy S24 Ultra 256GB", 26990000, 7, "Samsung", "Galaxy S24 Ultra với S Pen, camera 200MP, màn hình Dynamic AMOLED 6.8 inch", 4.7, 890, True, 10, 29990000, "s24ultra.jpg", "Samsung Store", "2024-02-01"),
            (3, "Xiaomi Redmi Note 13 Pro 8GB/256GB", 6990000, 7, "Xiaomi", "Redmi Note 13 Pro với camera 200MP, sạc nhanh 67W, màn hình AMOLED 6.67 inch", 4.5, 2340, True, 15, 8190000, "redmi13.jpg", "Xiaomi Store", "2024-01-20"),
            (4, "OPPO Reno11 F 5G 8GB/256GB", 7990000, 7, "OPPO", "OPPO Reno11 F với camera selfie 32MP, sạc nhanh SuperVOOC 67W", 4.4, 567, True, 12, 8990000, "reno11f.jpg", "OPPO Store", "2024-02-10"),
            (5, "Vivo V30e 8GB/256GB", 8490000, 7, "Vivo", "Vivo V30e với camera chân dung 50MP, thiết kế mỏng nhẹ", 4.3, 423, True, 8, 9190000, "v30e.jpg", "Vivo Store", "2024-01-25"),
            
            # Laptops
            (6, "MacBook Air M2 13 inch 8GB/256GB", 24990000, 10, "Apple", "MacBook Air với chip M2, màn hình Liquid Retina 13.6 inch, pin 18 giờ", 4.9, 567, True, 8, 26990000, "macbookair.jpg", "Apple Store", "2024-01-10"),
            (7, "Dell XPS 13 9320 i7/16GB/512GB", 32990000, 10, "Dell", "Dell XPS 13 với Intel Core i7 Gen 12, màn hình InfinityEdge 13.4 inch", 4.6, 234, True, 12, 37490000, "xps13.jpg", "Dell Store", "2024-01-18"),
            (8, "ASUS ZenBook 14 OLED i5/8GB/512GB", 18990000, 10, "ASUS", "ASUS ZenBook 14 với màn hình OLED 2.8K, Intel Core i5 Gen 12", 4.5, 345, True, 15, 22290000, "zenbook14.jpg", "ASUS Store", "2024-02-05"),
            (9, "HP Pavilion 15 i7/8GB/512GB", 16990000, 10, "HP", "HP Pavilion 15 với Intel Core i7, card đồ họa NVIDIA GTX 1650", 4.3, 456, True, 10, 18890000, "pavilion15.jpg", "HP Store", "2024-01-30"),
            (10, "Lenovo ThinkPad E14 i5/8GB/256GB", 14990000, 10, "Lenovo", "Lenovo ThinkPad E14 dành cho doanh nghiệp, độ bền cao", 4.4, 289, True, 18, 18290000, "thinkpad.jpg", "Lenovo Store", "2024-02-12"),
            
            # Headphones
            (11, "AirPods Pro 2 (USB-C)", 5990000, 8, "Apple", "AirPods Pro thế hệ 2 với chip H2, chống ồn chủ động", 4.8, 1890, True, 7, 6390000, "airpods.jpg", "Apple Store", "2024-01-12"),
            (12, "Sony WH-1000XM5", 7990000, 8, "Sony", "Tai nghe chống ồn cao cấp với thời lượng pin 30 giờ", 4.9, 456, True, 15, 9390000, "sony1000xm5.jpg", "Sony Store", "2024-01-22"),
            (13, "JBL Tune 770NC", 1990000, 8, "JBL", "Tai nghe chống ồn JBL với bass mạnh mẽ", 4.2, 678, True, 20, 2490000, "jbl770.jpg", "JBL Store", "2024-02-08"),
            (14, "Logitech G Pro X", 2990000, 8, "Logitech", "Tai nghe gaming chuyên nghiệp với micro Blue VO!CE", 4.6, 234, True, 12, 3390000, "gprox.jpg", "Logitech Store", "2024-01-28"),
            
            # Accessories
            (15, "Logitech MX Master 3S", 1990000, 11, "Logitech", "Chuột không dây cao cấp cho năng suất làm việc", 4.7, 567, True, 10, 2190000, "mxmaster3s.jpg", "Logitech Store", "2024-02-03"),
            (16, "Keychron K3 Wireless", 2490000, 11, "Keychron", "Bàn phím cơ không dây siêu mỏng với switch Gateron", 4.5, 345, True, 15, 2890000, "k3.jpg", "Keychron Store", "2024-01-16"),
            (17, "LG UltraWide 29WP60G 29 inch", 5990000, 12, "LG", "Màn hình cong UltraWide 21:9 cho đa nhiệm", 4.4, 123, True, 20, 7490000, "lg29wp60g.jpg", "LG Store", "2024-02-14"),
            
            # Home appliances
            (18, "Tủ lạnh Samsung Inverter 236L", 6990000, 15, "Samsung", "Tủ lạnh 2 cửa tiết kiệm điện với công nghệ Inverter", 4.3, 234, True, 12, 7890000, "samsung236l.jpg", "Samsung Store", "2024-01-20"),
            (19, "Máy giặt LG Inverter 9kg", 8990000, 16, "LG", "Máy giặt cửa trước tiết kiệm nước và điện", 4.5, 156, True, 15, 10490000, "lg9kg.jpg", "LG Store", "2024-02-06"),
            (20, "Điều hòa Daikin Inverter 1HP", 12990000, 4, "Daikin", "Điều hòa 1 chiều tiết kiệm điện, làm lạnh nhanh", 4.6, 345, True, 8, 14090000, "daikin1hp.jpg", "Daikin Store", "2024-01-14"),
            ]
            
            cursor.executemany("""
                INSERT INTO products (id, name, price, category_id, brand, description, rating, review_count, in_stock, discount_percent, original_price, image_url, seller_name, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, products)
            
            logger.info("Sample Tiki product data inserted successfully")
    
    def execute_query(self, sql_query: str, timeout: int = 10) -> List[Dict[str, Any]]:
        """Execute SQL query and return results with timeout"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=timeout)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Security: Basic SQL injection prevention
            if any(dangerous in sql_query.upper() for dangerous in ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER']):
                if not sql_query.upper().strip().startswith('SELECT'):
                    raise ValueError("Only SELECT queries are allowed for security")
            
            # Set a query timeout
            cursor.execute("PRAGMA busy_timeout = 5000")  # 5 second timeout
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            results = [dict(row) for row in rows]
            conn.close()
            
            logger.info(f"Executed query: {sql_query[:100]}... | Results: {len(results)} rows")
            return results
            
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                logger.error(f"Database locked error: {e}")
                # Return empty result instead of failing
                return []
            else:
                logger.error(f"SQL operational error: {e}")
                raise
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            raise
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        schema = {}
        
        # Get table info
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            schema[table_name] = {
                "columns": [{"name": col[1], "type": col[2], "nullable": not col[3]} for col in columns],
                "row_count": cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            }
        
        conn.close()
        return schema
    
    def export_to_csv(self, query: str, filename: str) -> str:
        """Export query results to CSV"""
        try:
            results = self.execute_query(query)
            if results:
                df = pd.DataFrame(results)
                csv_path = f"exports/{filename}"
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                df.to_csv(csv_path, index=False, encoding='utf-8')
                return csv_path
            else:
                raise ValueError("No results to export")
                
        except Exception as e:
            logger.error(f"CSV export error: {e}")
            raise

# Global database manager instance
db_manager = TikiDatabaseManager()
