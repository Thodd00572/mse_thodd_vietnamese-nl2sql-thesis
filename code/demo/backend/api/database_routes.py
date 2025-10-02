from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sqlite3
import logging
from typing import List, Dict, Any, Optional
from database.db_manager_normalized import DatabaseManager

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize database manager
db_manager = DatabaseManager()

class QueryRequest(BaseModel):
    query: str

@router.get("/database/stats")
async def get_database_stats():
    """Get comprehensive database statistics from normalized database"""
    logger.info("Getting real database stats from normalized database")
    
    try:
        # Get real stats from database manager with timeout
        real_stats = db_manager.get_database_stats()
        
        if not real_stats:
            raise Exception("Database stats returned empty")
        
        # Format for frontend compatibility
        stats = {
            "totalProducts": [{"count": real_stats.get('total_products', 0)}],
            "brandCount": real_stats.get('total_brands', 0),
            "categoryStats": [
                {"category": cat["name"], "count": cat["count"]} 
                for cat in real_stats.get('top_categories', [])
            ],
            "fileStats": [],  # Will be populated from source_file data
            "priceStats": [{"avg_price": real_stats.get('price_range', {}).get('avg', 0)}]
        }
        
        # Get source file statistics with timeout
        try:
            source_stats = db_manager.execute_query("""
                SELECT source_file, COUNT(*) as count 
                FROM products 
                GROUP BY source_file 
                ORDER BY count DESC
            """)
            stats["fileStats"] = [{"source_file": row["source_file"], "count": row["count"]} for row in source_stats]
        except Exception as e:
            logger.warning(f"Failed to get source file stats: {e}")
            stats["fileStats"] = []
        
        logger.info(f"Real stats retrieved: {real_stats.get('total_products', 0)} products")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        # Simplified fallback without complex queries
        try:
            import sqlite3
            import os
            
            # Try to find the database
            possible_paths = [
                "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db",
                "data/tiki_products_normalized.db",
                "../../../data/tiki_products_normalized.db"
            ]
            
            db_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    db_path = path
                    break
            
            if not db_path:
                raise Exception("Database file not found in any expected location")
            
            conn = sqlite3.connect(db_path, timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM products")
            count = cursor.fetchone()[0]
            conn.close()
            
            return {
                "totalProducts": [{"count": count}],
                "brandCount": 0,
                "categoryStats": [],
                "fileStats": [],
                "priceStats": [{"avg_price": 0}]
            }
        except Exception as fallback_error:
            logger.error(f"Fallback query failed: {fallback_error}")
            raise HTTPException(status_code=500, detail=f"Database connection failed: {fallback_error}")

@router.post("/database/query")
async def execute_database_query(request: QueryRequest):
    """Execute custom SQL query"""
    try:
        # Find database path
        import os
        possible_paths = [
            "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/data/tiki_products_normalized.db",
            "data/tiki_products_normalized.db",
            "../../../data/tiki_products_normalized.db"
        ]
        
        db_path = None
        for path in possible_paths:
            if os.path.exists(path):
                db_path = path
                break
        
        if not db_path:
            raise HTTPException(status_code=500, detail="Database file not found")
        
        # Database connection with timeout
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Basic security check
        if any(dangerous in request.query.upper() for dangerous in ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER']):
            if not request.query.upper().strip().startswith('SELECT'):
                raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")
        
        cursor.execute(request.query)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        conn.close()
        
        return {"results": results, "query": request.query}
        
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/products")
async def get_products(page: int = 1, limit: int = 50):
    """Get paginated product data from normalized database"""
    try:
        # Use database manager for normalized data with JOINs
        result = db_manager.get_products_paginated(page, limit)
        
        # Format for API response
        return {
            "products": result["products"],
            "page": result["pagination"]["page"],
            "limit": result["pagination"]["per_page"],
            "total": result["pagination"]["total"]
        }
        
    except Exception as e:
        logger.error(f"Products fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch products: {str(e)}")
