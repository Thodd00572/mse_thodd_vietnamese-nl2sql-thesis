# This file contains shared models, utilities, and functions used by the main FastAPI application
# The actual FastAPI app is defined in main.py

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sqlite3
import json
import time
import psutil
import torch
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared Pydantic models used across the application
class SearchRequest(BaseModel):
    query: str
    pipeline: Optional[str] = "both"

class PipelineResult(BaseModel):
    pipeline_name: str
    sql_query: str
    english_query: Optional[str] = None
    results: List[Dict[str, Any]]
    execution_time: float
    success: bool
    error: Optional[str] = None
    metrics: Dict[str, Any]

class Pipeline2Result(BaseModel):
    pipeline_name: str = "Pipeline 2"
    vietnamese_query: str
    english_query: Optional[str] = ""
    sql_query: str
    results: List[Dict[str, Any]]
    execution_time: float
    vn_en_time: float = 0.0
    en_sql_time: float = 0.0
    success: bool
    error: Optional[str] = None
    # Research metrics
    execution_accuracy: Optional[float] = None
    exact_match: Optional[bool] = None
    latency_ms: Optional[float] = None
    gpu_cost: Optional[Dict[str, float]] = None
    error_type: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    vietnamese_query: str
    pipeline1_result: Optional[PipelineResult]
    pipeline2_result: Optional[Pipeline2Result]
    timestamp: str
    query_id: str
    system_metrics: Optional[Dict[str, Any]] = None

class MetricsResponse(BaseModel):
    total_queries: int
    pipeline1_metrics: Dict[str, Any]
    pipeline2_metrics: Dict[str, Any]
    comparison_metrics: Dict[str, Any]
    system_metrics: Dict[str, Any]

# Global experiment data - shared across the application
experiment_data = {
    "queries": [],
    "pipeline1_stats": {"success": 0, "errors": 0, "total_time": 0, "sql_queries": []},
    "pipeline2_stats": {"success": 0, "errors": 0, "total_time": 0, "sql_queries": []},
    "start_time": datetime.now().isoformat()
}

class DatabaseManager:
    def __init__(self, db_path: str = "data/tiki_products_normalized.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database connection"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return results
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise e
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            schema = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                
                schema[table] = {
                    "columns": [{"name": col[1], "type": col[2]} for col in columns],
                    "row_count": row_count
                }
            
            conn.close()
            return schema
        except Exception as e:
            logger.error(f"Schema retrieval error: {e}")
            return {}

# Initialize database manager
db_manager = DatabaseManager()

def get_system_metrics():
    """Get current system metrics"""
    metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "timestamp": datetime.now().isoformat()
    }
    
    if torch.cuda.is_available():
        metrics.update({
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        })
    
    return metrics

def generate_query_id() -> str:
    """Generate unique query ID"""
    return f"q_{int(time.time() * 1000)}"

def execute_sql_query(sql_query: str) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Execute SQL query and return results with error handling"""
    try:
        results = db_manager.execute_query(sql_query)
        return results, None
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        return [], str(e)

def enhanced_vietnamese_to_sql(vietnamese_query: str) -> str:
    """Enhanced Vietnamese to SQL translation for Pipeline 1 fallback"""
    from models.pipelines import pipeline1
    import asyncio
    
    # Create event loop if none exists
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run the async method
    result = loop.run_until_complete(pipeline1.vietnamese_to_sql(vietnamese_query))
    return result.get("sql_query", "SELECT * FROM products LIMIT 10")
