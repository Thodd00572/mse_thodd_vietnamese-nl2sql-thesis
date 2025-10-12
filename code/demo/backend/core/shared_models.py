# This file contains shared models, utilities, and functions used by the main FastAPI application
# The actual FastAPI app is defined in main.py

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sqlite3
import json
import time
import psutil
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
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

class LocalModelResult(BaseModel):
    pipeline_name: str = "Local Model"
    vietnamese_query: str
    sql_query: str
    results: List[Dict[str, Any]]
    execution_time: float
    success: bool
    error: Optional[str] = None
    model_used: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

class SearchResponse(BaseModel):
    vietnamese_query: str
    pipeline1_result: Optional[PipelineResult]
    pipeline2_result: Optional[Pipeline2Result]
    pipeline3_result: Optional[PipelineResult]
    local_model_result: Optional[LocalModelResult]
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

# Import the real DatabaseManager from database module
import sys
from pathlib import Path

# Add parent directory to path to import database module
current_dir = Path(__file__).parent
backend_root = current_dir.parent
sys.path.insert(0, str(backend_root))

from database.db_manager_normalized import DatabaseManager

# Initialize database manager with correct path
db_manager = DatabaseManager()

def get_system_metrics():
    """Get current system metrics"""
    metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "timestamp": datetime.now().isoformat()
    }
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        metrics.update({
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        })
    
    return metrics

def generate_query_id() -> str:
    """Generate unique query ID"""
    return f"q_{int(time.time() * 1000)}"

def enrich_products_with_details(products: List[Dict]) -> List[Dict]:
    """Enrich product data with brand, category, price, and rating information"""
    enriched = []
    
    logger.info(f"Starting enrichment for {len(products)} products")
    
    for product in products:
        product_id = product.get('product_id')
        enriched_product = {**product}
        
        logger.debug(f"Enriching product_id={product_id}, brand_id={product.get('brand_id')}, category_id={product.get('category_id')}")
        
        # Get price data
        try:
            pricing_sql = f"SELECT current_price, quantity_sold FROM product_pricing WHERE product_id = {product_id}"
            pricing_result = db_manager.execute_query(pricing_sql)
            
            if pricing_result and isinstance(pricing_result, list) and len(pricing_result) > 0:
                enriched_product['price'] = int(pricing_result[0]['current_price'])
                enriched_product['quantity_sold'] = int(pricing_result[0]['quantity_sold'])
            else:
                enriched_product['price'] = None
                enriched_product['quantity_sold'] = None
        except Exception as e:
            logger.warning(f"Failed to get pricing for product {product_id}: {e}")
            enriched_product['price'] = None
            enriched_product['quantity_sold'] = None
        
        # Get brand data
        try:
            brand_id = product.get('brand_id')
            if brand_id:
                brand_sql = f"SELECT brand_name FROM brands WHERE brand_id = {brand_id}"
                brand_result = db_manager.execute_query(brand_sql)
                
                if brand_result and isinstance(brand_result, list) and len(brand_result) > 0:
                    enriched_product['brand'] = str(brand_result[0]['brand_name'])
                else:
                    enriched_product['brand'] = 'Unknown'
            else:
                enriched_product['brand'] = 'Unknown'
        except Exception as e:
            logger.warning(f"Failed to get brand for product {product_id}: {e}")
            enriched_product['brand'] = 'Unknown'
        
        # Get category data
        try:
            category_id = product.get('category_id')
            logger.debug(f"  🏷️ Looking up category_id={category_id}")
            if category_id:
                category_sql = f"SELECT category_name FROM categories WHERE category_id = {category_id}"
                logger.debug(f"  📝 Category SQL: {category_sql}")
                category_result = db_manager.execute_query(category_sql)
                logger.debug(f"  📊 Category result: {category_result}")
                
                if category_result and isinstance(category_result, list) and len(category_result) > 0:
                    enriched_product['category'] = str(category_result[0]['category_name'])
                    logger.info(f"  ✅ Category enriched: {enriched_product['category']}")
                else:
                    enriched_product['category'] = 'Unknown'
                    logger.warning(f"  ⚠️ No category found for category_id={category_id}")
            else:
                enriched_product['category'] = 'Unknown'
                logger.warning(f"  ⚠️ Product {product_id} has no category_id")
        except Exception as e:
            logger.error(f"  ❌ Failed to get category for product {product_id}: {e}", exc_info=True)
            enriched_product['category'] = 'Unknown'
        
        # Get rating data
        try:
            rating_sql = f"SELECT rating_average, review_count FROM product_reviews WHERE product_id = {product_id}"
            rating_result = db_manager.execute_query(rating_sql)
            
            if rating_result and isinstance(rating_result, list) and len(rating_result) > 0:
                enriched_product['rating'] = float(rating_result[0]['rating_average'])
                enriched_product['review_count'] = int(rating_result[0]['review_count'])
            else:
                enriched_product['rating'] = None
                enriched_product['review_count'] = None
        except Exception as e:
            logger.warning(f"Failed to get rating for product {product_id}: {e}")
            enriched_product['rating'] = None
            enriched_product['review_count'] = None
        
        enriched.append(enriched_product)
    
    return enriched

def execute_sql_query(sql_query: str, enrich: bool = True) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Execute SQL query and return results with error handling
    
    Args:
        sql_query: SQL query to execute
        enrich: If True and results contain product_id, enrich with brand/category/price/rating
    
    Returns:
        Tuple of (results, error_message)
    """
    try:
        logger.info(f"🔍 Executing SQL: {sql_query[:100]}...")
        results = db_manager.execute_query(sql_query)
        logger.info(f"📊 Got {len(results) if results else 0} raw results")
        
        # Auto-enrich if results contain product_id field
        if enrich and results and isinstance(results, list) and len(results) > 0:
            logger.info(f"🔎 Checking if enrichment needed. First row keys: {list(results[0].keys())}")
            if 'product_id' in results[0]:
                logger.info(f"✅ product_id found! Starting enrichment for {len(results)} products...")
                try:
                    results = enrich_products_with_details(results)
                    logger.info(f"✅ Successfully enriched {len(results)} products")
                    logger.info(f"📝 Sample enriched data: {results[0] if results else 'None'}")
                except Exception as e:
                    logger.error(f"❌ Enrichment failed: {e}", exc_info=True)
                    logger.warning(f"Returning raw results without enrichment")
            else:
                logger.info(f"⏭️ Skipping enrichment - no product_id in results")
        else:
            logger.info(f"⏭️ Skipping enrichment - enrich={enrich}, results={len(results) if results else 0}")
        
        return results, None
    except Exception as e:
        logger.error(f"❌ SQL execution error: {e}", exc_info=True)
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
