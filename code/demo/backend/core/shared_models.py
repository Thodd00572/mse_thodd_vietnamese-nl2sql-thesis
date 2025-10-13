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

def enrich_products_with_details(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich product data with brand, category, price, and rating information
    
    Args:
        products: List of product dictionaries (must have product_id field)
        
    Returns:
        List of enriched product dictionaries
    """
    logger.info(f"🎨 Starting enrichment for {len(products)} products...")
    enriched_products = []
    
    for product in products:
        enriched = {**product}  # Copy all existing fields
        
        # Get product_id for lookups
        product_id = product.get('product_id')
        
        if not product_id:
            logger.warning(f"⚠️ Product missing product_id, skipping enrichment: {product.get('name', 'unknown')[:50]}")
            enriched_products.append(enriched)
            continue
        
        try:
            # Enrich with pricing data
            if 'price' not in enriched or enriched.get('price') is None:
                pricing_query = "SELECT current_price, quantity_sold FROM product_pricing WHERE product_id = ?"
                pricing_results = db_manager.execute_query(pricing_query, (product_id,))
                if pricing_results:
                    enriched['price'] = pricing_results[0].get('current_price')
                    enriched['quantity_sold'] = pricing_results[0].get('quantity_sold', 0)
                else:
                    enriched['price'] = None
            
            # Enrich with brand name
            if 'brand' not in enriched or enriched.get('brand') is None:
                brand_id = product.get('brand_id')
                if brand_id:
                    brand_query = "SELECT brand_name FROM brands WHERE brand_id = ?"
                    brand_results = db_manager.execute_query(brand_query, (brand_id,))
                    if brand_results:
                        brand_name = brand_results[0].get('brand_name', 'Unknown Brand')
                        enriched['brand'] = 'No Brand' if brand_name == 'Generic Brand' else brand_name
                else:
                    enriched['brand'] = 'No Brand'
            
            # Enrich with category name
            if 'category' not in enriched or enriched.get('category') is None:
                category_id = product.get('category_id')
                if category_id:
                    category_query = "SELECT category_name FROM categories WHERE category_id = ?"
                    category_results = db_manager.execute_query(category_query, (category_id,))
                    if category_results:
                        enriched['category'] = category_results[0].get('category_name', 'Uncategorized')
                else:
                    enriched['category'] = 'Uncategorized'
            
            # Enrich with rating (handle NULL properly)
            if 'rating' not in enriched or enriched.get('rating') is None:
                rating_query = "SELECT rating_average, review_count FROM product_reviews WHERE product_id = ?"
                rating_results = db_manager.execute_query(rating_query, (product_id,))
                if rating_results and rating_results[0].get('rating_average') is not None:
                    enriched['rating'] = rating_results[0].get('rating_average')
                    enriched['review_count'] = rating_results[0].get('review_count', 0)
                else:
                    enriched['rating'] = None
                    enriched['review_count'] = 0
            
            # Add description if missing
            if 'description' not in enriched or not enriched.get('description'):
                enriched['description'] = product.get('description', 'No description available')
                
        except Exception as e:
            logger.error(f"❌ Enrichment error for product {product_id}: {e}")
        
        enriched_products.append(enriched)
    
    logger.info(f"✅ Enrichment complete. Enriched {len(enriched_products)} products")
    return enriched_products

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
        
        # Normalize field names to match frontend expectations
        if results and isinstance(results, list) and len(results) > 0:
            normalized_results = []
            for row in results:
                normalized_row = {**row}  # Copy original
                
                # Map SQL column names to frontend field names
                if 'brand_name' in row and 'brand' not in row:
                    normalized_row['brand'] = row['brand_name']
                if 'current_price' in row and 'price' not in row:
                    normalized_row['price'] = row['current_price']
                if 'rating_average' in row and 'rating' not in row:
                    normalized_row['rating'] = row['rating_average']
                if 'category_name' in row and 'category' not in row:
                    normalized_row['category'] = row['category_name']
                
                normalized_results.append(normalized_row)
            
            results = normalized_results
            logger.info(f"✅ Normalized field names. Sample: {list(results[0].keys())[:10]}")
        
        # Auto-enrich if results contain product_id field but missing enrichment data
        if enrich and results and isinstance(results, list) and len(results) > 0:
            logger.info(f"🔎 Checking if enrichment needed. First row keys: {list(results[0].keys())}")
            if 'product_id' in results[0] and 'brand' not in results[0]:
                logger.info(f"✅ product_id found! Starting enrichment for {len(results)} products...")
                try:
                    results = enrich_products_with_details(results)
                    logger.info(f"✅ Successfully enriched {len(results)} products")
                    logger.info(f"📝 Sample enriched data: {results[0] if results else 'None'}")
                except Exception as e:
                    logger.error(f"❌ Enrichment failed: {e}", exc_info=True)
                    logger.warning(f"Returning raw results without enrichment")
            else:
                logger.info(f"⏭️ Skipping enrichment - already has enrichment data or no product_id")
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
