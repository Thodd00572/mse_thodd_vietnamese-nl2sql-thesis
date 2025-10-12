"""
Colab Proxy Routes - Forward requests to Colab API and execute SQL on local DB
Handles P1 (mT5), P2 (SQLCoder), and P3 (Vanna AI) endpoints
"""

from typing import Optional, List, Dict, Any
import httpx
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import time

from core.shared_models import db_manager, enrich_products_with_details

logger = logging.getLogger(__name__)
router = APIRouter()

# Colab API base URL - will be configured from config file
COLAB_BASE_URL = "https://abnormally-direct-rhino.ngrok-free.app"

class QueryRequest(BaseModel):
    query: str

class PipelineResponse(BaseModel):
    pipeline: str
    vietnamese_query: str
    sql_query: str
    execution_time: float
    valid: bool
    success: bool
    error: Optional[str] = None
    metrics: Dict[str, Any]
    database_results: Optional[Dict[str, Any]] = None
    results: Optional[List[Dict[str, Any]]] = None  # For frontend compatibility

def execute_sql_on_local_db(sql: str) -> Dict[str, Any]:
    """Execute generated SQL on local database and return results"""
    try:
        # Clean up SQL
        sql = sql.strip()
        if not sql:
            return {
                "success": False,
                "error": "Empty SQL query",
                "rows": [],
                "row_count": 0
            }
        
        # Execute query
        start_time = time.time()
        results = db_manager.execute_query(sql)
        execution_time = time.time() - start_time
        
        if results is None:
            return {
                "success": False,
                "error": "Query execution failed",
                "rows": [],
                "row_count": 0,
                "execution_time_ms": execution_time * 1000
            }
        
        # Convert results to list of dicts
        rows = []
        if hasattr(results, 'to_dict'):
            # DataFrame
            rows = results.to_dict('records')
        elif isinstance(results, list):
            rows = results
        
        return {
            "success": True,
            "rows": rows,
            "row_count": len(rows),
            "execution_time_ms": execution_time * 1000,
            "columns": list(rows[0].keys()) if rows else []
        }
        
    except Exception as e:
        logger.error(f"Database execution error: {e}")
        return {
            "success": False,
            "error": str(e),
            "rows": [],
            "row_count": 0
        }

# Note: enrich_products_with_details is now imported from shared_models

async def call_colab_pipeline(endpoint: str, query: str, pipeline_name: str) -> PipelineResponse:
    """Call Colab API endpoint and execute SQL on local database"""
    try:
        # Call Colab API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{COLAB_BASE_URL}{endpoint}",
                json={"query": query},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Colab API returned status {response.status_code}: {response.text}"
                )
            
            colab_result = response.json()
        
        # Extract SQL from Colab response
        sql_query = colab_result.get("sql_query", "")
        
        # Execute SQL on local database
        db_results = None
        enriched_results = []
        
        if sql_query and sql_query.strip():
            db_results = execute_sql_on_local_db(sql_query)
            
            # Enrich results with pricing/brand/category/rating data if we have products
            if db_results and db_results.get("success") and db_results.get("rows"):
                try:
                    enriched_results = enrich_products_with_details(db_results["rows"])
                except Exception as e:
                    logger.warning(f"Failed to enrich products: {e}")
                    enriched_results = db_results["rows"]
        
        # Build response
        return PipelineResponse(
            pipeline=pipeline_name,
            vietnamese_query=query,
            sql_query=sql_query,
            execution_time=colab_result.get("execution_time", 0),
            valid=colab_result.get("valid", False),
            success=colab_result.get("success", False),
            error=colab_result.get("error"),
            metrics=colab_result.get("metrics", {}),
            database_results=db_results,
            results=enriched_results  # Frontend expects this
        )
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Colab API timeout")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Colab API connection error: {str(e)}")
    except Exception as e:
        logger.error(f"Pipeline {pipeline_name} error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/p1/generate", response_model=PipelineResponse)
async def generate_p1(request: QueryRequest):
    """
    P1: mT5 Zero-Shot Pipeline
    Forwards request to Colab, gets SQL, executes on local DB
    """
    logger.info(f"P1 request: {request.query}")
    return await call_colab_pipeline("/p1/generate", request.query, "P1_mT5_Zero_Shot")

@router.post("/p2/generate", response_model=PipelineResponse)
async def generate_p2(request: QueryRequest):
    """
    P2: SQLCoder Zero-Shot Pipeline
    Forwards request to Colab, gets SQL, executes on local DB
    """
    logger.info(f"P2 request: {request.query}")
    return await call_colab_pipeline("/p2/generate", request.query, "P2_SQLCoder_Zero_Shot")

@router.post("/p3/generate", response_model=PipelineResponse)
async def generate_p3(request: QueryRequest):
    """
    P3: Vanna AI RAG Pipeline
    Forwards request to Colab, gets SQL, executes on local DB
    """
    logger.info(f"P3 request: {request.query}")
    return await call_colab_pipeline("/p3/generate", request.query, "P3_Vanna_AI_RAG")

@router.get("/p1/metrics")
async def get_p1_metrics():
    """Get P1 metrics from Colab"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{COLAB_BASE_URL}/p1/metrics")
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch P1 metrics: {str(e)}")

@router.get("/p2/metrics")
async def get_p2_metrics():
    """Get P2 metrics from Colab"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{COLAB_BASE_URL}/p2/metrics")
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch P2 metrics: {str(e)}")

@router.get("/p3/metrics")
async def get_p3_metrics():
    """Get P3 metrics from Colab"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{COLAB_BASE_URL}/p3/metrics")
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch P3 metrics: {str(e)}")

@router.get("/compare/metrics")
async def compare_metrics():
    """Compare metrics across all pipelines from Colab"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{COLAB_BASE_URL}/compare/metrics")
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch comparison metrics: {str(e)}")

@router.get("/config/colab/status")
async def get_colab_status():
    """Get Colab server status"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{COLAB_BASE_URL}/health")
            colab_data = response.json()
            
            return {
                "colab_url": COLAB_BASE_URL,
                "colab_status": "connected" if response.status_code == 200 else "disconnected",
                "colab_health": colab_data,
                "local_db_status": "connected",
                "local_db_tables": len(db_manager.get_schema_info()),
                "pipelines": {
                    "p1": {"name": "mT5 Zero-Shot", "status": "ready"},
                    "p2": {"name": "SQLCoder Zero-Shot", "status": "ready"},
                    "p3": {"name": "Vanna AI RAG", "status": "ready"}
                }
            }
    except Exception as e:
        return {
            "colab_url": COLAB_BASE_URL,
            "colab_status": "disconnected",
            "error": str(e),
            "local_db_status": "connected",
            "local_db_tables": len(db_manager.get_schema_info())
        }

def set_colab_url(url: str):
    """Update Colab base URL"""
    global COLAB_BASE_URL
    COLAB_BASE_URL = url
    logger.info(f"Colab URL updated to: {url}")
