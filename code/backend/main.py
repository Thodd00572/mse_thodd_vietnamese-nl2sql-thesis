from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api import routes, sample_query_routes
from database.db_manager_normalized import DatabaseManager
db_manager = DatabaseManager()
from models.pipelines import pipeline1, pipeline2
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vietnamese-to-SQL Translation Thesis Experiment",
    description="Experimental app comparing two pipelines for Vietnamese to SQL translation (Colab-only)",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(routes.router, prefix="/api")
app.include_router(sample_query_routes.router, prefix="/api/sample-queries")

# Include database management routes
from api.database_routes import router as database_router
app.include_router(database_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup - Skip local models, use Colab API only"""
    try:
        logger.info("Starting Vietnamese-to-SQL Translation Thesis Experiment API...")
        
        # Skip local model loading if environment variable is set
        skip_models = os.getenv('SKIP_LOCAL_MODELS', 'false').lower() == 'true'
        
        if skip_models:
            logger.info(" Colab-only mode: Skipping local model loading")
            logger.info(" All ML processing will use Google Colab API")
        else:
            logger.warning(" Local models disabled - set SKIP_LOCAL_MODELS=true for Colab-only mode")
        
        # Load Colab configuration
        try:
            config_file = "config/colab_config.json"
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    colab_config = json.load(f)
                
                # Configure pipeline URLs
                pipeline1.set_colab_url(colab_config.get("pipeline1_url", ""))
                pipeline2.set_colab_url(colab_config.get("pipeline2_url", ""))
                
                logger.info(f"Colab URLs configured: P1={colab_config.get('pipeline1_url')}, P2={colab_config.get('pipeline2_url')}")
            else:
                logger.warning("Colab config file not found - pipelines will not work without URLs")
        except Exception as e:
            logger.error(f"Failed to load Colab config: {e}")
        
        # Verify database
        schema = db_manager.get_schema_info()
        logger.info(f"Database ready with {len(schema)} tables")
        
        logger.info("Application startup completed successfully!")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Continue startup even if models fail to load (for development)

@app.get("/")
async def root():
    return {
        "message": "Vietnamese-to-SQL Translation Thesis Experiment API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/api/search",
            "analyze": "/api/analyze", 
            "export": "/api/export/{data_type}",
            "schema": "/api/schema",
            "models": "/api/models/status"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check - Colab-only mode"""
    try:
        # Skip local model checks in Colab-only mode
        skip_models = os.getenv('SKIP_LOCAL_MODELS', 'false').lower() == 'true'
        
        model_status = {
            "mode": "colab_api_only" if skip_models else "local_models",
            "local_models_loaded": False if skip_models else True,
            "colab_api_ready": "unknown"  # Will be checked by frontend
        }
        
        # Check database
        schema = db_manager.get_schema_info()
        
        # System metrics
        import psutil
        
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
        }
        
        return {
            "status": "healthy",
            "models": model_status,
            "database": {
                "connected": True,
                "tables": len(schema),
                "products_count": schema.get("products", {}).get("row_count", 0)
            },
            "system": system_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
