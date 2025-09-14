"""
Production-ready FastAPI application for Vietnamese NL2SQL Pipeline
Optimized for Railway deployment with static file serving
"""

import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import existing modules
from api.routes import router as api_router
from api.database_routes import router as db_router
from database.db_manager_normalized import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vietnamese NL2SQL Translation API",
    description="Production API for Vietnamese to SQL translation using PhoBERT and SQLCoder models",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
db_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global db_manager
    
    try:
        # Initialize database
        logger.info("Initializing database connection")
        db_manager = DatabaseManager()
        
        # Verify database exists and has data
        schema_info = db_manager.get_schema_info()
        product_count = schema_info.get("products", {}).get("row_count", 0)
        logger.info(f"Database initialized with {product_count} products")
        
        # Set environment for Colab-only mode
        os.environ['SKIP_LOCAL_MODELS'] = 'true'
        logger.info("Production mode: Using Colab API for model inference")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Application shutting down")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    try:
        # Check database connection
        if db_manager:
            schema_info = db_manager.get_schema_info()
            product_count = schema_info.get("products", {}).get("row_count", 0)
            
            return {
                "status": "healthy",
                "database": "connected",
                "products": product_count,
                "mode": "production"
            }
        else:
            return {"status": "unhealthy", "error": "Database not initialized"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

# Include API routers
app.include_router(api_router, prefix="/api")
app.include_router(db_router, prefix="/api")

# Static file serving for frontend (if built)
frontend_build_path = Path(__file__).parent.parent / "frontend" / "out"
if frontend_build_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_build_path / "_next" / "static")), name="static")
    
    @app.get("/")
    async def serve_frontend():
        """Serve frontend index page"""
        index_file = frontend_build_path / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Vietnamese NL2SQL API", "docs": "/api/docs"}
    
    @app.get("/{path:path}")
    async def serve_frontend_routes(path: str):
        """Serve frontend routes"""
        # Try to serve the requested file
        file_path = frontend_build_path / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        
        # Try with .html extension
        html_path = frontend_build_path / f"{path}.html"
        if html_path.exists():
            return FileResponse(str(html_path))
        
        # Fallback to index.html for SPA routing
        index_file = frontend_build_path / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        
        raise HTTPException(status_code=404, detail="Page not found")
else:
    @app.get("/")
    async def root():
        """Root endpoint when no frontend build exists"""
        return {
            "message": "Vietnamese NL2SQL Translation API",
            "version": "1.0.0",
            "docs": "/api/docs",
            "health": "/health"
        }

if __name__ == "__main__":
    # Production server configuration
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting production server on {host}:{port}")
    
    uvicorn.run(
        "production_main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
