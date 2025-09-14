#!/usr/bin/env python3
"""
Minimal configuration server for Colab URL management
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import os
import httpx
import uvicorn

app = FastAPI(title="Colab Configuration API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Configuration Models
class ColabConfig(BaseModel):
    pipeline1_url: Optional[str] = ""
    pipeline2_url: Optional[str] = ""

class ColabStatus(BaseModel):
    pipeline1_url: Optional[str] = ""
    pipeline2_url: Optional[str] = ""
    pipeline1_healthy: bool = False
    pipeline2_healthy: bool = False

# Global configuration storage
colab_config = {
    "pipeline1_url": "",
    "pipeline2_url": ""
}

async def check_pipeline_health(url: str) -> bool:
    """Check if a pipeline URL is healthy"""
    if not url:
        return False
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/health")
            return response.status_code == 200
    except Exception:
        return False

@app.get("/")
async def root():
    return {
        "message": "Colab Configuration Server",
        "status": "running",
        "endpoints": ["/config/colab/status", "/config/colab"]
    }

@app.get("/config/colab/status")
async def get_colab_status():
    """Get current Colab configuration and health status"""
    pipeline1_healthy = await check_pipeline_health(colab_config["pipeline1_url"])
    pipeline2_healthy = await check_pipeline_health(colab_config["pipeline2_url"])
    
    status = ColabStatus(
        pipeline1_url=colab_config["pipeline1_url"],
        pipeline2_url=colab_config["pipeline2_url"],
        pipeline1_healthy=pipeline1_healthy,
        pipeline2_healthy=pipeline2_healthy
    )
    
    return {"status": status.dict()}

@app.post("/config/colab")
async def save_colab_config(config: ColabConfig):
    """Save Colab configuration"""
    global colab_config
    
    # Update global config
    colab_config["pipeline1_url"] = config.pipeline1_url.rstrip('/') if config.pipeline1_url else ""
    colab_config["pipeline2_url"] = config.pipeline2_url.rstrip('/') if config.pipeline2_url else ""
    
    # Save to file for persistence
    config_file = "config/colab_config.json"
    os.makedirs("config", exist_ok=True)
    
    try:
        with open(config_file, 'w') as f:
            json.dump(colab_config, f, indent=2)
        print(f"✅ Colab configuration saved: {colab_config}")
    except Exception as e:
        print(f"❌ Failed to save config file: {e}")
    
    # Check health status
    pipeline1_healthy = await check_pipeline_health(colab_config["pipeline1_url"])
    pipeline2_healthy = await check_pipeline_health(colab_config["pipeline2_url"])
    
    status = ColabStatus(
        pipeline1_url=colab_config["pipeline1_url"],
        pipeline2_url=colab_config["pipeline2_url"],
        pipeline1_healthy=pipeline1_healthy,
        pipeline2_healthy=pipeline2_healthy
    )
    
    return {"status": status.dict(), "message": "Configuration saved successfully"}

# Load configuration on startup
def load_colab_config():
    """Load Colab configuration from file"""
    global colab_config
    config_file = "config/colab_config.json"
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                colab_config.update(loaded_config)
                print(f"📋 Colab configuration loaded: {colab_config}")
    except Exception as e:
        print(f"⚠️  Failed to load config file: {e}")

if __name__ == "__main__":
    print("🚀 Starting Colab Configuration Server...")
    load_colab_config()
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
