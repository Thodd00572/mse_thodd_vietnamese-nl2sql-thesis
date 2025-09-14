# Configuration settings for the thesis experiment
import os

class Config:
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    DEBUG = True
    
    # Database Configuration
    DATABASE_PATH = "data/tiki_products.db"
    
    # Model Configuration
    MODELS = {
        "phobert_base": "vinai/phobert-base",
        "vn_en_translate": "Helsinki-NLP/opus-mt-vi-en",
        "sqlcoder": "defog/sqlcoder-7b-2"
    }
    
    # Hugging Face Configuration
    HF_CACHE_DIR = "models/cache"
    
    # Export Configuration
    EXPORT_DIR = "exports"
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
