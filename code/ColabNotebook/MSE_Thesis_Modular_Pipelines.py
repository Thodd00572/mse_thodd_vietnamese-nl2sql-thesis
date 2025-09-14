#!/usr/bin/env python3
"""
MINISTRY OF EDUCATION AND TRAINING
FPT UNIVERSITY

Enhancing the User Search Experience on e-commerce Platforms Using the Deep Learning-Based Approach

MODULAR Vietnamese NL2SQL System - Individual Executable Cells
Google Colab Notebook for Efficient Development and Testing

Version: 1.0
Created: 2025-01-07 14:32:15 UTC
Last Modified: 2025-01-07 14:32:15 UTC

Author: MSE14 Duong Dinh Tho
Thesis: Master of Software Engineering
© Copyright by MSE 14 Duong Dinh Tho, 2025

EXECUTION ORDER:
1. Run Cell 1 (Environment Setup) - ONCE per session
2. Run Cell 2 (Imports & Utils) - ONCE per session  
3. Run Cell 3 (Drive Mount) - ONCE per session
4. Run Cell 4 (Model Loader) - ONCE per session
5. Run Cell 5 (Pipeline Classes) - When updating pipeline logic
6. Run Cell 6A (Load Pipeline 1 Only) - When testing Pipeline 1
7. Run Cell 6B (Load Pipeline 2 Only) - When testing Pipeline 2
8. Run Cell 7 (FastAPI Setup) - When starting API server
9. Run Cell 8 (Start Server) - When ready to serve requests

MODULAR BENEFITS:
- Only run what you need
- Faster iteration during development
- Save GPU resources
- Independent testing of components
"""

# =============================================================================
# CELL 1: ENVIRONMENT SETUP (Run ONCE per session)
# =============================================================================
"""
# Cell 1: Environment Setup
Run this cell ONCE when starting a new Colab session.
"""

import subprocess
import sys

print("🔧 Installing required packages...")

# Install packages
packages = [
    "fastapi",
    "uvicorn", 
    "nest-asyncio",
    "pyngrok",
    "transformers",
    "torch",
    "datasets",
    "sentence-transformers", 
    "pandas",
    "numpy",
    "requests",
    "accelerate",
    "bitsandbytes",
    "psutil",
    "colorama",
    "sacremoses"
]

for package in packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        print(f"✅ {package}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")

print("✅ Package installation completed")

# =============================================================================
# CELL 2: IMPORTS & UTILITIES (Run ONCE per session)
# =============================================================================
"""
# Cell 2: Imports & Utilities
Run this cell ONCE to import all required libraries and set up utilities.
"""

import os
import time
import json
import logging
import psutil
import tracemalloc
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from datasets import Dataset
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Execution Metrics Class
@dataclass
class ExecutionMetrics:
    """Track execution metrics for pipeline performance"""
    start_time: float = field(default_factory=time.time)
    inference_times: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    
    def start_measurement(self):
        self.start_time = time.time()
        tracemalloc.start()
        
    def record_inference_time(self, step: str, duration: float):
        self.inference_times[step] = duration
        
    def end_measurement(self):
        total_time = (time.time() - self.start_time) * 1000
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            "execution_time_ms": total_time,
            "inference_times": self.inference_times,
            "memory_usage_mb": current / 1024**2,
            "peak_memory_mb": peak / 1024**2,
            "gpu_memory_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }

print("✅ Imports and utilities loaded")

# =============================================================================
# CELL 3: GOOGLE DRIVE MOUNT (Run ONCE per session)
# =============================================================================
"""
# Cell 3: Google Drive Mount
Run this cell ONCE to mount Google Drive for model storage.
"""

from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Set up model storage path
MODEL_STORAGE_PATH = "/content/drive/MyDrive/MSE_Thesis_Models"
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

print(f"📁 Google Drive mounted at: {MODEL_STORAGE_PATH}")

# =============================================================================
# CELL 4: MODEL LOADER CLASS (Run ONCE per session)
# =============================================================================
"""
# Cell 4: Model Loader Class
Run this cell ONCE to define the model loading infrastructure.
"""

class DriveModelManager:
    """Manage model storage and loading from Google Drive"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.model_urls = {
            'phobert_base': 'vinai/phobert-base',
            'vn_en_translator': 'Helsinki-NLP/opus-mt-vi-en', 
            'sqlcoder_model': 'defog/sqlcoder-7b-2'
        }
        
    def get_model_path(self, model_name: str) -> str:
        return os.path.join(self.base_path, model_name)
        
    def get_model_url(self, model_name: str) -> str:
        return self.model_urls.get(model_name, model_name)

class CombinedModelLoader:
    """Load and manage models for both pipelines"""
    
    def __init__(self):
        self.drive_manager = DriveModelManager(MODEL_STORAGE_PATH)
        self.device = device
        
        # Pipeline 1 models
        self.phobert_model = None
        self.phobert_tokenizer = None
        
        # Pipeline 2 models  
        self.vn_en_model = None
        self.vn_en_tokenizer = None
        self.sqlcoder_model = None
        self.sqlcoder_tokenizer = None
        
        # Vietnamese query templates based on actual Tiki database structure
        self.vietnamese_query_templates = {
            "simple": [
                "Tìm tất cả balo nữ",
                "Hiển thị các giày thể thao nam có giá dưới 500k",
                "Tìm túi xách của thương hiệu Samsonite",
                "Cho tôi xem tất cả dép tổ ong",
                "Tìm giày boots nữ có rating trên 4 sao",
                "Hiển thị kính mát có giá từ 100k đến 300k",
                "Tìm balo laptop",
                "Cho tôi xem vali vải",
                "Tìm giày sandals nam",
                "Hiển thị túi đeo chéo công sở",
                "Tìm nón lưỡi trai nam",
                "Cho tôi xem thắt lưng da nam"
            ],
            "medium": [
                "Tìm giày thể thao thương hiệu Nike có giá dưới 1 triệu",
                "Hiển thị balo nam hoặc balo nữ có rating trên 4.5 sao",
                "Tìm túi xách nữ có giá từ 200k đến 800k",
                "Cho tôi xem giày boots thương hiệu Timberland hoặc Dr.Martens",
                "Tìm kính mát có số lượt đánh giá trên 50 và rating trên 4 sao",
                "Hiển thị dép nam có giá dưới 200k sắp xếp theo giá tăng dần",
                "Tìm vali có kích thước lớn hoặc vừa thương hiệu American Tourister",
                "Cho tôi xem giày cao gót có giá từ 300k đến 1 triệu và rating trên 4 sao",
                "Tìm balo du lịch có số lượt bán trên 100 và giá dưới 600k",
                "Hiển thị túi đeo chéo thương hiệu Adidas hoặc Nike có rating từ 4 đến 5 sao"
            ],
            "complex": [
                "Tìm giày có giá cao nhất trong danh mục Giày thể thao nam",
                "Hiển thị top 10 balo bán chạy nhất có rating trên 4.5 sao",
                "Đếm số lượng túi xách của từng thương hiệu có giá dưới 500k",
                "Tìm kính mát có rating cao nhất trong khoảng giá từ 200k đến 1 triệu",
                "Hiển thị top 5 thương hiệu có nhiều giày dép nhất",
                "Tìm balo có số lượt đánh giá nhiều nhất trong danh mục Balo laptop",
                "Hiển thị giá trung bình của túi xách theo từng thương hiệu",
                "Tìm top 10 giày nữ có tỷ lệ rating/giá tốt nhất",
                "Đếm số sản phẩm theo khoảng giá: dưới 100k, 100k-500k, 500k-1 triệu, trên 1 triệu",
                "Hiển thị vali có giá gần với giá trung bình của danh mục Vali vải nhất"
            ]
        }
        
    def get_model_path(self, model_name: str) -> str:
        return os.path.join(self.drive_manager.base_path, model_name)
        
    def get_model_url(self, model_name: str) -> str:
        return self.drive_manager.model_urls.get(model_name, model_name)

    def load_pipeline1_models(self) -> bool:
        """Load Pipeline 1 models only"""
        try:
            print("Loading Pipeline 1 models...")
            
            # Load PhoBERT
            model_name = 'vinai/phobert-base'
            self.phobert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.phobert_model = AutoModel.from_pretrained(model_name).to(self.device)
            
            print("Pipeline 1 models loaded successfully")
            return True
            
        except Exception as e:
            print(f"Pipeline 1 model loading failed: {e}")
            return False
            
    def load_pipeline2_models(self) -> tuple:
        """Load Pipeline 2 models only"""
        translation_success = False
        sql_success = False
        
        try:
            print("Loading Pipeline 2 translation model...")
            
            # Load Vietnamese-English translator
            vn_en_model_name = 'Helsinki-NLP/opus-mt-vi-en'
            self.vn_en_tokenizer = AutoTokenizer.from_pretrained(vn_en_model_name)
            self.vn_en_model = AutoModelForSeq2SeqLM.from_pretrained(vn_en_model_name).to(self.device)
            
            translation_success = True
            print("Pipeline 2 translation model loaded")
            
        except Exception as e:
            print(f"Pipeline 2 translation model failed: {e}")
            
        try:
            print("Loading Pipeline 2 SQL model...")
            
            # Load SQLCoder (LLaMA-based model for SQL generation)
            sqlcoder_model_name = 'defog/sqlcoder-7b-2'
            self.sqlcoder_tokenizer = AutoTokenizer.from_pretrained(sqlcoder_model_name)
            
            # Fix padding token issue for LLaMA tokenizer
            if self.sqlcoder_tokenizer.pad_token is None:
                self.sqlcoder_tokenizer.pad_token = self.sqlcoder_tokenizer.eos_token
            
            # SQLCoder is a LLaMA-based model, use AutoModelForCausalLM
            from transformers import AutoModelForCausalLM
            self.sqlcoder_model = AutoModelForCausalLM.from_pretrained(
                sqlcoder_model_name,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            sql_success = True
            print("✅ Pipeline 2 SQL model loaded")
            
        except Exception as e:
            print(f"❌ Pipeline 2 SQL model failed: {e}")
            
        return translation_success, sql_success

# Initialize model loader
model_loader = CombinedModelLoader()
print("📦 Combined Model Loader ready")

# =============================================================================
# CELL 5: PIPELINE CLASSES (Run when updating pipeline logic)
# =============================================================================
"""
# Cell 5: Pipeline Classes
Run this cell when you modify pipeline logic or need to reload pipeline classes.
"""

class Pipeline1:
    """Pipeline 1: Vietnamese → PhoBERT-SQL → SQL"""
    
    def __init__(self):
        self.phobert_model = None
        self.phobert_tokenizer = None
        
    def load_models(self):
        """Load Pipeline 1 models"""
        success = model_loader.load_pipeline1_models()
        if success:
            self.phobert_model = model_loader.phobert_model
            self.phobert_tokenizer = model_loader.phobert_tokenizer
        return success
    
    def process(self, vietnamese_query: str, gold_sql: str = None):
        """Process Vietnamese query - FAIL FAST if models not loaded"""
        metrics = ExecutionMetrics()
        logger.info(f"[Pipeline1] Processing: {vietnamese_query}")
        metrics.start_measurement()
        
        try:
            # Validate model availability
            if not self.phobert_model or not self.phobert_tokenizer:
                raise RuntimeError("PhoBERT-SQL model not loaded. Cannot process query without trained model.")
            
            # Step 1: Tokenization
            tokenize_start = time.time()
            inputs = self.phobert_tokenizer(
                vietnamese_query, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=256
            )
            inputs = {k: v.to(model_loader.device) for k, v in inputs.items()}
            tokenize_time = (time.time() - tokenize_start) * 1000
            metrics.record_inference_time("tokenization", tokenize_time / 1000)
            logger.info(f"[Pipeline1] Tokenization completed in {tokenize_time:.1f}ms")
            
            # Step 2: PhoBERT Inference
            inference_start = time.time()
            with torch.no_grad():
                outputs = self.phobert_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            inference_time = (time.time() - inference_start) * 1000
            metrics.record_inference_time("phobert_inference", inference_time / 1000)
            logger.info(f"[Pipeline1] PhoBERT inference completed in {inference_time:.1f}ms")
            
            # Step 3: SQL Generation (placeholder - would use trained model)
            sql_gen_start = time.time()
            sql_query = self._generate_sql_placeholder(vietnamese_query)
            sql_gen_time = (time.time() - sql_gen_start) * 1000
            metrics.record_inference_time("sql_generation", sql_gen_time / 1000)
            logger.info(f"[Pipeline1] SQL generation completed in {sql_gen_time:.1f}ms")
            
            # End metrics collection
            final_metrics = metrics.end_measurement()
            
            logger.info(f"[Pipeline1] Generated SQL: {sql_query}")
            
            return {
                'sql_query': sql_query,
                'execution_time': final_metrics['execution_time_ms'] / 1000,
                'pipeline': 'Pipeline1',
                'method': 'Vietnamese → PhoBERT-SQL → SQL',
                'evaluation_metrics': final_metrics,
                'success': True,
                'version': '1.0'
            }
            
        except Exception as e:
            error_msg = f"Pipeline1 failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _generate_sql_placeholder(self, vietnamese_query: str) -> str:
        """Placeholder SQL generation - replace with trained model"""
        query_lower = vietnamese_query.lower()
        
        # Simple pattern matching for demonstration
        if any(word in query_lower for word in ['áo thun', 't-shirt']):
            return "SELECT * FROM products WHERE name LIKE '%áo thun%' LIMIT 10"
        elif any(word in query_lower for word in ['giày', 'shoes']):
            return "SELECT * FROM products WHERE name LIKE '%giày%' LIMIT 10"
        elif any(word in query_lower for word in ['túi', 'bag']):
            return "SELECT * FROM products WHERE name LIKE '%túi%' LIMIT 10"
        else:
            return "SELECT * FROM products LIMIT 10"

class Pipeline2:
    """Pipeline 2: Vietnamese → English → SQL"""
    
    def __init__(self):
        self.vn_en_model = None
        self.vn_en_tokenizer = None
        self.sqlcoder_model = None
        self.sqlcoder_tokenizer = None
        
    def load_models(self):
        """Load Pipeline 2 models"""
        translation_success, sql_success = model_loader.load_pipeline2_models()
        
        if not translation_success:
            raise RuntimeError("Pipeline2 Vietnamese-to-English translation model failed to load")
            
        if not sql_success:
            raise RuntimeError("Pipeline2 SQLCoder model failed to load")
        
        self.vn_en_model = model_loader.vn_en_model
        self.vn_en_tokenizer = model_loader.vn_en_tokenizer
        self.sqlcoder_model = model_loader.sqlcoder_model
        self.sqlcoder_tokenizer = model_loader.sqlcoder_tokenizer
        
        return True
    
    def vietnamese_to_sql_direct(self, vietnamese_text: str) -> tuple[str, float]:
        """Direct Vietnamese to SQL generation without translation step"""
        start_time = time.time()
        
        if not self.sqlcoder_model or not self.sqlcoder_tokenizer:
            error_msg = "Pipeline2 SQLCoder model not loaded"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            logger.info(f"[Pipeline2] Direct Vietnamese→SQL: {vietnamese_text}")
            
            schema = """CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    tiki_id INTEGER,
    name TEXT,
    description TEXT,
    original_price REAL,
    price REAL,
    fulfillment_type TEXT,
    brand TEXT,
    review_count INTEGER,
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
);"""

            prompt = f"""Generate SQLite query for Vietnamese question: {vietnamese_text}

### Database Schema
{schema}

### Guidelines:
- Use SQLite syntax ONLY (NOT PostgreSQL)
- Use LIKE operator (NOT ilike or ILIKE)
- Use % wildcards for text search
- Search both category and name columns
- Price is INTEGER (Vietnamese Dong)
- Vietnamese keywords: balo, túi, giày, áo, nữ, nam, thể thao, da

### Example:
Q: Find women's bags
A: SELECT * FROM products WHERE (category LIKE '%túi%' OR name LIKE '%túi%') AND (category LIKE '%nữ%' OR name LIKE '%nữ%');

Generate only the SQL query:
SELECT"""
            
            inputs = self.sqlcoder_tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(model_loader.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.sqlcoder_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.sqlcoder_tokenizer.eos_token_id,
                    eos_token_id=self.sqlcoder_tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            sql_query = self.sqlcoder_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            if not sql_query.upper().startswith('SELECT'):
                sql_query = 'SELECT ' + sql_query
            
            # Clean up SQL and fix PostgreSQL syntax
            sql_query = sql_query.split(';')[0] + ';'
            
            # Fix PostgreSQL-specific syntax for SQLite
            sql_query = sql_query.replace(' ilike ', ' LIKE ')
            sql_query = sql_query.replace(' ILIKE ', ' LIKE ')
            
            # Fix table alias issues - remove 'p.' prefix but keep column names
            import re
            sql_query = re.sub(r'\bFROM products p\b', 'FROM products', sql_query)
            sql_query = re.sub(r'\bp\.(\w+)', r'\1', sql_query)
            
            # Ensure proper column names exist in schema
            if 'PRODUCTS' in sql_query.upper():
                sql_query = sql_query.replace('PRODUCTS', 'products')
            
            execution_time = time.time() - start_time
            logger.info(f"[Pipeline2] Generated SQL: {sql_query}")
            return sql_query, execution_time
            
        except Exception as e:
            error_msg = f"Pipeline2 Vietnamese-to-English translation failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def english_to_sql(self, english_query: str):
        """Convert English to SQL"""
        start_time = time.time()
        
        if not self.sqlcoder_model or not self.sqlcoder_tokenizer:
            error_msg = "Pipeline2 SQLCoder model not loaded"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            logger.info(f"[Pipeline2] Generating SQL from: {english_query}")
            
            # Format optimized prompt for Tiki e-commerce database with CORRECT schema
            schema = """CREATE TABLE products (
    row_id INTEGER PRIMARY KEY,
    product_id INTEGER,
    name TEXT,
    description TEXT,
    original_price INTEGER,
    price INTEGER,
    fulfillment_type TEXT,
    brand TEXT,
    review_count INTEGER,
    rating_average REAL,
    favourite_count INTEGER,
    pay_later BOOLEAN,
    current_seller TEXT,
    date_created INTEGER,
    number_of_images INTEGER,
    vnd_cashback INTEGER,
    has_video BOOLEAN,
    category TEXT,
    quantity_sold INTEGER,
    source_file TEXT,
    created_at TIMESTAMP
);
CREATE TABLE categories (
    id INTEGER,
    name TEXT,
    parent_id INTEGER,
    level INTEGER
);"""

            examples = """### Query Pattern Examples:
Q: Show me bags
A: SELECT * FROM products WHERE category LIKE '%balo%' OR name LIKE '%túi%';

Q: Find products under price X
A: SELECT * FROM products WHERE price < X;

Q: Show high rated items  
A: SELECT * FROM products WHERE rating_average >= 4.0 ORDER BY rating_average DESC;"""

            prompt = f"""You are a SQL expert. Generate a SQLite query for this question: {english_query}

### Database Schema
{schema}

### Guidelines:
- Use LIKE with % wildcards for text search
- Price is INTEGER (Vietnamese Dong)
- Database contains Vietnamese text - use Vietnamese keywords in LIKE clauses
- Search both category and name columns for keywords
- Use proper SQLite syntax only

### Vietnamese Keywords in Database:
- balo (backpack), túi (bag), giày (shoes), áo (shirt/clothing)
- nữ (women's), nam (men's), trẻ em (children's)
- thể thao (sports), cao gót (high heel), da (leather)
- Brand names: Nike, Adidas, Gucci, etc.

Generate only the SQL query without explanation:
SELECT"""
            
            inputs = self.sqlcoder_tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(model_loader.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.sqlcoder_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.sqlcoder_tokenizer.eos_token_id,
                    eos_token_id=self.sqlcoder_tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            # Decode only the generated part (exclude input prompt)
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            sql_query = self.sqlcoder_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Clean up the SQL query - prepend SELECT if not present and remove extra text
            if not sql_query.upper().startswith('SELECT'):
                sql_query = 'SELECT ' + sql_query
            
            # Stop at first semicolon or newline to avoid extra text
            if ';' in sql_query:
                sql_query = sql_query.split(';')[0] + ';'
            elif '\n' in sql_query:
                sql_query = sql_query.split('\n')[0]
            
            # Remove common unwanted patterns
            sql_query = sql_query.replace('SELECT SELECT', 'SELECT')
            
            # Fix PostgreSQL syntax for SQLite compatibility
            sql_query = sql_query.replace(' ilike ', ' LIKE ')
            sql_query = sql_query.replace(' ILIKE ', ' LIKE ')
            
            # Fix hallucinated columns that don't exist in schema
            sql_query = sql_query.replace('p.gender', 'category')
            sql_query = sql_query.replace('.gender', '.category')
            sql_query = sql_query.replace('gender', 'category')
            
            # Fix incorrect column names to match actual schema
            sql_query = sql_query.replace('p.id', 'row_id')
            sql_query = sql_query.replace('p.tiki_id', 'product_id')
            sql_query = sql_query.replace('tiki_id', 'product_id')
            
            # Remove table aliases completely - they cause confusion
            import re
            sql_query = re.sub(r'\bp\.', '', sql_query)  # Remove p. aliases
            sql_query = re.sub(r'\bproducts p\b', 'products', sql_query)  # Remove alias definition
            
            execution_time = time.time() - start_time
            
            logger.info(f"[Pipeline2] Generated SQL: {sql_query}")
            return sql_query, execution_time
            
        except Exception as e:
            error_msg = f"Pipeline2 English-to-SQL generation failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def process(self, vietnamese_query: str, gold_sql: str = None) -> Dict[str, Any]:
        """Process Vietnamese query through Pipeline 2: Direct Vietnamese → SQL"""
        start_time = time.time()
        
        try:
            logger.info(f"[Pipeline2] Processing: {vietnamese_query}")
            
            # Direct Vietnamese to SQL generation (no translation step)
            sql_query, sql_gen_time = self.vietnamese_to_sql_direct(vietnamese_query)
            logger.info(f"[Pipeline2] Direct SQL generation completed in {sql_gen_time:.3f}s")
            
            total_time = time.time() - start_time
            logger.info(f"[Pipeline2] Completed successfully: {sql_query}")
            
            return {
                'sql_query': sql_query,
                'english_translation': f"Direct Vietnamese processing: {vietnamese_query}",
                'execution_time': total_time,
                'translation_time': 0.0,  # No translation step
                'sql_generation_time': sql_gen_time,
                'pipeline': 'Pipeline2',
                'method': 'Vietnamese → SQL (Direct)',
                'success': True,
                'error': None,
                'version': '2.0'
            }
            
        except Exception as e:
            error_msg = f"Pipeline2 failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

print("✅ Pipeline classes loaded")

# =============================================================================
# CELL 6A: LOAD PIPELINE 1 ONLY (Run when testing Pipeline 1)
# =============================================================================
"""
# Cell 6A: Load Pipeline 1 Only
Run this cell when you want to test Pipeline 1 specifically.
"""

# Initialize Pipeline 1
pipeline1 = Pipeline1()

print("🚀 Loading Pipeline 1 models...")

try:
    p1_success = pipeline1.load_models()
    print("✅ Pipeline 1 models loaded successfully")
    
    # Test Pipeline 1
    test_query = "Tìm áo thun"
    print(f"🧪 Testing Pipeline 1 with: '{test_query}'")
    result = pipeline1.process(test_query)
    print(f"✅ Test successful: {result['sql_query']}")
    
except Exception as e:
    print(f"❌ Pipeline 1 failed: {str(e)}")

# =============================================================================
# CELL 6B: LOAD PIPELINE 2 ONLY (Run when testing Pipeline 2)
# =============================================================================
"""
# Cell 6B: Load Pipeline 2 Only
Run this cell when you want to test Pipeline 2 specifically.
"""

# Initialize Pipeline 2
pipeline2 = Pipeline2()

print("🚀 Loading Pipeline 2 models...")

try:
    p2_success = pipeline2.load_models()
    print("✅ Pipeline 2 models loaded successfully")
    
    # Reduced test queries - 3 per category
    test_queries = [
        # Simple queries - Basic keyword matching
        "Tìm tất cả balo nữ",
        "tìm giày thể thao nam", 
        "túi xách nữ giá dưới 500k",
        
        # Medium queries - Multiple conditions
        "giày thể thao nữ màu trắng giá dưới 500k",
        "túi xách nữ da thật có đánh giá trên 4 sao",
        "balo giá từ 150k đến 600k",
        
        # Complex queries - Aggregation and sorting
        "top 10 giày nữ bán chạy nhất có giá dưới 1 triệu",
        "có bao nhiều sản phẩm của nữ có giá từ 200k đến 500k",
        "thương hiệu nào có nhiều sản phẩm nhất"
    ]
    
    print(f"🧪 Testing Pipeline 2 with {len(test_queries)} Vietnamese queries...")
    
    for i, test_query in enumerate(test_queries, 1):
        try:
            print(f"\n--- Test {i}/{len(test_queries)} ---")
            print(f"Vietnamese: {test_query}")
            
            result = pipeline2.process(test_query)
            
            print(f"English: {result['english_translation']}")
            print(f"SQL: {result['sql_query']}")
            print(f"Time: {result['execution_time']:.3f}s")
            print(f"Status: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
            
            if not result['success']:
                print(f"Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Test {i} failed: {str(e)}")
    
    print(f"\n🎉 Pipeline 2 testing completed!")
    
except Exception as e:
    print(f"❌ Pipeline 2 failed: {str(e)}")

# =============================================================================
# CELL 7: FASTAPI SETUP (Run when starting API server)
# =============================================================================
"""
# Cell 7: FastAPI Setup
Run this cell when you want to set up the API server.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import nest_asyncio
from pyngrok import ngrok
from typing import List, Optional

nest_asyncio.apply()

# Set up ngrok with your token
ngrok.set_auth_token("32BqVAspvTl3PmS23seCfxTxW93_7p3vCzKHixcdNg936rpXv")

# Create FastAPI app
app = FastAPI(
    title="Vietnamese NL2SQL Pipeline API v1.0",
    description="Modular Vietnamese NL2SQL system with individual pipeline control",
    version="1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str

class BatchQueryRequest(BaseModel):
    queries: List[str]
    batch_size: Optional[int] = 10

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
    success: bool
    error: Optional[str] = None

class SearchResponse(BaseModel):
    vietnamese_query: str
    pipeline1_result: Optional[PipelineResult]
    pipeline2_result: Optional[Pipeline2Result]
    timestamp: str
    query_id: str
    system_metrics: Optional[Dict[str, Any]] = None

class Pipeline1Response(BaseModel):
    sql_query: str
    execution_time: float
    pipeline: str
    method: str
    success: bool
    version: str
    error: Optional[str] = None

class Pipeline2Response(BaseModel):
    sql_query: str
    english_translation: str
    execution_time: float
    translation_time: float
    sql_generation_time: float
    pipeline: str
    method: str
    success: bool
    version: str
    error: Optional[str] = None

class BatchPipeline1Response(BaseModel):
    results: List[Pipeline1Response]
    batch_size: int
    total_execution_time: float
    avg_execution_time: float
    success_count: int
    error_count: int

class BatchPipeline2Response(BaseModel):
    results: List[Pipeline2Response]
    batch_size: int
    total_execution_time: float
    avg_execution_time: float
    success_count: int
    error_count: int

class ConfigRequest(BaseModel):
    colab_url: str

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Vietnamese NL2SQL Pipeline API v1.0",
        "version": "1.0",
        "status": "running",
        "device": str(device),
        "pipelines": {
            "pipeline1": {
                "loaded": 'pipeline1' in globals() and hasattr(pipeline1, 'phobert_model') and pipeline1.phobert_model is not None,
                "endpoint": "/pipeline1",
                "description": "Vietnamese → PhoBERT-SQL → SQL"
            },
            "pipeline2": {
                "loaded": 'pipeline2' in globals() and hasattr(pipeline2, 'vn_en_model') and pipeline2.vn_en_model is not None,
                "endpoint": "/pipeline2", 
                "description": "Vietnamese → English → SQL"
            }
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0",
        "device": str(device),
        "pipeline1_ready": 'pipeline1' in globals() and hasattr(pipeline1, 'phobert_model') and pipeline1.phobert_model is not None,
        "pipeline2_ready": 'pipeline2' in globals() and hasattr(pipeline2, 'vn_en_model') and pipeline2.vn_en_model is not None
    }

@app.post("/pipeline1", response_model=Pipeline1Response)
async def process_pipeline1(request: QueryRequest):
    """Pipeline 1: Vietnamese → PhoBERT-SQL → SQL"""
    try:
        if 'pipeline1' not in globals():
            raise HTTPException(status_code=503, detail="Pipeline 1 not initialized")
            
        if not hasattr(pipeline1, 'phobert_model') or pipeline1.phobert_model is None:
            raise HTTPException(status_code=503, detail="Pipeline 1 models not loaded")
        
        logger.info(f"[API] Processing Pipeline 1 query: {request.query}")
        result = pipeline1.process(request.query)
        logger.info(f"[API] Pipeline 1 completed in {result['execution_time']:.2f}s")
        
        return Pipeline1Response(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Pipeline 1 processing failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/pipeline2", response_model=Pipeline2Response)
async def process_pipeline2(request: QueryRequest):
    """Pipeline 2: Vietnamese → English → SQL"""
    try:
        if 'pipeline2' not in globals():
            raise HTTPException(status_code=503, detail="Pipeline 2 not initialized")
            
        if not hasattr(pipeline2, 'vn_en_model') or pipeline2.vn_en_model is None:
            raise HTTPException(status_code=503, detail="Pipeline 2 models not loaded")
        
        logger.info(f"[API] Processing Pipeline 2 query: {request.query}")
        result = pipeline2.process(request.query)
        logger.info(f"[API] Pipeline 2 completed in {result['execution_time']:.2f}s")
        
        return Pipeline2Response(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Pipeline 2 processing failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/batch/pipeline1", response_model=BatchPipeline1Response)
async def process_batch_pipeline1(request: BatchQueryRequest):
    """Batch Pipeline 1: Process multiple Vietnamese queries efficiently"""
    try:
        if 'pipeline1' not in globals():
            raise HTTPException(status_code=503, detail="Pipeline 1 not initialized")
            
        if not hasattr(pipeline1, 'phobert_model') or pipeline1.phobert_model is None:
            raise HTTPException(status_code=503, detail="Pipeline 1 models not loaded")
        
        batch_start_time = time.time()
        results = []
        success_count = 0
        error_count = 0
        
        logger.info(f"[BATCH-API] Processing {len(request.queries)} queries in Pipeline 1")
        
        for i, query in enumerate(request.queries):
            try:
                logger.info(f"[BATCH-API] Processing query {i+1}/{len(request.queries)}: {query}")
                result = pipeline1.process(query)
                
                response = Pipeline1Response(**result)
                results.append(response)
                
                if result['success']:
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                error_response = Pipeline1Response(
                    sql_query="",
                    execution_time=0.0,
                    pipeline="Pipeline 1",
                    method="batch_error",
                    success=False,
                    version="1.0",
                    error=str(e)
                )
                results.append(error_response)
                error_count += 1
                logger.error(f"[BATCH-API] Query {i+1} failed: {str(e)}")
        
        total_time = time.time() - batch_start_time
        avg_time = total_time / len(request.queries) if request.queries else 0
        
        logger.info(f"[BATCH-API] Pipeline 1 batch completed: {success_count} success, {error_count} errors, {total_time:.2f}s total")
        
        return BatchPipeline1Response(
            results=results,
            batch_size=len(request.queries),
            total_execution_time=total_time,
            avg_execution_time=avg_time,
            success_count=success_count,
            error_count=error_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Batch Pipeline 1 processing failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/batch/pipeline2", response_model=BatchPipeline2Response)
async def process_batch_pipeline2(request: BatchQueryRequest):
    """Batch Pipeline 2: Process multiple Vietnamese → English → SQL queries efficiently"""
    try:
        if 'pipeline2' not in globals():
            raise HTTPException(status_code=503, detail="Pipeline 2 not initialized")
            
        if not hasattr(pipeline2, 'vn_en_model') or pipeline2.vn_en_model is None:
            raise HTTPException(status_code=503, detail="Pipeline 2 models not loaded")
        
        batch_start_time = time.time()
        results = []
        success_count = 0
        error_count = 0
        
        logger.info(f"[BATCH-API] Processing {len(request.queries)} queries in Pipeline 2")
        
        for i, query in enumerate(request.queries):
            try:
                logger.info(f"[BATCH-API] Processing query {i+1}/{len(request.queries)}: {query}")
                result = pipeline2.process(query)
                
                response = Pipeline2Response(**result)
                results.append(response)
                
                if result['success']:
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                error_response = Pipeline2Response(
                    sql_query="",
                    english_translation="",
                    execution_time=0.0,
                    translation_time=0.0,
                    sql_generation_time=0.0,
                    pipeline="Pipeline 2",
                    method="batch_error",
                    success=False,
                    version="1.0",
                    error=str(e)
                )
                results.append(error_response)
                error_count += 1
                logger.error(f"[BATCH-API] Query {i+1} failed: {str(e)}")
        
        total_time = time.time() - batch_start_time
        avg_time = total_time / len(request.queries) if request.queries else 0
        
        logger.info(f"[BATCH-API] Pipeline 2 batch completed: {success_count} success, {error_count} errors, {total_time:.2f}s total")
        
        return BatchPipeline2Response(
            results=results,
            batch_size=len(request.queries),
            total_execution_time=total_time,
            avg_execution_time=avg_time,
            success_count=success_count,
            error_count=error_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Batch Pipeline 2 processing failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """Execute search using specified pipeline(s) and return results with metrics"""
    try:
        logger.info(f"Search request: {request.query} with pipeline: {request.pipeline}")
        
        # Initialize response
        response = SearchResponse(
            vietnamese_query=request.query,
            pipeline1_result=None,
            pipeline2_result=None,
            timestamp=datetime.now().isoformat(),
            query_id=f"query_{int(time.time())}"
        )
        
        if request.pipeline in ["pipeline1", "both"]:
            # Execute Pipeline 1
            try:
                if 'pipeline1' not in globals() or not hasattr(pipeline1, 'phobert_model') or pipeline1.phobert_model is None:
                    raise Exception("Pipeline 1 not initialized or models not loaded")
                
                logger.info("[Pipeline1] Starting Vietnamese NL2SQL processing...")
                result = pipeline1.process(request.query)
                
                # Log detailed metrics for Pipeline 1
                logger.info(f"[Pipeline1] SUCCESS - SQL: {result['sql_query']}")
                logger.info(f"[Pipeline1] Execution Time: {result['execution_time']:.3f}s")
                logger.info(f"[Pipeline1] Method: {result['method']}")
                if 'metrics' in result and result['metrics']:
                    logger.info(f"[Pipeline1] Detailed Metrics: {result['metrics']}")
                
                response.pipeline1_result = PipelineResult(
                    pipeline_name="Pipeline 1 (Vietnamese → PhoBERT-SQL)",
                    sql_query=result['sql_query'],
                    results=[],  # No actual SQL execution in Colab
                    execution_time=result['execution_time'],
                    success=result['success'],
                    error=result.get('error'),
                    metrics={
                        "method": result['method'],
                        "version": result['version'],
                        "pipeline": result['pipeline'],
                        "detailed_metrics": result.get('metrics', {})
                    }
                )
                
            except Exception as e:
                logger.error(f"[Pipeline1] Error: {str(e)}")
                response.pipeline1_result = PipelineResult(
                    pipeline_name="Pipeline 1 (Vietnamese → PhoBERT-SQL)",
                    sql_query="",
                    results=[],
                    execution_time=0.0,
                    success=False,
                    error=str(e),
                    metrics={}
                )
        
        if request.pipeline in ["pipeline2", "both"]:
            # Execute Pipeline 2
            try:
                if 'pipeline2' not in globals() or not hasattr(pipeline2, 'vn_en_model') or pipeline2.vn_en_model is None:
                    raise Exception("Pipeline 2 not initialized or models not loaded")
                
                logger.info("[Pipeline2] Starting Vietnamese → English → SQL processing...")
                result = pipeline2.process(request.query)
                
                # Log detailed metrics for Pipeline 2
                if result['success']:
                    logger.info(f"[Pipeline2] SUCCESS - English: {result['english_translation']}")
                    logger.info(f"[Pipeline2] SUCCESS - SQL: {result['sql_query']}")
                    logger.info(f"[Pipeline2] Translation Time: {result.get('translation_time', 0):.3f}s")
                    logger.info(f"[Pipeline2] SQL Generation Time: {result.get('sql_generation_time', 0):.3f}s")
                    logger.info(f"[Pipeline2] Total Execution Time: {result['execution_time']:.3f}s")
                    logger.info(f"[Pipeline2] Method: {result['method']}")
                else:
                    logger.error(f"[Pipeline2] FAILED - Error: {result.get('error', 'Unknown error')}")
                
                response.pipeline2_result = Pipeline2Result(
                    vietnamese_query=request.query,
                    english_query=result['english_translation'],
                    sql_query=result['sql_query'],
                    results=[],  # No actual SQL execution in Colab
                    execution_time=result['execution_time'],
                    success=result['success'],
                    error=result.get('error')
                )
                
            except Exception as e:
                logger.error(f"[Pipeline2] Error: {str(e)}")
                response.pipeline2_result = Pipeline2Result(
                    vietnamese_query=request.query,
                    english_query="",
                    sql_query="",
                    results=[],
                    execution_time=0.0,
                    success=False,
                    error=str(e)
                )
        
        return response
        
    except Exception as e:
        error_msg = f"Search processing failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/search", response_model=SearchResponse)
async def search_products_direct(request: SearchRequest):
    """Direct /search endpoint for frontend compatibility"""
    return await search_products(request)

@app.get("/config/colab/status")
async def get_colab_status():
    """Get current Colab configuration and health status"""
    try:
        pipeline1_ready = 'pipeline1' in globals() and hasattr(pipeline1, 'phobert_model') and pipeline1.phobert_model is not None
        pipeline2_ready = 'pipeline2' in globals() and hasattr(pipeline2, 'vn_en_model') and pipeline2.vn_en_model is not None
        
        return {
            "status": {
                "pipeline1_healthy": pipeline1_ready,
                "pipeline2_healthy": pipeline2_ready,
                "pipeline1_url": "https://abnormally-direct-rhino.ngrok-free.app",
                "pipeline2_url": "https://abnormally-direct-rhino.ngrok-free.app",
                "device": str(device),
                "colab_status": "running"
            }
        }
        
    except Exception as e:
        error_msg = f"Status check failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/config/colab")
async def configure_colab(request: ConfigRequest):
    """Configure Colab URL for frontend integration"""
    try:
        logger.info(f"Configuring Colab URL: {request.colab_url}")
        
        # Store the Colab URL for potential future use
        # In this case, we're already running in Colab, so this is mainly for frontend compatibility
        
        return {
            "status": "success",
            "message": "Colab URL configured successfully",
            "colab_url": request.colab_url,
            "current_status": "running_in_colab",
            "pipelines_ready": {
                "pipeline1": 'pipeline1' in globals() and hasattr(pipeline1, 'phobert_model') and pipeline1.phobert_model is not None,
                "pipeline2": 'pipeline2' in globals() and hasattr(pipeline2, 'vn_en_model') and pipeline2.vn_en_model is not None
            }
        }
        
    except Exception as e:
        error_msg = f"Colab configuration failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

print("✅ FastAPI setup complete")

# =============================================================================
# CELL 8: START SERVER (Run when ready to serve requests)
# =============================================================================
"""
# Cell 8: Start Server
Run this cell when you're ready to start serving API requests.
"""

# Start ngrok tunnel with custom domain
print("🚇 Starting ngrok tunnel for Vietnamese NL2SQL API...")
try:
    public_url = ngrok.connect(8000, domain="abnormally-direct-rhino.ngrok-free.app")
    print(f"🌍 API URL: {public_url}")
    print(f"📡 Pipeline 1 API: {public_url}/pipeline1")
    print(f"📡 Pipeline 2 API: {public_url}/pipeline2")
    
    api_url = f"{public_url}"
    print(f"\n📋 Your Vietnamese NL2SQL API is available at:")
    print(f"Base URL: {api_url}")
    print(f"Health Check: {api_url}/health")
    print(f"API Docs: {api_url}/docs")
    print(f"Pipeline 1: {api_url}/pipeline1")
    print(f"Pipeline 2: {api_url}/pipeline2")
    
except Exception as e:
    print(f"❌ Custom domain failed: {e}")
    print("🔄 Falling back to random domain...")
    public_url = ngrok.connect(8000)
    print(f"🌍 Fallback URL: {public_url}")
    api_url = f"{public_url}"

print(f"\n🚀 Starting Vietnamese NL2SQL API server on port 8000...")
print("⚠️  Keep this cell running to maintain the API!")
print(f"🔗 Configure this URL in your local frontend: {api_url}")

# Run the server
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
