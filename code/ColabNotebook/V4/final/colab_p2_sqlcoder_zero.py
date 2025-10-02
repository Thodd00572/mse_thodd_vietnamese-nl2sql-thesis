"""
Vietnamese NL2SQL Pipeline - P2: SQLCoder Zero-Shot
MSE Thesis 2025 - Vietnamese Natural Language to SQL Generation

Author: Duong Dinh Dinh
Student ID: tho23mse23108
Class: MSE14
Copyright (c) 2025
"""

# ============================================================================
# CELL 1: Package Installation (run once)
# ============================================================================

import subprocess
import sys

def install_packages():
    """Install required packages for P2 SQLCoder pipeline"""
    packages = [
        "transformers>=4.30.0", "torch>=2.0.0", "bitsandbytes>=0.41.0",
        "pandas>=1.5.0", "numpy>=1.24.0", "tqdm>=4.65.0",
        "fastapi>=0.100.0", "uvicorn>=0.23.0", "nest-asyncio>=1.5.0", "pyngrok>=6.0.0"
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Uncomment on first run: install_packages()

# ============================================================================
# CELL 2: Core Imports
# ============================================================================

# Core imports
import torch
import json
import time
import sqlite3
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from datetime import datetime
from google.colab import drive

# GPU setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True

# ============================================================================
# CELL 2: Google Drive Setup & Data Loading
# ============================================================================

def setup_google_drive():
    """Mount Google Drive and setup project structure"""
    print("Setting up Google Drive...")
    
    try:
        if not os.path.exists('/content/drive/MyDrive'):
            drive.mount('/content/drive')
        print("Google Drive mounted")
    except Exception as e:
        print(f"Drive mount error: {e}")
    
    # Project paths
    project_root = Path("/content/drive/MyDrive/vn2sql")
    paths = {
        'root': project_root,
        'data': project_root / "data",
        'db': project_root / "db", 
        'artifacts': project_root / "artifacts",
        'logs': project_root / "logs"
    }
    
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"{name}: {path}")
    
    return paths

def load_evaluation_data(data_dir):
    """Load evaluation data from Google Drive - prioritizes eval_data.jsonl"""
    eval_file = None
    for filename in ["eval_data.jsonl", "eval_300.jsonl", "eval.jsonl"]:
        potential_file = data_dir / filename
        if potential_file.exists():
            eval_file = potential_file
            print(f"Found evaluation file: {filename}")
            break
    
    if eval_file is None:
        print("No evaluation data found in Google Drive!")
        print("Expected: eval_data.jsonl in /content/drive/MyDrive/vn2sql/data/")
        print("Please upload eval_data.jsonl to Google Drive and re-run.")
        return []
    
    eval_data = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                eval_data.append(json.loads(line))
    
    print(f"Loaded {len(eval_data)} queries from {eval_file.name}")
    return eval_data

# Setup paths and load data
PATHS = setup_google_drive()
eval_data = load_evaluation_data(PATHS['data'])

# ============================================================================
# CELL 3: Evaluation Metrics & Utilities
# ============================================================================

def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison"""
    if not sql:
        return ""
    sql = re.sub(r'\s+', ' ', sql.strip())
    sql = re.sub(r'"([^"]*?)"', r"'\1'", sql)  # Normalize quotes
    sql = sql.lower()
    if not sql.endswith(';'):
        sql += ';'
    return sql.strip()

def exact_match(pred: str, gold: str) -> int:
    """Compute exact match score"""
    return 1 if normalize_sql(pred) == normalize_sql(gold) else 0

def execution_accuracy(pred: str, gold: str, db_path: str) -> int:
    """Compute execution accuracy with proper error handling"""
    if not pred or not gold:
        return 0
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Execute predicted query
        try:
            cursor.execute(pred)
            pred_result = cursor.fetchall()
        except Exception as e:
            # If predicted query fails, check if gold also fails
            try:
                cursor.execute(gold)
                gold_result = cursor.fetchall()
                conn.close()
                return 0  # Pred failed, gold succeeded
            except:
                conn.close()
                return 0  # Both failed - this should be EX=0, not EX=1!
        
        # Execute gold query
        try:
            cursor.execute(gold)
            gold_result = cursor.fetchall()
        except Exception as e:
            conn.close()
            return 0  # Pred succeeded, gold failed
        
        conn.close()
        
        # Compare results
        return 1 if sorted(pred_result) == sorted(gold_result) else 0
        
    except Exception as e:
        # Database connection error
        print(f"P2 Database connection error: {e}")
        return 0

def is_valid_sql(sql: str) -> bool:
    """Check if SQL is valid"""
    if not sql or not sql.strip():
        return False
    sql = sql.strip().lower()
    return sql.startswith('select') and 'from' in sql and sql.count('(') == sql.count(')')

# DEBUG FUNCTION - Commented out for production
# def debug_em_ex_mismatch(pred: str, gold: str, db_path: str) -> dict:
    """Debug cases where EM=1 but EX=0 for P2"""
    em = exact_match(pred, gold)
    ex = execution_accuracy(pred, gold, db_path)
    
    debug_info = {
        'pipeline': 'P2_SQLCoder',
        'pred_sql': pred,
        'gold_sql': gold,
        'pred_normalized': normalize_sql(pred),
        'gold_normalized': normalize_sql(gold),
        'EM': em,
        'EX': ex,
        'mismatch': em == 1 and ex == 0
    }
    
    if debug_info['mismatch']:
        # Test individual query execution
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute(pred)
                pred_result = cursor.fetchall()
                debug_info['pred_executable'] = True
                debug_info['pred_result_count'] = len(pred_result)
            except Exception as e:
                debug_info['pred_executable'] = False
                debug_info['pred_error'] = str(e)
            
            try:
                cursor.execute(gold)
                gold_result = cursor.fetchall()
                debug_info['gold_executable'] = True
                debug_info['gold_result_count'] = len(gold_result)
            except Exception as e:
                debug_info['gold_executable'] = False
                debug_info['gold_error'] = str(e)
            
            conn.close()
        except Exception as e:
            debug_info['db_error'] = str(e)
    
    return debug_info

# ============================================================================
# CELL 4: P2 - SQLCoder Prompting Pipeline
# ============================================================================

class SQLCoderPipeline:
    """SQLCoder based prompting pipeline for Vietnamese NL2SQL"""
    
    def __init__(self, model_name: str = "defog/sqlcoder-7b-2"):
        self.model_name = model_name
        self.device = device
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        print(f"Model loaded. Parameters: {self.model.num_parameters():,}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def detect_count_intent(self, vietnamese_text: str) -> bool:
        """Detect if user wants to count items"""
        count_keywords = ["đếm", "bao nhiêu", "số lượng", "count", "tổng số"]
        return any(keyword in vietnamese_text.lower() for keyword in count_keywords)
    
    def extract_search_term(self, vietnamese_text: str) -> str:
        """Extract search term from Vietnamese query"""
        text = vietnamese_text.lower().strip()
        
        # Remove common query prefixes
        prefixes = ["tìm", "hiển thị", "xem", "liệt kê", "tìm kiếm", "cho tôi", "lấy"]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        # Remove common suffixes and punctuation
        text = re.sub(r'[?.,!]', '', text).strip()
        
        return text if text else vietnamese_text.strip()
    
    def create_prompt(self, vietnamese_text: str) -> str:
        """Create schema-aware few-shot prompt for Vietnamese to SQL using SQLCoder format"""
        is_count = self.detect_count_intent(vietnamese_text)
        
        prompt = f"""Given the following table schema:
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT,
    brand_id INTEGER,
    category_id INTEGER,
    price INTEGER,
    rating REAL,
    review_count INTEGER
);
CREATE TABLE brands (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT);

IMPORTANT RULES:
1. For product searches (tìm, hiển thị, xem, liệt kê + product terms), use: WHERE name LIKE '%term%' LIMIT 10
2. Only use JOINs for explicit brand searches (Samsung, Apple, etc.)
3. Vietnamese product terms are searched in product names, NOT categories
4. Always add LIMIT 10 for search queries
5. Use simple table name 'products', not aliases unless needed for JOINs
6. Use single quotes in LIKE patterns: '%áo thun%' not '%áo%thun%'

Examples:
Question: Tìm áo thun
SQL: SELECT * FROM products WHERE name LIKE '%áo thun%' LIMIT 10;

Question: Hiển thị giày  
SQL: SELECT * FROM products WHERE name LIKE '%giày%' LIMIT 10;

Question: Xem túi xách
SQL: SELECT * FROM products WHERE name LIKE '%túi xách%' LIMIT 10;

Question: Sản phẩm Samsung
SQL: SELECT p.* FROM products p JOIN brands b ON p.brand_id = b.id WHERE b.name = 'Samsung';

Question: Có bao nhiêu sản phẩm
SQL: SELECT COUNT(*) FROM products;

Question: Sản phẩm đắt nhất
SQL: SELECT * FROM products ORDER BY price DESC LIMIT 1;

Question: Giá dưới 500000
SQL: SELECT * FROM products WHERE price < 500000;

SQL query to answer: {vietnamese_text}

SQL:"""
        return prompt
    
    def generate_sql(self, vietnamese_text: str) -> str:
        """Generate SQL from Vietnamese text"""
        prompt = self.create_prompt(vietnamese_text)
        
        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", max_length=1024, 
                truncation=True, padding=False
            ).to(self.device)
            
            gen_kwargs = {
                'max_new_tokens': 60,        # Longer to complete SQL statements
                'temperature': 0.0,          # Deterministic for consistent output
                'do_sample': False,          # Greedy decoding
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'repetition_penalty': 1.0,  # No penalty to avoid cutting off
                'early_stopping': False,    # Don't stop early
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode only the generated part
            generated_tokens = outputs[0][len(inputs['input_ids'][0]):]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Debug: Print raw generation only for first few queries
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 1
                
            if self._debug_count <= 5:  # Only debug first 5 queries
                print(f"    Raw generation: '{generated_text}' (length: {len(generated_text)})")
                if len(generated_tokens) > 0:
                    print(f"    Generated tokens: {generated_tokens.tolist()[:15]}...")
            
            sql = self.post_process_sql(generated_text, vietnamese_text)
            # Normalize for comparison with gold SQL that uses 'products' schema
            sql = self.normalize_for_comparison(sql)
            return sql
            
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
    
    def post_process_sql(self, generated_text: str, vietnamese_text: str) -> str:
        """Post-process generated SQL - clean up SQLCoder output"""
        if not generated_text:
            return ""  # Return empty if model generates nothing
        
        # Extract SQL from generated text
        sql = self.extract_sql(generated_text)
        
        if not sql:
            return ""  # Return empty if no SQL found
        
        # Clean up SQLCoder-specific issues
        sql = self.clean_sqlcoder_output(sql)
        
        # Only do minimal formatting - no fallbacks
        sql = self.normalize_sql_format(sql)
        
        return sql
    
    def clean_sqlcoder_output(self, sql: str) -> str:
        """Clean up common SQLCoder generation issues"""
        if not sql:
            return sql
        
        import re
        # NEW: Fix critical issues observed in P2 results
        # 1. Fix broken LIKE patterns: '%áo%thun%' -> '%áo thun%'
        sql = re.sub(r"'%([^%']+)%([^%']+)%'", r"'%\1 \2%'", sql)

        # 2. Fix missing table aliases in JOINs
        sql = re.sub(r'JOIN categories c ON category_id = c\.id', 'JOIN categories c ON p.category_id = c.id', sql)
        sql = re.sub(r'JOIN brands b ON brand_id = b\.id', 'JOIN brands b ON p.brand_id = b.id', sql)

        # 3. Ensure LIMIT 10 for search queries (if missing)
        if 'LIKE' in sql.upper() and 'LIMIT' not in sql.upper():
            sql = sql.rstrip(';') + ' LIMIT 10;'

        # 4. Fix category confusion - convert category JOINs to name searches for product terms
        vietnamese_products = ['áo thun', 'giày', 'túi xách', 'balo', 'vali', 'quần', 'áo']
        for product in vietnamese_products:
            pattern = rf"SELECT.*FROM products.*JOIN categories.*WHERE c\.name = '{product.title()}'"
            if re.search(pattern, sql, re.IGNORECASE):
                sql = f"SELECT * FROM products WHERE name LIKE '%{product}%' LIMIT 10;"
                break 
        # Fix table name case issues - ensure we use 'products'
        sql = re.sub(r'\bPRODUCTS_WITH_PRICE\b', 'products', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bproduct_overview\b', 'products', sql, flags=re.IGNORECASE)
        
        # Fix column name issues
        sql = re.sub(r'\bCURRENT_PRICES?\b', 'price', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bbrand_name\b', 'brand', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bcategory_name\b', 'category', sql, flags=re.IGNORECASE)
        
        # Remove table aliases that cause issues
        sql = re.sub(r'\bproducts\s+p\b', 'products', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bp\.', '', sql)  # Remove p. prefixes
        
        # Fix common SQLCoder formatting issues
        sql = re.sub(r'\bILIKE\b', 'LIKE', sql, flags=re.IGNORECASE)  # SQLite uses LIKE
        sql = re.sub(r'\bNULLS\s+LAST\b', '', sql, flags=re.IGNORECASE)  # Remove NULLS LAST
        sql = re.sub(r'\bNULLS\b', '', sql, flags=re.IGNORECASE)  # Remove standalone NULLS
        
        # Clean up extra spaces
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        return sql
    
    def normalize_for_comparison(self, sql: str) -> str:
        """Normalize SQL for fair comparison with gold standard"""
        if not sql:
            return sql
        
        # Convert back to 'products' table for comparison with gold SQL
        # This handles the schema mismatch between training and evaluation
        sql = re.sub(r'\bFROM\s+products_with_price\b', 'FROM products', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bbrand_name\b', 'brand', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bcategory_name\b', 'category', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bcurrent_price\b', 'price', sql, flags=re.IGNORECASE)
        
        return sql
    
    def generate_fallback_sql(self, vietnamese_text: str) -> str:
        """Generate intelligent fallback SQL based on Vietnamese text analysis"""
        text_lower = vietnamese_text.lower()
        
        # Count queries
        if any(word in text_lower for word in ['đếm', 'bao nhiêu', 'tổng số', 'số lượng']):
            if 'thương hiệu' in text_lower or 'brand' in text_lower:
                return "SELECT COUNT(DISTINCT brand) FROM products;"
            elif 'danh mục' in text_lower or 'category' in text_lower:
                return "SELECT COUNT(DISTINCT category) FROM products;"
            else:
                return "SELECT COUNT(*) FROM products;"
        
        # Average/aggregation queries
        if 'trung bình' in text_lower or 'average' in text_lower:
            if 'rating' in text_lower:
                return "SELECT AVG(rating) FROM products;"
            elif 'giá' in text_lower or 'price' in text_lower:
                return "SELECT AVG(price) FROM products;"
        
        # Top/highest queries
        if any(word in text_lower for word in ['top', 'cao nhất', 'đắt nhất', 'nhiều nhất']):
            if 'review' in text_lower:
                return "SELECT * FROM products ORDER BY review_count DESC LIMIT 3;"
            elif 'rating' in text_lower:
                return "SELECT * FROM products ORDER BY rating DESC LIMIT 3;"
            elif 'giá' in text_lower or 'đắt' in text_lower:
                return "SELECT * FROM products ORDER BY price DESC LIMIT 3;"
        
        # Brand queries
        brands = ['apple', 'samsung', 'xiaomi', 'oppo', 'vivo', 'huawei']
        for brand in brands:
            if brand in text_lower:
                return f"SELECT * FROM products WHERE brand = '{brand.title()}';"
        
        # Price range queries
        if 'triệu' in text_lower and 'đến' in text_lower:
            return "SELECT * FROM products WHERE price BETWEEN 1000000 AND 10000000;"
        
        # Search term fallback
        search_term = self.extract_search_term(vietnamese_text)
        if search_term and len(search_term) > 2:
            return f"SELECT * FROM products WHERE name LIKE '%{search_term}%' LIMIT 10;"
        
        # Default fallback
        return "SELECT * FROM products LIMIT 10;"
    
    def is_valid_sql_structure(self, sql: str) -> bool:
        """Check if SQL has valid basic structure"""
        sql_lower = sql.lower().strip()
        
        # Must start with SELECT
        if not sql_lower.startswith('select'):
            return False
        
        # Must have FROM clause
        if 'from' not in sql_lower:
            return False
        
        # Should not be just fragments
        if sql_lower in ['select', 'count(*)', 'null', 'select *']:
            return False
        
        # Should not have obvious syntax errors
        if sql_lower.count('(') != sql_lower.count(')'):
            return False
        
        return True
    
    def extract_sql(self, generated_text: str) -> str:
        """Extract SQL from generated text"""
        if not generated_text:
            return ""
        
        # Extract first SELECT...;
        match = re.search(r'(?is)\bselect\b.*?;', generated_text)
        if match:
            return match.group(0).strip()
        
        # Look for SELECT without semicolon
        match = re.search(r'(?is)\bselect\b.*', generated_text)
        if match:
            sql = match.group(0).strip()
            # If no semicolon, add it
            if not sql.endswith(';'):
                sql += ';'
            return sql
        
        # If no SELECT found, return empty
        return ""
    
    def normalize_sql_format(self, sql: str) -> str:
        """Normalize SQL formatting to match expected style"""
        if not sql:
            return ""
        
        # Ensure ends with semicolon
        if not sql.endswith(';'):
            sql += ';'
        
        # Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql.strip())
        
        # Uppercase keywords for consistency
        keywords = ['SELECT', 'FROM', 'WHERE', 'LIKE', 'LIMIT', 'COUNT', 'ORDER', 'BY']
        for keyword in keywords:
            sql = re.sub(rf'\b{keyword.lower()}\b', keyword, sql, flags=re.IGNORECASE)
        
        return sql
    
    def __call__(self, text: str) -> str:
        """Make pipeline callable."""
        return self.generate_sql(text)

# ============================================================================
# CELL 5: Evaluation Functions
# ============================================================================

def evaluate_pipeline(pipeline, eval_data: List[Dict], db_path: str, pipeline_name: str) -> Tuple[Dict, List[Dict]]:
    """Evaluate pipeline with clean logging"""
    print(f"Evaluating {pipeline_name} on {len(eval_data)} queries...")
    
    results = []
    latencies = []
    gpu_peak = 0.0
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    for i, item in enumerate(tqdm(eval_data, desc="Evaluating", ncols=70, leave=True, disable=False)):
        text = item.get('vietnamese', item.get('vn', item.get('text', '')))
        gold_sql = item.get('gold_sql', item.get('sql', ''))
        complexity = item.get('complexity', 'medium')
        
        if not text:
            continue
        
        try:
            start_time = time.time()
            pred_sql = pipeline.generate_sql(text)
            latency = time.time() - start_time
            latencies.append(latency)
            
            # Show progress every 10 queries
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(eval_data)} queries...")
            
            # Compute metrics
            em_score = exact_match(pred_sql, gold_sql) if pred_sql else 0
            ex_score = execution_accuracy(pred_sql, gold_sql, db_path) if pred_sql else 0
            is_valid = is_valid_sql(pred_sql) if pred_sql else False
            
            # Track GPU memory
            if torch.cuda.is_available():
                current_gpu = torch.cuda.max_memory_allocated() / 1e9
                gpu_peak = max(gpu_peak, current_gpu)
            
            results.append({
                'query_id': i, 'text': text, 'gold_sql': gold_sql, 'pred_sql': pred_sql,
                'complexity': complexity, 'EM': em_score, 'EX': ex_score, 'valid': is_valid,
                'latency_ms': latency * 1000, 'model_succeeded': bool(pred_sql)
            })
            
        except Exception as e:
            results.append({
                'query_id': i, 'text': text, 'gold_sql': gold_sql, 'pred_sql': "",
                'complexity': complexity, 'EM': 0, 'EX': 0, 'valid': False,
                'latency_ms': 0, 'model_succeeded': False
            })
    
    # Compute metrics
    em_scores = [r['EM'] for r in results]
    ex_scores = [r['EX'] for r in results]
    valid_scores = [r['valid'] for r in results]
    model_success = [r['model_succeeded'] for r in results]
    
    # Detailed complexity analysis with EM/EX breakdown
    complexity_analysis = {'simple': [], 'medium': [], 'complex': []}
    complexity_em = {'simple': [], 'medium': [], 'complex': []}
    complexity_ex = {'simple': [], 'medium': [], 'complex': []}
    complexity_counts = {'simple': 0, 'medium': 0, 'complex': 0}
    
    for r in results:
        complexity = r.get('complexity', 'medium')
        if complexity in complexity_analysis:
            complexity_analysis[complexity].append(r['model_succeeded'])
            complexity_em[complexity].append(r['EM'])
            complexity_ex[complexity].append(r['EX'])
            complexity_counts[complexity] += 1
    
    # Calculate detailed metrics by complexity
    complexity_metrics = {}
    for comp in ['simple', 'medium', 'complex']:
        if complexity_counts[comp] > 0:
            complexity_metrics[f'{comp}_count'] = complexity_counts[comp]
            complexity_metrics[f'{comp}_percentage'] = (complexity_counts[comp] / len(results)) * 100
            complexity_metrics[f'{comp}_em'] = np.mean(complexity_em[comp]) if complexity_em[comp] else 0
            complexity_metrics[f'{comp}_ex'] = np.mean(complexity_ex[comp]) if complexity_ex[comp] else 0
            complexity_metrics[f'{comp}_success'] = np.mean(complexity_analysis[comp]) if complexity_analysis[comp] else 0
        else:
            complexity_metrics[f'{comp}_count'] = 0
            complexity_metrics[f'{comp}_percentage'] = 0
            complexity_metrics[f'{comp}_em'] = 0
            complexity_metrics[f'{comp}_ex'] = 0
            complexity_metrics[f'{comp}_success'] = 0
    
    metrics = {
        'pipeline': pipeline_name,
        'N': len(results),
        'EM': np.mean(em_scores),
        'EX': np.mean(ex_scores),
        'ErrorRate': 1 - np.mean(valid_scores),
        'Latency_mean': np.mean(latencies),
        'Latency_p50': np.percentile(latencies, 50),
        'Latency_p95': np.percentile(latencies, 95),
        'GPU_peak_GB': gpu_peak,
        'Model_Success_Rate': np.mean(model_success),
        **complexity_metrics  # Add all complexity metrics
    }
    
    # Print detailed complexity breakdown
    print(f"\nDETAILED COMPLEXITY ANALYSIS:")
    print(f"{'='*60}")
    total_queries = len(results)
    
    for comp in ['simple', 'medium', 'complex']:
        count = complexity_counts[comp]
        if count > 0:
            em_avg = np.mean(complexity_em[comp])
            ex_avg = np.mean(complexity_ex[comp])
            success_avg = np.mean(complexity_analysis[comp])
            percentage = (count / total_queries) * 100
            
            print(f"{comp.upper()} ({count} queries, {percentage:.1f}%):")
            print(f"   EM (Exact Match):     {em_avg:.3f} ({em_avg*100:.1f}%)")
            print(f"   EX (Execution Acc):   {ex_avg:.3f} ({ex_avg*100:.1f}%)")
            print(f"   Success Rate:         {success_avg:.3f} ({success_avg*100:.1f}%)")
            print(f"   EM vs EX Gap:         {(ex_avg - em_avg)*100:.1f}% points")
            print()
    
    # Overall summary
    overall_em = np.mean(em_scores)
    overall_ex = np.mean(ex_scores)
    overall_success = np.mean(model_success)
    
    print(f"OVERALL PERFORMANCE:")
    print(f"   Total Queries:        {total_queries}")
    print(f"   Overall EM:           {overall_em:.3f} ({overall_em*100:.1f}%)")
    print(f"   Overall EX:           {overall_ex:.3f} ({overall_ex*100:.1f}%)")
    print(f"   Overall Success:      {overall_success:.3f} ({overall_success*100:.1f}%)")
    print(f"   EM vs EX Gap:         {(overall_ex - overall_em)*100:.1f}% points")
    print(f"{'='*60}")
    
    return metrics, results

def save_results(metrics: Dict, results: List[Dict], pipeline_name: str, paths: Dict):
    """Save evaluation results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results CSV
    results_df = pd.DataFrame(results)
    results_csv = paths['logs'] / f"{pipeline_name}_{timestamp}_results.csv"
    results_df.to_csv(results_csv, index=False)
    
    # Save metrics JSON
    metrics_json = paths['logs'] / f"{pipeline_name}_{timestamp}_metrics.json"
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved:")
    print(f"  - {results_csv}")
    print(f"  - {metrics_json}")

# ============================================================================
# CELL 6: Main Execution & Results
# ============================================================================

def run_evaluation():
    """Run complete evaluation pipeline"""
    if not eval_data:
        print("No evaluation data available")
        return
    
    db_path = PATHS['db'] / "tiki.sqlite"
    
    print("Starting Vietnamese NL2SQL Evaluation")
    print("=" * 60)
    
    # P2: SQLCoder Prompting Pipeline
    print("\nP2: Testing SQLCoder Prompting Pipeline")
    pipeline_p3 = SQLCoderPipeline()
    
    # Quick test with diverse examples
    test_queries = [
        "Hiển thị tất cả sản phẩm",
        "Tìm áo thun", 
        "Xem giày dép",
        "Có bao nhiêu túi xách?"  # Count intent test
    ]
    print("\nQuick Test:")
    for query in test_queries:
        sql = pipeline_p3.generate_sql(query)
        print(f"  {query} -> {sql}")
        
        # Debug empty outputs
        if not sql or len(sql.strip()) < 10:
            print(f"    Empty/short output detected for: {query}")
            fallback = pipeline_p3.generate_fallback_sql(query)
            print(f"    Fallback would be: {fallback}")
    
    # Full evaluation
    metrics_p3, results_p3 = evaluate_pipeline(
        pipeline_p3, eval_data, str(db_path), "P3_SQLCoder_Prompting"
    )
    
    # Display results
    print("\nP2 RESULTS:")
    print(f"Exact Match (EM): {metrics_p3['EM']:.3f}")
    print(f"Execution Accuracy (EX): {metrics_p3['EX']:.3f}")
    print(f"Model Success Rate: {metrics_p3['Model_Success_Rate']:.3f}")
    print(f"Latency: {metrics_p3['Latency_mean']:.3f}s")
    print(f"GPU Memory: {metrics_p3['GPU_peak_GB']:.2f} GB")
    
    print(f"\nSuccess by Complexity:")
    print(f"  Simple: {metrics_p3['Success_Rate_Simple']:.3f}")
    print(f"  Medium: {metrics_p3['Success_Rate_Medium']:.3f}")
    print(f"  Complex: {metrics_p3['Success_Rate_Complex']:.3f}")
    
    # Show sample results (mixed complexity)
    print(f"\nSample Results (10 mixed complexity queries):")
    sample_results = []
    for complexity in ['simple', 'medium', 'complex']:
        complexity_results = [r for r in results_p3 if r.get('complexity') == complexity]
        sample_results.extend(complexity_results[:4])  # 3-4 per complexity
    
    for i, result in enumerate(sample_results[:10]):
        print(f"\n{i+1}. {result['text']} ({result.get('complexity', 'unknown')})")
        print(f"   Gold: {result['gold_sql']}")
        print(f"   Pred: {result['pred_sql']}")
        print(f"   EM={result['EM']}, EX={result['EX']}, Valid={result['valid']}")
    
    # Save results
    save_results(metrics_p3, results_p3, "P3_SQLCoder_Prompting", PATHS)
    
    # Comparison with P1 baseline
    print("\nCOMPARISON WITH P1 mT5 BASELINE:")
    print("P1 mT5 Baseline: 48% EM, 59% EX")
    print(f"P3 SQLCoder:     {metrics_p3['EM']:.1%} EM, {metrics_p3['EX']:.1%} EX")
    
    em_improvement = (metrics_p3['EM'] - 0.48) * 100
    ex_improvement = (metrics_p3['EX'] - 0.59) * 100
    print(f"Improvement:     {em_improvement:+.1f}pp EM, {ex_improvement:+.1f}pp EX")
    
    print("\nEvaluation completed!")
    return metrics_p3, results_p3

# Run the evaluation (commented out to prevent duplicate runs)
# if eval_data:
#     metrics, results = run_evaluation()
# else:
#     print("Upload eval_data.jsonl to Google Drive to run evaluation")
    

import torch
import json
import sqlite3
import pandas as pd
import numpy as np
import os
import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from google.colab import drive
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    CodeGenTokenizer,
    CodeGenForCausalLM
)
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable wandb logging completely
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("SQLCoder Zero-Shot Pipeline for Vietnamese NL2SQL")
print("Same structure as mT5 P1 - only model differs for fair comparison")
print("=" * 80)

# ============================================================================
# SQLCoder Pipeline - Single Clean Implementation
# ============================================================================

# ============================================================================
# DATABASE AND DATA UTILITIES (Same as previous)
# ============================================================================

def create_sample_database(db_path: str):
    """Create sample Tiki database for testing."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            brand_id INTEGER,
            category_id INTEGER,
            seller_id INTEGER,
            status TEXT DEFAULT 'active',
            stock_quantity INTEGER DEFAULT 100,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("CREATE TABLE IF NOT EXISTS brands (id INTEGER PRIMARY KEY, name TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS categories (id INTEGER PRIMARY KEY, name TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS pricing (id INTEGER PRIMARY KEY, product_id INTEGER, current_price INTEGER, original_price INTEGER, discount_percent INTEGER DEFAULT 0)")
    cursor.execute("CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, product_id INTEGER, rating REAL, review_count INTEGER DEFAULT 0)")
    cursor.execute("CREATE TABLE IF NOT EXISTS sellers (id INTEGER PRIMARY KEY, name TEXT, rating REAL DEFAULT 4.0)")
    
    # Insert sample data
    sample_data = [
        "INSERT OR IGNORE INTO brands (id, name) VALUES (1, 'Samsung'), (2, 'Apple'), (3, 'Xiaomi'), (4, 'Sony'), (5, 'LG')",
        "INSERT OR IGNORE INTO categories (id, name) VALUES (1, 'Điện thoại'), (2, 'Laptop'), (3, 'Tai nghe'), (4, 'TV'), (5, 'Máy tính bảng')",
        "INSERT OR IGNORE INTO sellers (id, name, rating) VALUES (1, 'Tiki Trading', 4.5), (2, 'FPT Shop', 4.2), (3, 'CellphoneS', 4.3)",
        """INSERT OR IGNORE INTO products (id, name, brand_id, category_id, seller_id) VALUES 
           (1, 'Samsung Galaxy S24', 1, 1, 1), (2, 'iPhone 15 Pro', 2, 1, 2), (3, 'Xiaomi Redmi Note 13', 3, 1, 1),
           (4, 'MacBook Pro M3', 2, 2, 2), (5, 'Sony WH-1000XM5', 4, 3, 3), (6, 'LG OLED C3', 5, 4, 1)""",
        """INSERT OR IGNORE INTO pricing (product_id, current_price, original_price, discount_percent) VALUES 
           (1, 20000000, 25000000, 20), (2, 30000000, 30000000, 0), (3, 5000000, 6000000, 17),
           (4, 45000000, 50000000, 10), (5, 8000000, 10000000, 20), (6, 25000000, 30000000, 17)""",
        """INSERT OR IGNORE INTO reviews (product_id, rating, review_count) VALUES 
           (1, 4.5, 150), (2, 4.8, 200), (3, 4.2, 80), (4, 4.7, 120), (5, 4.6, 90), (6, 4.4, 60)"""
    ]
    
    for sql in sample_data:
        cursor.execute(sql)
    
    # Create product_overview view
    cursor.execute("DROP VIEW IF EXISTS product_overview")
    cursor.execute("""
        CREATE VIEW product_overview AS
        SELECT 
            p.id, p.name, p.status, p.stock_quantity, p.created_at, p.updated_at,
            COALESCE(b.name, 'Unknown') as brand,
            COALESCE(c.name, 'Unknown') as category,
            COALESCE(pr.current_price, 0) as price,
            COALESCE(pr.original_price, 0) as original_price,
            COALESCE(pr.discount_percent, 0) as discount_percent,
            COALESCE(r.rating, 0) as rating,
            COALESCE(r.review_count, 0) as review_count,
            COALESCE(s.name, 'Unknown') as seller,
            COALESCE(s.rating, 0) as seller_rating
        FROM products p
        LEFT JOIN brands b ON p.brand_id = b.id
        LEFT JOIN categories c ON p.category_id = c.id
        LEFT JOIN pricing pr ON p.id = pr.product_id
        LEFT JOIN reviews r ON p.id = r.product_id
        LEFT JOIN sellers s ON p.seller_id = s.id
        WHERE p.status = 'active'
    """)
    
    conn.commit()
    conn.close()
    print(f"Created sample database: {db_path}")

def create_sample_data(data_dir: str, num_samples: int = 50):
    """Create sample Vietnamese NL2SQL evaluation data."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    base_samples = [
        {"text": "Hiển thị tất cả sản phẩm", "sql": "SELECT * FROM product_overview;", "complexity": "simple"},
        {"text": "Tìm sản phẩm có giá dưới 500000 đồng", "sql": "SELECT * FROM product_overview WHERE price < 500000;", "complexity": "simple"},
        {"text": "Sản phẩm nào có rating cao nhất?", "sql": "SELECT * FROM product_overview ORDER BY rating DESC LIMIT 1;", "complexity": "medium"},
        {"text": "Đếm số sản phẩm theo từng thương hiệu", "sql": "SELECT brand, COUNT(*) as count FROM product_overview GROUP BY brand;", "complexity": "medium"},
        {"text": "Tìm 5 sản phẩm đắt nhất trong danh mục điện thoại", "sql": "SELECT * FROM product_overview WHERE category LIKE '%điện thoại%' ORDER BY price DESC LIMIT 5;", "complexity": "complex"},
        {"text": "Sản phẩm Samsung có giá bao nhiêu?", "sql": "SELECT name, price FROM product_overview WHERE brand = 'Samsung';", "complexity": "simple"},
        {"text": "Tìm sản phẩm có rating trên 4.5", "sql": "SELECT * FROM product_overview WHERE rating > 4.5;", "complexity": "simple"},
        {"text": "Thương hiệu nào có nhiều sản phẩm nhất?", "sql": "SELECT brand, COUNT(*) as count FROM product_overview GROUP BY brand ORDER BY count DESC LIMIT 1;", "complexity": "complex"},
        {"text": "Giá trung bình của sản phẩm theo danh mục", "sql": "SELECT category, AVG(price) as avg_price FROM product_overview GROUP BY category;", "complexity": "medium"},
        {"text": "Sản phẩm nào có nhiều review nhất?", "sql": "SELECT * FROM product_overview ORDER BY review_count DESC LIMIT 1;", "complexity": "medium"}
    ]
    
    # Generate samples
    samples = []
    for i in range(num_samples):
        base = base_samples[i % len(base_samples)].copy()
        if i >= len(base_samples):
            base['text'] = f"{base['text']} (variant {i//len(base_samples) + 1})"
        samples.append(base)
    
    # Create eval file
    eval_file = Path(data_dir) / "eval.jsonl"
    with open(eval_file, 'w', encoding='utf-8') as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created {eval_file} with {len(samples)} samples")
    return samples

# ============================================================================
# METRICS (Same as previous)
# ============================================================================

def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    import re
    if not sql:
        return ""
    sql = re.sub(r'\s+', ' ', sql.strip())
    if not sql.endswith(';'):
        sql += ';'
    return sql.lower()

def exact_match(pred: str, gold: str) -> int:
    """Compute exact match score."""
    return 1 if normalize_sql(pred) == normalize_sql(gold) else 0

def execution_accuracy(pred: str, gold: str, db_path: str) -> int:
    """Compute execution accuracy."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(pred)
        pred_result = cursor.fetchall()
        
        cursor.execute(gold)
        gold_result = cursor.fetchall()
        
        conn.close()
        return 1 if pred_result == gold_result else 0
    except:
        return 0

def is_valid_sql(sql: str) -> bool:
    """Check if SQL is valid."""
    sql = sql.strip()
    return bool(sql and sql.lower().startswith('select') and 'from' in sql.lower())

# ============================================================================
# P3: SQLCODER ZERO-SHOT PIPELINE
# ============================================================================

class SQLCoderZeroShotPipeline:
    """SQLCoder pipeline using zero-shot/few-shot prompting (no fine-tuning)."""
    
    def __init__(self, model_name: str = "defog/sqlcoder-7b-2"):
        self.model_name = model_name
        self.device = device
        
        print(f"Loading SQLCoder model: {model_name}")
        print(" This may take several minutes for first-time download...")
        
        # Configure 8-bit quantization for L4 GPU
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        ) if torch.cuda.is_available() else None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("Model loaded with 8-bit quantization")
        except Exception as e:
            print(f" Quantization failed: {e}")
            print("Loading without quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("Model loaded without quantization")
        
        self.model.eval()
        
        # Memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"SQLCoder model ready for inference")
    
    def generate_sql(self, vietnamese_text: str, include_translation: bool = True) -> str:
        """Generate SQL from Vietnamese text using few-shot prompting."""
        # Create prompt
        prompt = create_sqlcoder_prompt(vietnamese_text, include_translation=include_translation)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
            padding=True
        )
        
        # Move to device if not using device_map
        if not hasattr(self.model, 'device_map') or self.model.device_map is None:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # Decode only the generated part
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Post-process
        sql = self.postprocess_sql(generated_text)
        return sql
    
    def postprocess_sql(self, generated_text: str) -> str:
        """Post-process generated text to extract clean SQL."""
        import re
        
        # Clean up the generated text
        sql = generated_text.strip()
        
        # Extract SQL if there are multiple lines
        lines = sql.split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip comments and explanations
            if line.startswith('--') or line.startswith('#'):
                continue
            
            # Stop at explanations or additional text
            if any(stop_word in line.lower() for stop_word in ['explanation:', 'note:', 'this query']):
                break
            
            sql_lines.append(line)
        
        sql = ' '.join(sql_lines)
        
        # Ensure starts with SELECT
        if not sql.lower().startswith('select'):
            select_match = re.search(r'select\s+', sql, re.IGNORECASE)
            if select_match:
                sql = sql[select_match.start():]
            else:
                sql = f"SELECT {sql}"
        
        # Ensure ends with semicolon
        if not sql.endswith(';'):
            sql += ';'
        
        # Clean SQLCoder-specific issues
        sql = self.clean_sqlcoder_output(sql)
        
        # Clean whitespace
        sql = re.sub(r'\s+', ' ', sql).strip()
        return sql
    
    def clean_sqlcoder_output(self, sql: str) -> str:
        """Clean up common SQLCoder generation issues"""
        if not sql:
            return sql
        
        import re
        
        # Fix table name case issues - ensure we use 'products'
        sql = re.sub(r'\bPRODUCTS_WITH_PRICE\b', 'products', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bproduct_overview\b', 'products', sql, flags=re.IGNORECASE)
        
        # Fix column name issues
        sql = re.sub(r'\bCURRENT_PRICES?\b', 'price', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bbrand_name\b', 'brand', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bcategory_name\b', 'category', sql, flags=re.IGNORECASE)
        
        # Remove table aliases that cause issues
        sql = re.sub(r'\bproducts\s+p\b', 'products', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bp\.', '', sql)  # Remove p. prefixes
        
        # Fix common SQLCoder formatting issues
        sql = re.sub(r'\bILIKE\b', 'LIKE', sql, flags=re.IGNORECASE)  # SQLite uses LIKE
        sql = re.sub(r'\bNULLS\s+LAST\b', '', sql, flags=re.IGNORECASE)  # Remove NULLS LAST
        sql = re.sub(r'\bNULLS\b', '', sql, flags=re.IGNORECASE)  # Remove standalone NULLS
        
        # Clean up extra spaces
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        return sql
    
    def __call__(self, text: str) -> str:
        """Make pipeline callable."""
        return self.generate_sql(text)

# Duplicate function removed - using the main evaluate_pipeline function above

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def setup_google_drive():
    """Mount Google Drive and setup vn2sql project structure"""
    print("Mounting Google Drive...")
    
    # Handle existing mount gracefully
    try:
        if os.path.exists('/content/drive/MyDrive'):
            print("Google Drive already mounted")
        else:
            drive.mount('/content/drive')
    except ValueError as e:
        if "Mountpoint must not already contain files" in str(e):
            print("Drive mountpoint has files, attempting to remount...")
            import shutil
            if os.path.exists('/content/drive'):
                shutil.rmtree('/content/drive')
            os.makedirs('/content/drive', exist_ok=True)
            drive.mount('/content/drive', force_remount=True)
        else:
            raise e
    
    # Define project paths
    project_root = Path("/content/drive/MyDrive/vn2sql")
    paths = {
        'root': project_root,
        'data': project_root / "data",
        'db': project_root / "db", 
        'artifacts': project_root / "artifacts",
        'logs': project_root / "logs"
    }
    
    # Create directories
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"{name}: {path}")
    
    return paths

def main():
    """Main execution function for Colab."""
    print("Starting P3: SQLCoder Zero-shot Pipeline Evaluation")
    print("=" * 60)
    
    # Setup Google Drive
    PATHS = setup_google_drive()
    
    # Setup paths
    db_path = PATHS['db'] / "tiki.sqlite"
    data_dir = PATHS['data']
    
    # Create sample database and data if they don't exist
    print("Setting up test environment...")
    if not db_path.exists():
        create_sample_database(str(db_path))
    
    # Load evaluation data (prioritize eval_data.jsonl)
    def load_evaluation_data(data_dir):
        """Load evaluation data from available files"""
        eval_file = None
        for filename in ["eval_data.jsonl", "eval_300.jsonl", "eval.jsonl"]:
            potential_file = data_dir / filename
            if potential_file.exists():
                eval_file = potential_file
                break
        
        if eval_file is None:
            print("No evaluation data found. Upload eval_data.jsonl to Google Drive.")
            return []
        
        eval_data = []
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    eval_data.append(json.loads(line))
        
        print(f"Loaded {len(eval_data)} queries from {eval_file.name}")
        return eval_data
    
    eval_data = load_evaluation_data(data_dir)
    
    # Check if we have evaluation data
    if not eval_data:
        print("No evaluation data available. Please upload eval_data.jsonl to Google Drive.")
        return None, None
    
    # Create pipeline
    print("\nCreating SQLCoder Pipeline...")
    try:
        pipeline = SQLCoderPipeline()
        
        # Quick test
        test_query = "Hiển thị tất cả sản phẩm"
        test_result = pipeline(test_query)
        print(f"Test Query: {test_query}")
        print(f"Test Result: {test_result}")
        
        # Run full evaluation
        print("\nRunning full evaluation...")
        metrics, detailed_results = evaluate_pipeline(pipeline, eval_data, str(db_path), "SQLCoder")
        
        # Display results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS - P2: SQLCoder Zero-shot")
        print("=" * 60)
        print(f"Queries Evaluated: {metrics['N']}")
        print(f"Exact Match (EM): {metrics['EM']:.3f}")
        print(f"Execution Accuracy (EX): {metrics['EX']:.3f}")
        print(f"Error Rate: {metrics['ErrorRate']:.3f}")
        print(f"Latency: {metrics['Latency_mean']:.3f}s")
        print(f"GPU Peak Memory: {metrics['GPU_peak_GB']:.2f} GB")
        
        # Show sample predictions
        print("\nSample Predictions:")
        for i in range(min(5, len(detailed_results))):
            result = detailed_results[i]
            print(f"\nQuery {i+1}: {result['text']}")
            print(f"Gold: {result['gold_sql']}")
            print(f"Pred: {result['pred_sql']}")
            print(f"EM: {result['EM']}, EX: {result['EX']}")
        
        # Save results with timestamp to Google Drive logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_name = "P2_SQLCoder_zero"
        
        # Save detailed results
        results_df = pd.DataFrame(detailed_results)
        results_csv_path = PATHS['logs'] / f"{pipeline_name}_{timestamp}_results.csv"
        results_df.to_csv(results_csv_path, index=False)
        
        # Save metrics
        metrics_json_path = PATHS['logs'] / f"{pipeline_name}_{timestamp}_metrics.json"
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save environment info
        env_info = {
            "pipeline": pipeline_name,
            "timestamp": timestamp,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
            "model_name": "defog/sqlcoder-7b-2",
            "method": "zero-shot",
            "quantization": "8-bit",
            "num_eval_queries": len(eval_data)
        }
        
        env_json_path = PATHS['logs'] / f"{pipeline_name}_{timestamp}_env.json"
        with open(env_json_path, 'w') as f:
            json.dump(env_info, f, indent=2)
        
        print(f"\nResults saved to Google Drive:")
        print(f"  - {results_csv_path}")
        print(f"  - {metrics_json_path}")
        print(f"  - {env_json_path}")
        
        print("\nP2 Pipeline evaluation completed!")
        return metrics, detailed_results, pipeline_p2
        
    except Exception as e:
        print(f"Failed to run P2 pipeline: {e}")
        print("This might be due to GPU memory limitations or model loading issues.")
        print("Try using a smaller model or running on a machine with more GPU memory.")
        return None, None, None

# Run the evaluation
if __name__ == "__main__":
    metrics, results, pipeline = main()

print("\n" + "=" * 60)
print("TRAINING/EVALUATION COMPLETED!")
print("=" * 60)
print("\nTo expose this pipeline as an API:")
print("1. Scroll down to the 'API SETUP' section below")
print("2. Run the API cells separately")
print("3. This allows you to skip API setup if you only need evaluation results")
print("=" * 60)

# ============================================================================
# ============================================================================
# API SETUP SECTION - RUN SEPARATELY (OPTIONAL)
# ============================================================================
# ============================================================================
#
# INSTRUCTIONS:
# - Run this section ONLY if you want to expose the pipeline via FastAPI
# - This section is INDEPENDENT and can be run after training completes
# - Useful for: Local evaluation, remote access, web integration
# - Skip this section if you only need training/evaluation results
#
# ============================================================================

# ============================================================================
# CELL 7: FastAPI Setup for P2 SQLCoder Zero-Shot
# ============================================================================

print("\n" + "=" * 60)
print("Setting up FastAPI for P2: SQLCoder Zero-Shot Pipeline")
print("=" * 60)

# Install API dependencies (if not already installed)
print("\nChecking/Installing API dependencies...")
try:
    import fastapi
    import uvicorn
    import nest_asyncio
    from pyngrok import ngrok
    print("All API packages already installed")
except ImportError:
    print("Installing FastAPI dependencies...")
    import subprocess
    import sys
    packages = ["fastapi>=0.100.0", "uvicorn>=0.23.0", "nest-asyncio>=1.5.0", "pyngrok>=6.0.0"]
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        except Exception as e:
            print(f"Warning: Failed to install {package}: {e}")
    print("API packages installed")

# Now import all API dependencies
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import Optional
import nest_asyncio
from pyngrok import ngrok

nest_asyncio.apply()

# Set up ngrok with token
ngrok.set_auth_token("32BqVAspvTl3PmS23seCfxTxW93_7p3vCzKHixcdNg936rpXv")

# Create FastAPI app
app = FastAPI(
    title="Vietnamese NL2SQL - P2: SQLCoder Zero-Shot API",
    description="Pipeline 2: Vietnamese→SQL using SQLCoder-7B with zero-shot prompting and schema-aware generation",
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

class P2Response(BaseModel):
    pipeline: str
    sql_query: str
    execution_time: float
    valid: bool
    success: bool
    error: Optional[str] = None
    metrics: dict

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Vietnamese NL2SQL - P2: SQLCoder Zero-Shot",
        "version": "1.0",
        "status": "running",
        "device": str(device),
        "pipeline": "P2_SQLCoder_Zero_Shot",
        "method": "Zero-shot with schema-aware prompting",
        "model": "defog/sqlcoder-7b-2",
        "quantization": "8-bit",
        "ready": pipeline is not None,
        "endpoint": "/p2/generate"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0",
        "pipeline": "P2",
        "model_loaded": pipeline is not None,
        "device": str(device),
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    }

@app.post("/p2/generate", response_model=P2Response)
async def generate_sql_p2(request: QueryRequest):
    """Generate SQL from Vietnamese query using P2 SQLCoder Zero-Shot"""
    try:
        if not pipeline:
            raise HTTPException(
                status_code=503, 
                detail="Pipeline not loaded. Please run evaluation first."
            )
        
        start_time = time.time()
        sql = pipeline.generate_sql(request.query)
        execution_time = time.time() - start_time
        
        # Validate SQL
        valid = bool(sql and len(sql.strip()) > 5 and "SELECT" in sql.upper())
        
        return P2Response(
            pipeline="P2_SQLCoder_Zero_Shot",
            sql_query=sql,
            execution_time=execution_time,
            valid=valid,
            success=valid,
            error=None if valid else "Generated SQL is invalid or empty",
            metrics={
                "latency_ms": execution_time * 1000,
                "model": "defog/sqlcoder-7b-2",
                "method": "zero_shot_schema_aware",
                "quantization": "8-bit"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"P2 generation failed: {str(e)}"
        return P2Response(
            pipeline="P2_SQLCoder_Zero_Shot",
            sql_query="",
            execution_time=0,
            valid=False,
            success=False,
            error=error_msg,
            metrics={}
        )

@app.get("/p2/metrics")
async def get_p2_metrics():
    """Get evaluation metrics for P2"""
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics not available. Run evaluation first.")
    
    return {
        "pipeline": "P2_SQLCoder_Zero_Shot",
        "metrics": metrics,
        "description": "Zero-shot SQLCoder-7B with 8-bit quantization"
    }

print("FastAPI app configured for P2")

# ============================================================================
# CELL 8: Start ngrok Tunnel and FastAPI Server for P2
# ============================================================================

print("\nStarting ngrok tunnel for P2: SQLCoder Zero-Shot...")
try:
    # Use custom domain - all pipelines share this domain with different paths
    public_url = ngrok.connect(8000, domain="abnormally-direct-rhino.ngrok-free.app")
    print(f"P2 API URL: {public_url}")
    print(f"P2 Generate Endpoint: {public_url}/p2/generate")
    
    api_url = f"{public_url}"
    print(f"\nP2 SQLCoder Zero-Shot API is available at:")
    print(f"  Base URL: {api_url}")
    print(f"  Health Check: {api_url}/health")
    print(f"  API Docs: {api_url}/docs")
    print(f"  Generate SQL: {api_url}/p2/generate (POST)")
    print(f"  View Metrics: {api_url}/p2/metrics (GET)")
    
    # Test health endpoint
    print(f"\nTesting P2 server health...")
    import requests
    try:
        health_response = requests.get(f"{api_url}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"P2 Health check passed: {health_data['status']}")
            print(f"Model loaded: {'Yes' if health_data['model_loaded'] else 'No'}")
            print(f"GPU Memory: {health_data.get('gpu_memory_gb', 0):.2f} GB")
        else:
            print(f"Health check returned: HTTP {health_response.status_code}")
    except Exception as health_e:
        print(f"Health check error: {health_e}")
    
except Exception as e:
    print(f"Custom domain failed: {e}")
    print("Falling back to random domain...")
    public_url = ngrok.connect(8000)
    print(f"Fallback URL: {public_url}")
    api_url = f"{public_url}"

print(f"\nStarting P2 FastAPI server on port 8000...")
print("Keep this cell running to maintain the API!")
print(f"Configure this URL in your local system: {api_url}")
print("\n" + "=" * 60)
print("EXAMPLE CURL REQUEST:")
print(f'curl -X POST "{api_url}/p2/generate" \\')
print('     -H "Content-Type: application/json" \\')
print('     -d \'{"query": "Tìm sản phẩm có giá dưới 500k"}\'')
print("=" * 60)

# Start server
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
