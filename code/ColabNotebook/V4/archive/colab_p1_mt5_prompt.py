"""
Google Colab Pipeline P1: mT5 Prompt-Only
Vietnamese NL2SQL using google/mt5-small with few-shot prompting.
Optimized for Google Colab L4 GPU environment.
"""

# ============================================================================
# GOOGLE COLAB SETUP
# ============================================================================

# Install required packages
import subprocess
import sys

def install_packages():
    """Install required packages for Colab environment."""
    packages = [
        "transformers>=4.30.0",
        "datasets>=2.10.0", 
        "torch>=2.0.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")

# Uncomment the line below on first run in Colab
# install_packages()

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

import torch
import json
import time
import sqlite3
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import logging
from datetime import datetime
from google.colab import drive

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check GPU availability and force CUDA if available
if torch.cuda.is_available():
    device = "cuda"
    print(f"🚀 Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"
    print(f"⚠️  Using device: {device} (GPU not available)")
    print("Note: This will be significantly slower. Enable GPU runtime in Colab.")

# ============================================================================
# VIETNAMESE NL2SQL SCHEMA AND PROMPTS
# ============================================================================

TIKI_SCHEMA_CARD = """
Database Schema - Tiki E-commerce:

Table: product_overview (main view)
Columns:
- id: product ID (integer)
- name: product name (text)
- brand: brand name (text)
- category: category name (text)  
- price: current price in VND (integer)
- original_price: original price in VND (integer)
- discount_percent: discount percentage (integer)
- rating: average rating 1-5 (real)
- review_count: number of reviews (integer)
- seller: seller name (text)
- seller_rating: seller rating 1-5 (real)
- description: product description (text)
- stock_quantity: available stock (integer)
- status: product status (text)
- created_at: creation date (text)
- updated_at: last update date (text)

Note: All queries should use 'FROM product_overview' for product data.
"""

FEW_SHOT_EXAMPLES = [
    {
        "vietnamese": "Hiển thị tất cả sản phẩm",
        "sql": "SELECT * FROM product_overview;"
    },
    {
        "vietnamese": "Tìm sản phẩm có giá dưới 500000 đồng",
        "sql": "SELECT * FROM product_overview WHERE price < 500000;"
    },
    {
        "vietnamese": "Sản phẩm nào có rating cao nhất?",
        "sql": "SELECT * FROM product_overview ORDER BY rating DESC LIMIT 1;"
    },
    {
        "vietnamese": "Đếm số sản phẩm theo từng thương hiệu",
        "sql": "SELECT brand, COUNT(*) as count FROM product_overview GROUP BY brand;"
    },
    {
        "vietnamese": "Tìm 5 sản phẩm đắt nhất trong danh mục điện thoại",
        "sql": "SELECT * FROM product_overview WHERE category LIKE '%điện thoại%' ORDER BY price DESC LIMIT 5;"
    }
]

def create_mt5_prompt(question: str, include_examples: bool = True) -> str:
    """Generate mT5 prompt for Vietnamese -> SQL translation."""
    prompt_parts = [
        "Translate Vietnamese to SQL:",
        "",
        TIKI_SCHEMA_CARD,
        ""
    ]
    
    if include_examples:
        prompt_parts.append("Examples:")
        for i, example in enumerate(FEW_SHOT_EXAMPLES[:3], 1):
            prompt_parts.append(f"Q{i}: {example['vietnamese']}")
            prompt_parts.append(f"A{i}: {example['sql']}")
            prompt_parts.append("")
    
    prompt_parts.extend([
        f"Q: {question}",
    ])
    
    return "\n".join(prompt_parts)

# ============================================================================
# GOOGLE DRIVE SETUP
# ============================================================================

def setup_google_drive():
    """Mount Google Drive and setup vn2sql project structure"""
    print(" Mounting Google Drive...")
    
    # Handle existing mount gracefully
    try:
        if os.path.exists('/content/drive/MyDrive'):
            print(" Google Drive already mounted")
        else:
            drive.mount('/content/drive')
    except ValueError as e:
        if "Mountpoint must not already contain files" in str(e):
            print(" Drive mountpoint has files, attempting to remount...")
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
        print(f" {name}: {path}")
    
    return paths

# ============================================================================
# DATABASE AND DATA UTILITIES
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
    
    # Create both views for compatibility
    cursor.execute("DROP VIEW IF EXISTS product_overview")
    cursor.execute("DROP VIEW IF EXISTS products_with_price")
    
    # Original product_overview view
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
    
    # New products_with_price view matching the evaluation data schema
    cursor.execute("""
        CREATE VIEW products_with_price AS
        SELECT 
            p.id as product_id,
            p.name,
            p.status as description,
            p.brand_id,
            p.category_id,
            p.seller_id,
            COALESCE(b.name, 'Unknown') as brand_name,
            COALESCE(c.name, 'Unknown') as category_name,
            COALESCE(pr.current_price, 0) as price,
            COALESCE(pr.original_price, 0) as original_price,
            COALESCE(pr.current_price, 0) as current_price,
            0 as quantity_sold,
            COALESCE(r.rating, 0) as rating_average,
            COALESCE(r.review_count, 0) as review_count,
            0 as favourite_count
        FROM products p
        LEFT JOIN brands b ON p.brand_id = b.id
        LEFT JOIN categories c ON p.category_id = c.id
        LEFT JOIN pricing pr ON p.id = pr.product_id
        LEFT JOIN reviews r ON p.id = r.product_id
        WHERE p.status = 'active'
    """)
    
    conn.commit()
    conn.close()
    print(f"✅ Created sample database: {db_path}")

def create_sample_data(data_dir: str, num_samples: int = 100):
    """Create sample Vietnamese NL2SQL data."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Extended sample data
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
    
    # Generate extended samples
    samples = []
    for i in range(num_samples):
        base = base_samples[i % len(base_samples)].copy()
        if i >= len(base_samples):
            base['text'] = f"{base['text']} (variant {i//len(base_samples) + 1})"
        samples.append(base)
    
    # Split data
    train_size = int(0.7 * num_samples)
    dev_size = int(0.15 * num_samples)
    
    splits = {
        'train': samples[:train_size],
        'dev': samples[train_size:train_size + dev_size],
        'eval': samples[train_size + dev_size:]
    }
    
    for split_name, split_data in splits.items():
        file_path = Path(data_dir) / f"{split_name}.jsonl"
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅ Created {file_path} with {len(split_data)} samples")

# ============================================================================
# METRICS AND EVALUATION
# ============================================================================

def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    import re
    if not sql:
        return ""
    
    # Remove extra whitespace
    sql = re.sub(r'\s+', ' ', sql.strip())
    
    # Remove comments
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    
    # Normalize quotes (both single and double to single)
    sql = re.sub(r'"([^"]*?)"', r"'\1'", sql)
    
    # Normalize table/column names to lowercase but preserve string literals
    parts = []
    in_quotes = False
    current_part = ""
    
    for char in sql:
        if char == "'" and (not current_part or current_part[-1] != '\\'):
            in_quotes = not in_quotes
        current_part += char if in_quotes else char.lower()
    
    sql = current_part
    
    # Ensure ends with semicolon
    if not sql.endswith(';'):
        sql += ';'
    
    return sql.strip()

def exact_match(pred: str, gold: str) -> int:
    """Compute exact match score."""
    return 1 if normalize_sql(pred) == normalize_sql(gold) else 0

def execution_accuracy(pred: str, gold: str, db_path: str) -> int:
    """Compute execution accuracy with better error handling."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Execute predicted SQL
        try:
            cursor.execute(pred)
            pred_result = cursor.fetchall()
            pred_columns = [desc[0] for desc in cursor.description] if cursor.description else []
        except sqlite3.Error as e:
            logger.debug(f"Predicted SQL execution failed: {e}")
            conn.close()
            return 0
        
        # Execute gold SQL
        try:
            cursor.execute(gold)
            gold_result = cursor.fetchall()
            gold_columns = [desc[0] for desc in cursor.description] if cursor.description else []
        except sqlite3.Error as e:
            logger.debug(f"Gold SQL execution failed: {e}")
            conn.close()
            return 0
        
        conn.close()
        
        # Compare results (columns and rows)
        if pred_columns != gold_columns:
            return 0
        
        # Sort results for comparison (in case order differs)
        try:
            pred_sorted = sorted(pred_result) if pred_result else []
            gold_sorted = sorted(gold_result) if gold_result else []
            return 1 if pred_sorted == gold_sorted else 0
        except TypeError:
            # If sorting fails, compare directly
            return 1 if pred_result == gold_result else 0
            
    except Exception as e:
        logger.debug(f"EX evaluation error: {e}")
        return 0

def is_valid_sql(sql: str) -> bool:
    """Check if SQL is valid with more comprehensive checks."""
    if not sql or not sql.strip():
        return False
    
    sql = sql.strip().lower()
    
    # Must start with SELECT
    if not sql.startswith('select'):
        return False
    
    # Must have FROM clause
    if 'from' not in sql:
        return False
    
    # Check for balanced parentheses
    if sql.count('(') != sql.count(')'):
        return False
    
    # Check for basic SQL structure
    if 'products_with_price' not in sql and 'products' not in sql and 'product_overview' not in sql:
        return False
    
    return True

# ============================================================================
# P1: mT5 PROMPT-ONLY PIPELINE
# ============================================================================

class MT5PromptPipeline:
    """mT5 pipeline using only prompting (no fine-tuning)."""
    
    def __init__(self, model_name: str = "google/mt5-small"):
        self.model_name = model_name
        self.device = device
        
        print(f"🔄 Loading mT5 model: {model_name}")
        
        # Load model and tokenizer
        # Use T5Tokenizer instead of MT5Tokenizer to avoid mismatch
        from transformers import T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded. Parameters: {self.model.num_parameters():,}")
        
        # GPU memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate_sql(self, vietnamese_text: str) -> str:
        """Generate SQL from Vietnamese text."""
        # Create comprehensive prompt with correct schema
        examples = [
            "Vietnamese: Hiển thị tất cả sản phẩm SQL: SELECT * FROM products_with_price;",
            "Vietnamese: Tìm sản phẩm có giá dưới 500000 SQL: SELECT * FROM products_with_price WHERE price < 500000;",
            "Vietnamese: Đếm số sản phẩm SQL: SELECT COUNT(*) FROM products_with_price;",
            "Vietnamese: Sản phẩm Apple SQL: SELECT * FROM products_with_price WHERE brand_name = 'Apple';"
        ]
        
        prompt = f"""Database: products_with_price table with columns product_id, name, brand_name, category_name, price, original_price, rating_average, review_count.
Examples:
{' '.join(examples)}
Vietnamese: {vietnamese_text} SQL:"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate with proper T5 settings
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=200,  # Use max_length instead of max_new_tokens for T5
                num_beams=4,
                no_repeat_ngram_size=3,
                do_sample=False,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process
        sql = self.postprocess_sql(generated_text)
        return sql
    
    def postprocess_sql(self, generated_text: str) -> str:
        """Post-process generated text to extract clean SQL."""
        import re
        
        sql = generated_text.strip()
        
        # Handle T5 special tokens
        if '<extra_id_' in sql:
            # Try to extract meaningful SQL from T5 output
            # This is a common issue with T5 models
            sql = re.sub(r'<extra_id_\d+>', '', sql)
            sql = sql.replace('(main view)', 'FROM product_overview')
        
        # Remove any remaining special tokens
        sql = re.sub(r'<[^>]+>', '', sql)
        
        # Clean and normalize
        sql = sql.strip()
        
        # If empty or too short, provide a basic fallback
        if len(sql) < 5 or not any(word in sql.lower() for word in ['select', 'from']):
            sql = "SELECT * FROM products_with_price"
        
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
        
        # Clean whitespace
        sql = re.sub(r'\s+', ' ', sql).strip()
        return sql
    
    def __call__(self, text: str) -> str:
        """Make pipeline callable."""
        return self.generate_sql(text)

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_pipeline(pipeline, eval_data: List[Dict], db_path: str) -> Dict:
    """Evaluate pipeline on test data."""
    print(f"🔄 Evaluating pipeline on {len(eval_data)} queries...")
    
    results = []
    latencies = []
    gpu_peak = 0.0
    
    # Reset GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    for i, item in enumerate(tqdm(eval_data, desc="Evaluating")):
        # Handle different field names for Vietnamese text
        text = item.get('text', item.get('vietnamese', item.get('vn', item.get('question', ''))))
        gold_sql = item.get('sql', item.get('gold_sql', ''))
        complexity = item.get('complexity', 'medium')
        
        # Skip if no text found
        if not text:
            print(f"⚠️  Skipping item {i}: no text field found in {list(item.keys())}")
            continue
        
        try:
            # Generate with timing
            start_time = time.time()
            pred_sql = pipeline(text)
            latency = time.time() - start_time
            latencies.append(latency)
            
            # Compute metrics
            em_score = exact_match(pred_sql, gold_sql)
            ex_score = execution_accuracy(pred_sql, gold_sql, db_path)
            is_valid = is_valid_sql(pred_sql)
            
            # Track GPU memory
            if torch.cuda.is_available():
                current_gpu = torch.cuda.max_memory_allocated() / 1e9
                gpu_peak = max(gpu_peak, current_gpu)
            
            result = {
                'query_id': i,
                'text': text,
                'gold_sql': gold_sql,
                'pred_sql': pred_sql,
                'complexity': complexity,
                'EM': em_score,
                'EX': ex_score,
                'valid': is_valid,
                'latency_ms': latency * 1000
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing query {i}: {e}")
            results.append({
                'query_id': i,
                'text': text,
                'gold_sql': gold_sql,
                'pred_sql': "",
                'complexity': complexity,
                'EM': 0,
                'EX': 0,
                'valid': False,
                'latency_ms': 0
            })
    
    # Compute aggregate metrics
    em_scores = [r['EM'] for r in results]
    ex_scores = [r['EX'] for r in results]
    valid_scores = [r['valid'] for r in results]
    
    metrics = {
        'pipeline': 'P1_mT5_prompt',
        'N': len(results),
        'EM': np.mean(em_scores),
        'EX': np.mean(ex_scores),
        'ErrorRate': 1 - np.mean(valid_scores),
        'Latency_mean': np.mean(latencies),
        'Latency_p50': np.percentile(latencies, 50),
        'Latency_p95': np.percentile(latencies, 95),
        'GPU_peak_GB': gpu_peak
    }
    
    return metrics, results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function for Colab."""
    print("🚀 Starting P1: mT5 Prompt-Only Pipeline Evaluation")
    print("=" * 60)
    
    # Setup Google Drive
    PATHS = setup_google_drive()
    
    # Setup paths
    db_path = PATHS['db'] / "tiki.sqlite"
    data_dir = PATHS['data']
    
    # Create sample database and data if they don't exist
    print("📊 Setting up test environment...")
    if not db_path.exists():
        create_sample_database(str(db_path))
    
    # Check for existing data, prioritize eval_300.jsonl
    eval_file = data_dir / "eval_300.jsonl"
    if not eval_file.exists():
        # Fallback to eval.jsonl
        eval_file = data_dir / "eval.jsonl"
        if not eval_file.exists():
            # Create sample data as last resort
            create_sample_data(str(data_dir), num_samples=50)
            eval_file = data_dir / "eval.jsonl"
    
    # Load evaluation data
    eval_data = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            eval_data.append(json.loads(line))
    
    print(f"📝 Loaded {len(eval_data)} evaluation queries")
    
    # Debug: Check data format
    if eval_data:
        print(f"🔍 Sample data fields: {list(eval_data[0].keys())}")
        print(f"🔍 Sample item: {eval_data[0]}")
    
    # Create and test pipeline
    print("\n🤖 Creating mT5 Prompt Pipeline...")
    pipeline = MT5PromptPipeline()
    
    # Quick test
    test_query = "Hiển thị tất cả sản phẩm"
    test_result = pipeline(test_query)
    print(f"🧪 Test Query: {test_query}")
    print(f"🧪 Test Result: {test_result}")
    
    # Run full evaluation
    print("\n📊 Running full evaluation...")
    metrics, detailed_results = evaluate_pipeline(pipeline, eval_data, str(db_path))
    
    # Display results
    print("\n" + "=" * 60)
    print("📈 EVALUATION RESULTS - P1: mT5 Prompt-Only")
    print("=" * 60)
    print(f"Queries Evaluated: {metrics['N']}")
    print(f"Exact Match (EM): {metrics['EM']:.3f}")
    print(f"Execution Accuracy (EX): {metrics['EX']:.3f}")
    print(f"Error Rate: {metrics['ErrorRate']:.3f}")
    print(f"Latency Mean: {metrics['Latency_mean']:.3f}s")
    print(f"Latency P50: {metrics['Latency_p50']:.3f}s")
    print(f"Latency P95: {metrics['Latency_p95']:.3f}s")
    print(f"GPU Peak Memory: {metrics['GPU_peak_GB']:.2f} GB")
    
    # Show sample predictions
    print("\n🔍 Sample Predictions:")
    for i in range(min(5, len(detailed_results))):
        result = detailed_results[i]
        print(f"\nQuery {i+1}: {result['text']}")
        print(f"Gold: {result['gold_sql']}")
        print(f"Pred: {result['pred_sql']}")
        print(f"EM: {result['EM']}, EX: {result['EX']}")
    
    # Save results with timestamp to Google Drive logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_name = "P1_mT5_prompt"
    
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
        "model_name": "google/mt5-small",
        "method": "prompt-only",
        "num_eval_queries": len(eval_data)
    }
    
    env_json_path = PATHS['logs'] / f"{pipeline_name}_{timestamp}_env.json"
    with open(env_json_path, 'w') as f:
        json.dump(env_info, f, indent=2)
    
    print(f"\n💾 Results saved to Google Drive:")
    print(f"  - {results_csv_path}")
    print(f"  - {metrics_json_path}")
    print(f"  - {env_json_path}")
    
    print("\n✅ P1 Pipeline evaluation completed!")
    return metrics, detailed_results

# Run the evaluation
if __name__ == "__main__":
    metrics, results = main()
