"""
Google Colab Pipeline P1: mT5 Prompt-Only (IMPROVED VERSION)
Vietnamese NL2SQL using google/mt5-small with few-shot prompting.
Optimized for Google Colab L4 GPU environment.

IMPROVEMENTS:
- Uses eval_300.jsonl for comprehensive evaluation
- Better prompt engineering for mT5
- Improved SQL generation and post-processing
- Enhanced EM/EX measurement
- GPU optimization and error handling
"""

# ============================================================================
# GOOGLE COLAB SETUP
# ============================================================================

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
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from transformers import T5ForConditionalGeneration, T5Tokenizer
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
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
else:
    device = "cpu"
    print(f"⚠️  Using device: {device} (GPU not available)")
    print("Note: This will be significantly slower. Enable GPU runtime in Colab.")

# ============================================================================
# GOOGLE DRIVE SETUP
# ============================================================================

def setup_google_drive():
    """Mount Google Drive and setup vn2sql project structure"""
    print("📁 Mounting Google Drive...")
    
    # Handle existing mount gracefully
    try:
        if os.path.exists('/content/drive/MyDrive'):
            print("✅ Google Drive already mounted")
        else:
            drive.mount('/content/drive')
    except ValueError as e:
        if "Mountpoint must not already contain files" in str(e):
            print("🔄 Drive mountpoint has files, attempting to remount...")
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
        print(f"📂 {name}: {path}")
    
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

# ============================================================================
# IMPROVED METRICS AND EVALUATION
# ============================================================================

def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison with comprehensive cleaning."""
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
    """Compute execution accuracy with comprehensive error handling."""
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
    """Check if SQL is valid with comprehensive validation."""
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
# P1: IMPROVED mT5 PROMPT-ONLY PIPELINE
# ============================================================================

class ImprovedMT5PromptPipeline:
    """Improved mT5 pipeline using optimized prompting."""
    
    def __init__(self, model_name: str = "google/mt5-small"):
        self.model_name = model_name
        self.device = device
        
        print(f"🔄 Loading mT5 model: {model_name}")
        
        # Load model and tokenizer (use T5 components for mT5)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded. Parameters: {self.model.num_parameters():,}")
        
        # GPU memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def create_optimized_prompt(self, vietnamese_text: str) -> str:
        """Create optimized prompt for mT5 with better structure."""
        
        # Schema information
        schema = "products_with_price(product_id, name, brand_name, category_name, price, rating_average, review_count)"
        
        # Few-shot examples optimized for Vietnamese with correct schema
        examples = [
            "Q: Hiển thị tất cả sản phẩm A: SELECT * FROM products_with_price;",
            "Q: Tìm sản phẩm Samsung A: SELECT * FROM products_with_price WHERE brand_name = 'Samsung';",
            "Q: Đếm số sản phẩm A: SELECT COUNT(*) FROM products_with_price;",
            "Q: Sản phẩm có giá cao nhất A: SELECT * FROM products_with_price ORDER BY price DESC LIMIT 1;"
        ]
        
        # Construct prompt
        prompt = f"""Schema: {schema}
Examples: {' '.join(examples)}
Q: {vietnamese_text} A:"""
        
        return prompt
    
    def generate_sql(self, vietnamese_text: str) -> str:
        """Generate SQL from Vietnamese text with improved processing."""
        
        # Create optimized prompt
        prompt = self.create_optimized_prompt(vietnamese_text)
        
        # Tokenize with proper settings
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate with optimized parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 100,  # Relative to input length
                num_beams=5,  # Increased for better quality
                no_repeat_ngram_size=3,
                repetition_penalty=1.1,
                length_penalty=0.9,
                do_sample=False,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Post-process to clean SQL
        sql = self.postprocess_sql(generated_text)
        return sql
    
    def postprocess_sql(self, generated_text: str) -> str:
        """Advanced post-processing for better SQL extraction."""
        
        sql = generated_text.strip()
        
        # Handle T5 artifacts and clean up
        sql = re.sub(r'<[^>]*>', '', sql)  # Remove any XML-like tags
        sql = re.sub(r'\s+', ' ', sql)     # Normalize whitespace
        
        # Extract SQL from potential conversational output
        if 'SELECT' in sql.upper():
            # Find the SELECT statement
            select_match = re.search(r'SELECT.*?;', sql, re.IGNORECASE | re.DOTALL)
            if select_match:
                sql = select_match.group(0)
            else:
                # Find SELECT to end of string
                select_pos = sql.upper().find('SELECT')
                if select_pos >= 0:
                    sql = sql[select_pos:]
        
        # Clean and validate
        sql = sql.strip()
        
        # If no valid SQL found, create a basic fallback
        if not sql or len(sql) < 10:
            sql = "SELECT * FROM products_with_price"
        
        # Ensure proper SQL format
        if not sql.upper().startswith('SELECT'):
            sql = f"SELECT {sql}"
        
        # Ensure semicolon
        if not sql.endswith(';'):
            sql += ';'
        
        # Final cleanup
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        return sql
    
    def __call__(self, text: str) -> str:
        """Make pipeline callable."""
        return self.generate_sql(text)

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_pipeline(pipeline, eval_data: List[Dict], db_path: str) -> Tuple[Dict, List[Dict]]:
    """Evaluate pipeline on test data with comprehensive metrics."""
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
        'pipeline': 'P1_mT5_prompt_improved',
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
    print("🚀 Starting P1: Improved mT5 Prompt-Only Pipeline Evaluation")
    print("=" * 60)
    
    # Setup Google Drive
    PATHS = setup_google_drive()
    
    # Setup paths
    db_path = PATHS['db'] / "tiki.sqlite"
    data_dir = PATHS['data']
    
    # Create sample database if it doesn't exist
    print("📊 Setting up test environment...")
    if not db_path.exists():
        create_sample_database(str(db_path))
    
    # Load evaluation data - prioritize eval_300.jsonl
    eval_file = None
    for filename in ["eval_300.jsonl", "eval.jsonl"]:
        potential_file = data_dir / filename
        if potential_file.exists():
            eval_file = potential_file
            break
    
    if eval_file is None:
        print("⚠️  No evaluation data found. Creating sample data...")
        # Create minimal sample for testing
        sample_data = [
            {"vn": "Hiển thị tất cả sản phẩm", "sql": "SELECT * FROM products_with_price;", "complexity": "simple"},
            {"vn": "Tìm sản phẩm Samsung", "sql": "SELECT * FROM products_with_price WHERE brand_name = 'Samsung';", "complexity": "simple"},
            {"vn": "Đếm số sản phẩm", "sql": "SELECT COUNT(*) FROM products_with_price;", "complexity": "medium"}
        ]
        eval_file = data_dir / "eval_sample.jsonl"
        with open(eval_file, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\\n')
    
    # Load evaluation data
    eval_data = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                eval_data.append(json.loads(line))
    
    print(f"📝 Loaded {len(eval_data)} evaluation queries from {eval_file.name}")
    
    # Create and test pipeline
    print("\\n🤖 Creating Improved mT5 Prompt Pipeline...")
    pipeline = ImprovedMT5PromptPipeline()
    
    # Quick test
    test_query = "Hiển thị tất cả sản phẩm"
    test_result = pipeline(test_query)
    print(f"🧪 Test Query: {test_query}")
    print(f"🧪 Test Result: {test_result}")
    
    # Run full evaluation
    print("\\n📊 Running full evaluation...")
    metrics, detailed_results = evaluate_pipeline(pipeline, eval_data, str(db_path))
    
    # Display results
    print("\\n" + "=" * 60)
    print("📈 EVALUATION RESULTS - P1: Improved mT5 Prompt-Only")
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
    print("\\n🔍 Sample Predictions:")
    for i in range(min(5, len(detailed_results))):
        result = detailed_results[i]
        print(f"\\nQuery {i+1}: {result['text']}")
        print(f"Gold: {result['gold_sql']}")
        print(f"Pred: {result['pred_sql']}")
        print(f"EM: {result['EM']}, EX: {result['EX']}")
    
    # Save results with timestamp to Google Drive logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_name = "P1_mT5_prompt_improved"
    
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
        "method": "improved-prompt-only",
        "num_eval_queries": len(eval_data),
        "eval_file_used": eval_file.name
    }
    
    env_json_path = PATHS['logs'] / f"{pipeline_name}_{timestamp}_env.json"
    with open(env_json_path, 'w') as f:
        json.dump(env_info, f, indent=2)
    
    print(f"\\n💾 Results saved to Google Drive:")
    print(f"  - {results_csv_path}")
    print(f"  - {metrics_json_path}")
    print(f"  - {env_json_path}")
    
    print("\\n✅ P1 Improved Pipeline evaluation completed!")
    return metrics, detailed_results

# Run the evaluation
if __name__ == "__main__":
    metrics, results = main()
