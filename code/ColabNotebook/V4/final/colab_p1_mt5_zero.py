"""
Vietnamese NL2SQL Pipeline - P1: mT5 Zero-Shot Prompting
MSE Thesis 2025 - Vietnamese Natural Language to SQL Generation

Author: Duong Dinh Dinh
Student ID: tho23mse23108
Class: MSE14
Copyright (c) 2025
"""

# ============================================================================
# CELL 1: Environment Setup & Imports
# ============================================================================

# Install packages (run once)
import subprocess
import sys

def install_packages():
    packages = [
        "transformers>=4.30.0", "datasets>=2.10.0", "torch>=2.0.0",
        "pandas>=1.5.0", "numpy>=1.24.0", "tqdm>=4.65.0",
        "fastapi>=0.100.0", "uvicorn>=0.23.0", "nest-asyncio>=1.5.0", "pyngrok>=6.0.0"
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Uncomment on first run: install_packages()

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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
        print(f"P1 Database connection error: {e}")
        return 0

def is_valid_sql(sql: str) -> bool:
    """Check if SQL is valid"""
    if not sql or not sql.strip():
        return False
    sql = sql.strip().lower()
    return sql.startswith('select') and 'from' in sql and sql.count('(') == sql.count(')')

# DEBUG FUNCTION - Commented out for production
# def debug_em_ex_mismatch(pred: str, gold: str, db_path: str) -> dict:
    """Debug cases where EM=1 but EX=0 for P1"""
    em = exact_match(pred, gold)
    ex = execution_accuracy(pred, gold, db_path)
    
    debug_info = {
        'pipeline': 'P1_mT5',
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
# CELL 4: P1 - Prompting Pipeline (mT5)
# ============================================================================

class PromptingPipeline:
    """mT5 based prompting pipeline for Vietnamese NL2SQL"""
    
    def __init__(self, model_name: str = "google/mt5-base"):
        self.model_name = model_name
        self.device = device
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
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
        """Create schema-aware few-shot prompt for Vietnamese to SQL"""
        is_count = self.detect_count_intent(vietnamese_text)
        
        prompt = f"""You are a SQL generator for a Vietnamese e-commerce app.

Rules:
- Use SQLite syntax.
- Database has ONE table: products(id, name, category, brand, price, description).
- If the user asks to "tìm/hiển thị/liệt kê/xem" something by name, use:
  SELECT * FROM products WHERE name LIKE '%<term>%' LIMIT 10;
- Do NOT use COUNT(*) unless the user explicitly asks to count ("đếm", "bao nhiêu", "số lượng").
- Output exactly ONE SQL statement ending with a semicolon. No explanations.

Examples:
VN: "Tìm áo thun"
SQL: SELECT * FROM products WHERE name LIKE '%áo thun%' LIMIT 10;

VN: "Hiển thị giày"
SQL: SELECT * FROM products WHERE name LIKE '%giày%' LIMIT 10;

VN: "Xem túi xách"
SQL: SELECT * FROM products WHERE name LIKE '%túi xách%' LIMIT 10;

VN: "Có bao nhiêu áo thun?"
SQL: SELECT COUNT(*) FROM products WHERE name LIKE '%áo thun%';

VN: "{vietnamese_text}"
SQL:"""
        
        # Add explicit COUNT ban if no count intent
        if not is_count:
            prompt += "\n(Never use COUNT(*) for this query.)"
        
        return prompt
    
    def generate_sql(self, vietnamese_text: str) -> str:
        """Generate SQL from Vietnamese text"""
        prompt = self.create_prompt(vietnamese_text)
        
        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", max_length=512, 
                truncation=True, padding=False
            ).to(self.device)
            
            # Block extra_id tokens for mT5
            bad_words = []
            for i in range(200):
                token_id = self.tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
                if token_id != self.tokenizer.unk_token_id:
                    bad_words.append([token_id])
            
            gen_kwargs = {
                'max_new_tokens': 64,        # Give space for full query
                'temperature': 0.0,          # Deterministic
                'num_beams': 4,              # Helps with structure
                'early_stopping': True,
                'do_sample': False,
                'no_repeat_ngram_size': 2,
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'bad_words_ids': bad_words if bad_words else None
            }
            
            if "t5" in self.model_name.lower():
                gen_kwargs["decoder_start_token_id"] = self.tokenizer.pad_token_id
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return self.post_process_sql(generated_text, vietnamese_text)
            
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
    
    def post_process_sql(self, generated_text: str, vietnamese_text: str) -> str:
        """Post-process and validate generated SQL"""
        if not generated_text:
            return ""
        
        # Extract SQL from generated text
        sql = self.extract_sql(generated_text)
        
        # Validate and fix common issues
        is_count = self.detect_count_intent(vietnamese_text)
        
        # If we got COUNT but didn't want it, try to fix
        if 'count(' in sql.lower() and not is_count:
            search_term = self.extract_search_term(vietnamese_text)
            if search_term:
                sql = f"SELECT * FROM products WHERE name LIKE '%{search_term}%' LIMIT 10;"
            else:
                sql = "SELECT * FROM products LIMIT 10;"
        
        # If we have no WHERE clause but should (non-empty search term)
        elif 'where' not in sql.lower() and not is_count:
            search_term = self.extract_search_term(vietnamese_text)
            if search_term and search_term != vietnamese_text.strip():
                sql = f"SELECT * FROM products WHERE name LIKE '%{search_term}%' LIMIT 10;"
        
        # Ensure proper formatting
        sql = self.normalize_sql_format(sql)
        
        return sql
    
    def extract_sql(self, generated_text: str) -> str:
        """Extract SQL from generated text"""
        if not generated_text:
            return ""
        
        # Extract first SELECT...;
        match = re.search(r'(?is)\bselect\b.*?;', generated_text)
        if match:
            return match.group(0).strip()
        
        # If no semicolon, add it
        if not generated_text.endswith(';'):
            generated_text += ';'
        
        return generated_text
    
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
    
    # P1: Prompting Pipeline
    print("\nP1: Testing Prompting Pipeline (mT5)")
    pipeline_p1 = PromptingPipeline()
    
    # Quick test with diverse examples
    test_queries = [
        "Hiển thị tất cả sản phẩm",
        "Tìm áo thun", 
        "Xem giày dép",
        "Có bao nhiêu túi xách?"  # Count intent test
    ]
    print("\nQuick Test:")
    for query in test_queries:
        sql = pipeline_p1.generate_sql(query)
        print(f"  {query} -> {sql}")
    
    # Full evaluation
    metrics_p1, results_p1 = evaluate_pipeline(
        pipeline_p1, eval_data, str(db_path), "P1_Prompting_mT5"
    )
    
    # Display results
    print("\nP1 RESULTS:")
    print(f"Exact Match (EM): {metrics_p1['EM']:.3f}")
    print(f"Execution Accuracy (EX): {metrics_p1['EX']:.3f}")
    print(f"Model Success Rate: {metrics_p1['Model_Success_Rate']:.3f}")
    print(f"Latency: {metrics_p1['Latency_mean']:.3f}s")
    print(f"GPU Memory: {metrics_p1['GPU_peak_GB']:.2f} GB")
    
    print(f"\nSuccess by Complexity:")
    print(f"  Simple: {metrics_p1['simple_success']:.3f}")
    print(f"  Medium: {metrics_p1['medium_success']:.3f}")
    print(f"  Complex: {metrics_p1['complex_success']:.3f}")
    
    # Show sample results (mixed complexity)
    print(f"\nSample Results (10 mixed complexity queries):")
    sample_results = []
    for complexity in ['simple', 'medium', 'complex']:
        complexity_results = [r for r in results_p1 if r.get('complexity') == complexity]
        sample_results.extend(complexity_results[:4])  # 3-4 per complexity
    
    for i, result in enumerate(sample_results[:10]):
        print(f"\n{i+1}. {result['text']} ({result.get('complexity', 'unknown')})")
        print(f"   Gold: {result['gold_sql']}")
        print(f"   Pred: {result['pred_sql']}")
        print(f"   EM={result['EM']}, EX={result['EX']}, Valid={result['valid']}")
    
    # Save results
    save_results(metrics_p1, results_p1, "P1_Prompting_mT5", PATHS)
    
    print("\nEvaluation completed!")
    return metrics_p1, results_p1, pipeline_p1

# Run the evaluation
if eval_data:
    metrics, results, pipeline = run_evaluation()
else:
    print("Upload eval_data.jsonl to Google Drive to run evaluation")
    pipeline = None

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
# CELL 7: FastAPI Setup for P1 mT5 Zero-Shot
# ============================================================================

print("\n" + "=" * 60)
print("Setting up FastAPI for P1: mT5 Zero-Shot Pipeline")
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
import nest_asyncio
from pyngrok import ngrok

nest_asyncio.apply()

# Set up ngrok with token
ngrok.set_auth_token("32BqVAspvTl3PmS23seCfxTxW93_7p3vCzKHixcdNg936rpXv")

# Create FastAPI app
app = FastAPI(
    title="Vietnamese NL2SQL - P1: mT5 Zero-Shot API",
    description="Pipeline 1: Direct Vietnamese→SQL generation using mT5 multilingual model with zero-shot prompting",
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

class P1Response(BaseModel):
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
        "message": "Vietnamese NL2SQL - P1: mT5 Zero-Shot",
        "version": "1.0",
        "status": "running",
        "device": str(device),
        "pipeline": "P1_mT5_Zero_Shot",
        "method": "Direct Vietnamese→SQL prompting",
        "model": "google/mt5-base",
        "ready": pipeline is not None,
        "endpoint": "/p1/generate"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0",
        "pipeline": "P1",
        "model_loaded": pipeline is not None,
        "device": str(device)
    }

@app.post("/p1/generate", response_model=P1Response)
async def generate_sql_p1(request: QueryRequest):
    """Generate SQL from Vietnamese query using P1 mT5 Zero-Shot"""
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
        valid = bool(sql and len(sql.strip()) > 5)
        
        return P1Response(
            pipeline="P1_mT5_Zero_Shot",
            sql_query=sql,
            execution_time=execution_time,
            valid=valid,
            success=valid,
            error=None if valid else "Generated SQL is empty or too short",
            metrics={
                "latency_ms": execution_time * 1000,
                "model": "google/mt5-base",
                "method": "zero_shot_prompting"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"P1 generation failed: {str(e)}"
        return P1Response(
            pipeline="P1_mT5_Zero_Shot",
            sql_query="",
            execution_time=0,
            valid=False,
            success=False,
            error=error_msg,
            metrics={}
        )

@app.get("/p1/metrics")
async def get_p1_metrics():
    """Get evaluation metrics for P1"""
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics not available. Run evaluation first.")
    
    return {
        "pipeline": "P1_mT5_Zero_Shot",
        "metrics": metrics,
        "description": "Zero-shot prompting with mT5 multilingual model"
    }

print("FastAPI app configured for P1")

# ============================================================================
# CELL 8: Start ngrok Tunnel and FastAPI Server for P1
# ============================================================================

print("\nStarting ngrok tunnel for P1: mT5 Zero-Shot...")
try:
    # Use custom domain - all pipelines share this domain with different paths
    public_url = ngrok.connect(8000, domain="abnormally-direct-rhino.ngrok-free.app")
    print(f"P1 API URL: {public_url}")
    print(f"P1 Generate Endpoint: {public_url}/p1/generate")
    
    api_url = f"{public_url}"
    print(f"\nP1 mT5 Zero-Shot API is available at:")
    print(f"  Base URL: {api_url}")
    print(f"  Health Check: {api_url}/health")
    print(f"  API Docs: {api_url}/docs")
    print(f"  Generate SQL: {api_url}/p1/generate (POST)")
    print(f"  View Metrics: {api_url}/p1/metrics (GET)")
    
    # Test health endpoint
    print(f"\nTesting P1 server health...")
    import requests
    try:
        health_response = requests.get(f"{api_url}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"P1 Health check passed: {health_data['status']}")
            print(f"Model loaded: {'Yes' if health_data['model_loaded'] else 'No'}")
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

print(f"\nStarting P1 FastAPI server on port 8000...")
print("Keep this cell running to maintain the API!")
print(f"Configure this URL in your local system: {api_url}")
print("\n" + "=" * 60)
print("EXAMPLE CURL REQUEST:")
print(f'curl -X POST "{api_url}/p1/generate" \\')
print('     -H "Content-Type: application/json" \\')
print('     -d \'{"query": "Hiển thị tất cả sản phẩm"}\'')
print("=" * 60)

# Start server
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
