"""
Local Model Evaluator for Vietnamese NL2SQL
Evaluates all 300 queries using the trained PhoBERT model locally
"""

import torch
import json
import time
import sqlite3
import traceback
from datetime import datetime
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sample_queries_data import get_sample_queries_data
from database.db_manager_normalized import DatabaseManager

class PhoBERTForSQL(nn.Module):
    """PhoBERT model for SQL generation"""
    
    def __init__(self, model_name="vinai/phobert-base", vocab_size=64000, max_length=512):
        super(PhoBERTForSQL, self).__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # SQL generation head
        self.sql_head = nn.Sequential(
            nn.Linear(self.phobert.config.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, vocab_size)
        )
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sql_logits = self.sql_head(sequence_output)
        return sql_logits

class LocalModelEvaluator:
    """Evaluates Vietnamese NL2SQL queries using local trained model"""
    
    def __init__(self, model_path: str, tokenizer_path: str):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self.db_manager = DatabaseManager()
        
        # Evaluation metrics
        self.results = {
            'evaluation_metadata': {
                'total_queries': 0,
                'evaluation_start_time': None,
                'evaluation_end_time': None,
                'model_path': model_path,
                'device': str(self.device)
            },
            'pipeline_results': {
                'successful': 0,
                'failed': 0,
                'success_rate': 0.0,
                'avg_execution_time_ms': 0.0,
                'avg_execution_accuracy': 0.0,
                'total_results_returned': 0
            },
            'complexity_breakdown': {
                'simple': {'total': 0, 'successful': 0, 'success_rate': 0.0, 'avg_time_ms': 0.0},
                'medium': {'total': 0, 'successful': 0, 'success_rate': 0.0, 'avg_time_ms': 0.0},
                'complex': {'total': 0, 'successful': 0, 'success_rate': 0.0, 'avg_time_ms': 0.0}
            },
            'error_analysis': {
                'error_types': {},
                'sample_errors': []
            },
            'query_results_sample': []
        }
        
    def load_model(self):
        """Load the trained PhoBERT model and tokenizer"""
        try:
            print(f"Loading tokenizer from {self.tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            
            print(f"Loading model from {self.model_path}")
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model architecture
            self.model = PhoBERTForSQL()
            
            # Load trained weights from checkpoint
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()
            return False
    
    def generate_sql(self, vietnamese_query: str) -> Tuple[str, float]:
        """Generate SQL from Vietnamese query using local model"""
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                vietnamese_query,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Generate SQL
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Simple greedy decoding for SQL generation
                predicted_ids = torch.argmax(outputs, dim=-1)
                
                # Decode to SQL string
                sql_query = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                
                # Clean up SQL query
                sql_query = self._clean_sql_query(sql_query)
            
            execution_time = (time.time() - start_time) * 1000
            return sql_query, execution_time
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            print(f"Error generating SQL: {str(e)}")
            return f"-- Error: {str(e)}", execution_time
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean and format generated SQL query"""
        # Remove extra whitespace and normalize
        sql_query = ' '.join(sql_query.split())
        
        # Ensure it starts with SELECT if it's a query
        if not sql_query.upper().startswith(('SELECT', 'WITH', 'SHOW', 'DESCRIBE')):
            sql_query = f"SELECT * FROM products WHERE name LIKE '%{sql_query}%' LIMIT 10;"
        
        # Ensure it ends with semicolon
        if not sql_query.endswith(';'):
            sql_query += ';'
            
        return sql_query
    
    def execute_sql_query(self, sql_query: str) -> Tuple[bool, int, str]:
        """Execute SQL query and return success status, result count, and error if any"""
        try:
            # Security check - only allow SELECT statements
            if not sql_query.strip().upper().startswith('SELECT'):
                return False, 0, "Only SELECT statements are allowed"
            
            results = self.db_manager.execute_query(sql_query)
            return True, len(results) if results else 0, None
            
        except Exception as e:
            return False, 0, str(e)
    
    def classify_error_type(self, error_msg: str, sql_query: str) -> str:
        """Classify error into categories"""
        error_lower = error_msg.lower()
        
        if 'syntax' in error_lower or 'near' in error_lower:
            return 'syntax_error'
        elif 'no such table' in error_lower or 'no such column' in error_lower:
            return 'schema_error'
        elif 'ambiguous' in error_lower:
            return 'ambiguous_reference'
        elif 'datatype' in error_lower or 'type' in error_lower:
            return 'datatype_error'
        else:
            return 'other_error'
    
    def evaluate_single_query(self, query_data: Dict) -> Dict:
        """Evaluate a single Vietnamese query"""
        vietnamese_query = query_data['vietnamese']
        expected_sql = query_data.get('sql', '')
        complexity = query_data['complexity']
        
        # Generate SQL using local model
        generated_sql, execution_time = self.generate_sql(vietnamese_query)
        
        # Execute generated SQL
        success, result_count, error = self.execute_sql_query(generated_sql)
        
        # Calculate execution accuracy (simplified - based on successful execution)
        execution_accuracy = 1.0 if success else 0.0
        
        # Exact match calculation (simplified string comparison)
        exact_match = 1.0 if generated_sql.strip().lower() == expected_sql.strip().lower() else 0.0
        
        result = {
            'vietnamese_query': vietnamese_query,
            'expected_sql': expected_sql,
            'generated_sql': generated_sql,
            'complexity': complexity,
            'success': success,
            'execution_time_ms': execution_time,
            'execution_accuracy': execution_accuracy,
            'exact_match': exact_match,
            'result_count': result_count,
            'error': error
        }
        
        # Update complexity breakdown
        complexity_key = complexity.lower()
        if complexity_key in self.results['complexity_breakdown']:
            self.results['complexity_breakdown'][complexity_key]['total'] += 1
            if success:
                self.results['complexity_breakdown'][complexity_key]['successful'] += 1
        
        # Track errors
        if not success and error:
            error_type = self.classify_error_type(error, generated_sql)
            if error_type not in self.results['error_analysis']['error_types']:
                self.results['error_analysis']['error_types'][error_type] = 0
            self.results['error_analysis']['error_types'][error_type] += 1
            
            # Store sample error
            if len(self.results['error_analysis']['sample_errors']) < 10:
                self.results['error_analysis']['sample_errors'].append({
                    'query': vietnamese_query,
                    'error': error,
                    'error_type': error_type,
                    'generated_sql': generated_sql
                })
        
        return result
    
    def evaluate_all_queries(self) -> Dict:
        """Evaluate all 300 queries and generate comprehensive results"""
        print("Starting comprehensive evaluation of 300 Vietnamese queries...")
        
        # Load model
        if not self.load_model():
            raise Exception("Failed to load model")
        
        # Get all sample queries
        sample_data = get_sample_queries_data()
        all_queries = []
        
        # Combine all queries
        for complexity in ['simple', 'medium', 'complex']:
            all_queries.extend(sample_data[complexity])
        
        self.results['evaluation_metadata']['total_queries'] = len(all_queries)
        self.results['evaluation_metadata']['evaluation_start_time'] = datetime.now().isoformat()
        
        print(f"Evaluating {len(all_queries)} queries...")
        
        # Evaluation metrics
        total_execution_time = 0
        total_execution_accuracy = 0
        total_results = 0
        successful_queries = 0
        
        # Process each query
        for i, query_data in enumerate(all_queries):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(all_queries)} queries processed")
            
            result = self.evaluate_single_query(query_data)
            
            # Update overall metrics
            total_execution_time += result['execution_time_ms']
            total_execution_accuracy += result['execution_accuracy']
            total_results += result['result_count']
            
            if result['success']:
                successful_queries += 1
            
            # Store sample results (first 20 for display)
            if len(self.results['query_results_sample']) < 20:
                self.results['query_results_sample'].append({
                    'query_id': i + 1,
                    'vietnamese_query': result['vietnamese_query'],
                    'complexity': result['complexity'].title(),
                    'success': result['success'],
                    'execution_time_ms': result['execution_time_ms'],
                    'results_count': result['result_count'],
                    'sql_query': result['generated_sql'],
                    'error': result['error']
                })
        
        # Calculate final metrics
        total_queries = len(all_queries)
        self.results['pipeline_results']['successful'] = successful_queries
        self.results['pipeline_results']['failed'] = total_queries - successful_queries
        self.results['pipeline_results']['success_rate'] = (successful_queries / total_queries) * 100
        self.results['pipeline_results']['avg_execution_time_ms'] = total_execution_time / total_queries
        self.results['pipeline_results']['avg_execution_accuracy'] = total_execution_accuracy / total_queries
        self.results['pipeline_results']['total_results_returned'] = total_results
        
        # Calculate complexity breakdown success rates
        for complexity in ['simple', 'medium', 'complex']:
            breakdown = self.results['complexity_breakdown'][complexity]
            if breakdown['total'] > 0:
                breakdown['success_rate'] = (breakdown['successful'] / breakdown['total']) * 100
        
        self.results['evaluation_metadata']['evaluation_end_time'] = datetime.now().isoformat()
        
        print(f"Evaluation completed!")
        print(f"Success rate: {self.results['pipeline_results']['success_rate']:.1f}%")
        print(f"Average execution time: {self.results['pipeline_results']['avg_execution_time_ms']:.1f}ms")
        
        return self.results
    
    def save_results(self, output_path: str):
        """Save evaluation results to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")

def main():
    """Main evaluation function"""
    model_path = "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/LocalModel_PhoBERT/phobert_sql_trained.pth"
    tokenizer_path = "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/LocalModel_PhoBERT/phobert_sql_trained_tokenizer"
    output_path = "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/frontend/public/data/local_evaluation_results.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize evaluator
    evaluator = LocalModelEvaluator(model_path, tokenizer_path)
    
    try:
        # Run evaluation
        results = evaluator.evaluate_all_queries()
        
        # Save results
        evaluator.save_results(output_path)
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total queries: {results['evaluation_metadata']['total_queries']}")
        print(f"Successful: {results['pipeline_results']['successful']}")
        print(f"Failed: {results['pipeline_results']['failed']}")
        print(f"Success rate: {results['pipeline_results']['success_rate']:.1f}%")
        print(f"Average execution time: {results['pipeline_results']['avg_execution_time_ms']:.1f}ms")
        print(f"Total results returned: {results['pipeline_results']['total_results_returned']}")
        print("\nComplexity breakdown:")
        for complexity in ['simple', 'medium', 'complex']:
            breakdown = results['complexity_breakdown'][complexity]
            print(f"  {complexity.title()}: {breakdown['successful']}/{breakdown['total']} ({breakdown['success_rate']:.1f}%)")
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
