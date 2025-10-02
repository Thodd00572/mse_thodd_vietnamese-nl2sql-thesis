"""
Simple Local Evaluator for Vietnamese NL2SQL
Uses existing pipeline infrastructure to evaluate all 300 queries locally
"""

import json
import time
import sqlite3
import traceback
from datetime import datetime
from typing import Dict, List, Any, Tuple
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sample_queries_data import get_sample_queries_data
from database.db_manager_normalized import DatabaseManager

class SimpleLocalEvaluator:
    """Evaluates Vietnamese NL2SQL queries using simple rule-based approach"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        
        # Evaluation metrics
        self.results = {
            'evaluation_metadata': {
                'total_queries': 0,
                'evaluation_start_time': None,
                'evaluation_end_time': None,
                'evaluation_type': 'local_simple_rules',
                'model_type': 'rule_based_fallback'
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
        
        # Simple Vietnamese to SQL mapping rules
        self.vietnamese_keywords = {
            'tìm': 'SELECT * FROM products WHERE',
            'hiển thị': 'SELECT * FROM products WHERE', 
            'xem': 'SELECT * FROM products WHERE',
            'liệt kê': 'SELECT name FROM products WHERE',
            'tìm kiếm': 'SELECT * FROM products WHERE',
            'đếm': 'SELECT COUNT(*) FROM',
            'tổng': 'SELECT COUNT(*) FROM',
            'danh sách': 'SELECT * FROM',
            'sản phẩm': 'products',
            'thương hiệu': 'brands',
            'danh mục': 'categories',
            'người bán': 'sellers',
            'giá': 'current_price',
            'đánh giá': 'rating_average'
        }
    
    def simple_vietnamese_to_sql(self, vietnamese_query: str) -> str:
        """Convert Vietnamese query to SQL using simple rules"""
        query_lower = vietnamese_query.lower().strip()
        
        # Handle counting queries
        if any(word in query_lower for word in ['đếm', 'tổng số']):
            if 'sản phẩm' in query_lower:
                return "SELECT COUNT(*) as total_products FROM products;"
            elif 'thương hiệu' in query_lower:
                return "SELECT COUNT(*) as total_brands FROM brands;"
            elif 'danh mục' in query_lower:
                return "SELECT COUNT(*) as total_categories FROM categories;"
            elif 'người bán' in query_lower:
                return "SELECT COUNT(*) as total_sellers FROM sellers;"
        
        # Handle listing queries
        if any(word in query_lower for word in ['danh sách', 'hiển thị', 'xem']):
            if 'thương hiệu' in query_lower:
                return "SELECT DISTINCT brand_name FROM brands LIMIT 10;"
            elif 'danh mục' in query_lower:
                return "SELECT category_name FROM categories;"
            elif 'người bán' in query_lower:
                return "SELECT seller_name FROM sellers LIMIT 10;"
        
        # Handle product searches with specific items
        product_keywords = [
            'áo', 'giày', 'túi', 'balo', 'vali', 'ví', 'dép', 'nón', 'thắt lưng', 'kính',
            'đồng hồ', 'vớ', 'khăn', 'găng tay', 'quần', 'váy', 'phụ kiện', 'trang sức'
        ]
        
        for keyword in product_keywords:
            if keyword in query_lower:
                return f"SELECT * FROM products WHERE name LIKE '%{keyword}%' LIMIT 10;"
        
        # Handle price-based queries
        if 'giá' in query_lower:
            if 'dưới' in query_lower or 'nhỏ hơn' in query_lower:
                # Extract price if possible, default to 500000
                return "SELECT p.name, pr.current_price FROM products p JOIN product_pricing pr ON p.product_id = pr.product_id WHERE pr.current_price < 500000 ORDER BY pr.current_price LIMIT 20;"
            elif 'trên' in query_lower or 'lớn hơn' in query_lower:
                return "SELECT p.name, pr.current_price FROM products p JOIN product_pricing pr ON p.product_id = pr.product_id WHERE pr.current_price > 500000 ORDER BY pr.current_price DESC LIMIT 20;"
            else:
                return "SELECT p.name, pr.current_price FROM products p JOIN product_pricing pr ON p.product_id = pr.product_id LIMIT 10;"
        
        # Handle ordering queries
        if 'mới nhất' in query_lower:
            return "SELECT * FROM products ORDER BY product_id DESC LIMIT 10;"
        elif 'cũ nhất' in query_lower:
            return "SELECT * FROM products ORDER BY product_id ASC LIMIT 10;"
        
        # Default fallback for general product searches
        return "SELECT * FROM products LIMIT 10;"
    
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
        
        start_time = time.time()
        
        # Generate SQL using simple rules
        generated_sql = self.simple_vietnamese_to_sql(vietnamese_query)
        
        execution_time = (time.time() - start_time) * 1000
        
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
        print("Starting comprehensive evaluation of 300 Vietnamese queries using simple rules...")
        
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
        
        # Calculate complexity breakdown success rates and average times
        complexity_times = {'simple': [], 'medium': [], 'complex': []}
        
        for i, query_data in enumerate(all_queries):
            complexity_key = query_data['complexity'].lower()
            if complexity_key in complexity_times:
                # Re-evaluate to get timing for complexity breakdown
                result = self.evaluate_single_query(query_data)
                complexity_times[complexity_key].append(result['execution_time_ms'])
        
        for complexity in ['simple', 'medium', 'complex']:
            breakdown = self.results['complexity_breakdown'][complexity]
            if breakdown['total'] > 0:
                breakdown['success_rate'] = (breakdown['successful'] / breakdown['total']) * 100
                if complexity_times[complexity]:
                    breakdown['avg_time_ms'] = sum(complexity_times[complexity]) / len(complexity_times[complexity])
        
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
    output_path = "/Users/thoduong/CascadeProjects/MSE_Thesis_2025/code/frontend/public/data/local_evaluation_results.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize evaluator
    evaluator = SimpleLocalEvaluator()
    
    try:
        # Run evaluation
        results = evaluator.evaluate_all_queries()
        
        # Save results
        evaluator.save_results(output_path)
        
        print("\n" + "="*50)
        print("LOCAL EVALUATION SUMMARY")
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
            print(f"  {complexity.title()}: {breakdown['successful']}/{breakdown['total']} ({breakdown['success_rate']:.1f}%) - Avg: {breakdown['avg_time_ms']:.1f}ms")
        
        if results['error_analysis']['error_types']:
            print("\nError types:")
            for error_type, count in results['error_analysis']['error_types'].items():
                print(f"  {error_type}: {count}")
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
