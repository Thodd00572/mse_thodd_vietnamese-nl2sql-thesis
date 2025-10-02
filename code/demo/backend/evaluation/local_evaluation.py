#!/usr/bin/env python3
"""
Local Evaluation Script for Vietnamese NL2SQL API
Runs comprehensive evaluation by calling API endpoints
"""

import os
import sys
import time
import json
import sqlite3
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE GENERATION
# =============================================================================

def generate_sample_database(db_path: str = "./vietnamese_ecommerce.db"):
    """Generate sample database with Vietnamese e-commerce products"""
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create products table
    cursor.execute('''
    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        brand TEXT,
        category TEXT,
        price DECIMAL(10,2),
        rating DECIMAL(3,2),
        review_count INTEGER,
        color TEXT,
        material TEXT,
        gender TEXT,
        size TEXT,
        description TEXT
    )
    ''')
    
    # Sample data for Vietnamese e-commerce
    sample_products = [
        # Balo (Backpacks)
        (1, 'Balo laptop Samsonite đen', 'Samsonite', 'balo', 1200000, 4.5, 150, 'đen', 'vải', 'unisex', 'L', 'Balo laptop cao cấp'),
        (2, 'Balo du lịch Nike xanh', 'Nike', 'balo', 800000, 4.2, 89, 'xanh', 'vải', 'unisex', 'M', 'Balo thể thao du lịch'),
        (3, 'Balo nữ Adidas hồng', 'Adidas', 'balo', 650000, 4.3, 76, 'hồng', 'vải', 'nữ', 'S', 'Balo thời trang nữ'),
        
        # Túi xách (Handbags)
        (4, 'Túi xách da Gucci đen', 'Gucci', 'túi xách', 2500000, 4.8, 45, 'đen', 'da', 'nữ', 'M', 'Túi xách da cao cấp'),
        (5, 'Túi xách vải Chanel trắng', 'Chanel', 'túi xách', 3200000, 4.9, 32, 'trắng', 'vải', 'nữ', 'L', 'Túi xách thời trang'),
        (6, 'Túi xách da nâu Louis Vuitton', 'Louis Vuitton', 'túi xách', 4500000, 4.7, 28, 'nâu', 'da', 'nữ', 'M', 'Túi xách luxury'),
        
        # Giày (Shoes)
        (7, 'Giày thể thao Nike Air Max', 'Nike', 'giày', 2200000, 4.6, 234, 'trắng', 'da', 'nam', '42', 'Giày chạy bộ'),
        (8, 'Giày boots Timberland nâu', 'Timberland', 'giày', 3500000, 4.4, 156, 'nâu', 'da', 'nam', '43', 'Giày boots cao cổ'),
        (9, 'Giày cao gót đen', 'Zara', 'giày', 1200000, 4.1, 67, 'đen', 'da', 'nữ', '38', 'Giày cao gót công sở'),
        (10, 'Giày sneaker Adidas trắng', 'Adidas', 'giày', 1800000, 4.5, 189, 'trắng', 'vải', 'unisex', '41', 'Giày thể thao casual'),
        
        # Kính mát (Sunglasses)
        (11, 'Kính mát Ray-Ban đen', 'Ray-Ban', 'kính mát', 3200000, 4.7, 89, 'đen', 'kim loại', 'unisex', 'M', 'Kính mát thời trang'),
        (12, 'Kính mát Oakley xanh', 'Oakley', 'kính mát', 2800000, 4.5, 67, 'xanh', 'nhựa', 'nam', 'L', 'Kính mát thể thao'),
        (13, 'Kính mát nữ Dior hồng', 'Dior', 'kính mát', 4500000, 4.8, 34, 'hồng', 'kim loại', 'nữ', 'S', 'Kính mát cao cấp'),
        
        # Vali (Suitcases)
        (14, 'Vali American Tourister đen', 'American Tourister', 'vali', 2200000, 4.3, 123, 'đen', 'nhựa', 'unisex', 'L', 'Vali du lịch cứng'),
        (15, 'Vali Samsonite xanh', 'Samsonite', 'vali', 3500000, 4.6, 89, 'xanh', 'nhựa', 'unisex', 'XL', 'Vali cao cấp'),
        (16, 'Vali vải đỏ', 'Delsey', 'vali', 1800000, 4.2, 56, 'đỏ', 'vải', 'unisex', 'M', 'Vali vải mềm')
    ]
    
    cursor.executemany('''
    INSERT INTO products (id, name, brand, category, price, rating, review_count, color, material, gender, size, description)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', sample_products)
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database created successfully at: {db_path}")
    logger.info(f"Added {len(sample_products)} sample products")
    return db_path

# =============================================================================
# EVALUATION QUERIES GENERATOR
# =============================================================================

class EvaluationQueryGenerator:
    """Generate evaluation queries for testing"""
    
    def __init__(self):
        self.simple_queries = [
            "Tìm tất cả balo",
            "Tìm giày thể thao",
            "Tìm túi xách nữ",
            "Tìm kính mát",
            "Tìm vali",
            "Tìm sản phẩm Nike",
            "Tìm sản phẩm Adidas",
            "Tìm sản phẩm màu đen",
            "Tìm sản phẩm màu trắng",
            "Tìm sản phẩm cho nam"
        ]
        
        self.medium_queries = [
            "Tìm balo màu đen giá dưới 1 triệu",
            "Tìm giày thể thao Nike màu trắng",
            "Tìm túi xách nữ có đánh giá trên 4.5",
            "Tìm kính mát giá từ 2 triệu đến 4 triệu",
            "Tìm vali Samsonite có đánh giá cao",
            "Tìm sản phẩm da có giá dưới 2 triệu",
            "Tìm giày nữ size 38",
            "Tìm balo unisex size L",
            "Tìm sản phẩm có nhiều đánh giá",
            "Tìm sản phẩm giá rẻ nhất"
        ]
        
        self.complex_queries = [
            "Tìm top 5 sản phẩm có đánh giá cao nhất",
            "Tìm sản phẩm Nike hoặc Adidas giá dưới 2 triệu",
            "Tìm trung bình giá của từng loại sản phẩm",
            "Tìm thương hiệu có nhiều sản phẩm nhất",
            "Tìm sản phẩm có giá cao nhất trong từng danh mục",
            "Đếm số sản phẩm theo màu sắc",
            "Tìm sản phẩm có tỷ lệ đánh giá/giá tốt nhất",
            "So sánh giá trung bình giữa sản phẩm nam và nữ",
            "Tìm sản phẩm có nhiều review nhất trong từng thương hiệu",
            "Tính tổng giá trị của tất cả sản phẩm"
        ]
    
    def generate_evaluation_dataset(self, queries_per_complexity: int = 100) -> List[Dict]:
        """Generate evaluation dataset with specified number of queries per complexity"""
        dataset = []
        
        # Generate queries for each complexity level
        for complexity, base_queries in [
            ('simple', self.simple_queries),
            ('medium', self.medium_queries), 
            ('complex', self.complex_queries)
        ]:
            # Repeat and vary base queries to reach target count
            for i in range(queries_per_complexity):
                query_idx = i % len(base_queries)
                query = base_queries[query_idx]
                
                dataset.append({
                    'id': len(dataset) + 1,
                    'vietnamese': query,
                    'complexity': complexity,
                    'expected_sql': None  # Would need manual annotation for proper evaluation
                })
        
        return dataset

# =============================================================================
# API CLIENT
# =============================================================================

class NL2SQLAPIClient:
    """Client for Vietnamese NL2SQL API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def query(self, vietnamese_query: str) -> Dict:
        """Send query to API"""
        try:
            payload = {
                "vietnamese_query": vietnamese_query,
                "include_metrics": True
            }
            
            response = self.session.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {
                "sql_query": "",
                "success": False,
                "execution_time": 0.0,
                "error": str(e),
                "pipeline": "Pipeline1"
            }

# =============================================================================
# DATABASE EXECUTOR
# =============================================================================

def execute_sql_query(sql_query: str, db_path: str) -> Dict:
    """Execute SQL query against the database and return results"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
        return {'success': True, 'results': results, 'error': None, 'row_count': len(results)}
    except Exception as e:
        return {'success': False, 'results': None, 'error': str(e), 'row_count': 0}

# =============================================================================
# EVALUATION RUNNER
# =============================================================================

class EvaluationRunner:
    """Run comprehensive evaluation of Vietnamese NL2SQL API"""
    
    def __init__(self, api_url: str = "http://localhost:8000", db_path: str = "./vietnamese_ecommerce.db"):
        self.api_client = NL2SQLAPIClient(api_url)
        self.db_path = db_path
        self.query_generator = EvaluationQueryGenerator()
    
    def run_evaluation(self, queries_per_complexity: int = 100) -> Dict:
        """Run comprehensive evaluation"""
        logger.info("Starting Vietnamese NL2SQL evaluation...")
        
        # Check API health
        health = self.api_client.health_check()
        if health.get("status") != "healthy":
            raise RuntimeError(f"API not healthy: {health}")
        
        logger.info("✅ API is healthy")
        
        # Generate database
        logger.info("Generating sample database...")
        generate_sample_database(self.db_path)
        
        # Generate evaluation dataset
        logger.info(f"Generating evaluation dataset ({queries_per_complexity} per complexity)...")
        dataset = self.query_generator.generate_evaluation_dataset(queries_per_complexity)
        
        # Initialize results
        results = {
            'total_queries': len(dataset),
            'successful_queries': 0,
            'failed_queries': 0,
            'db_execution_success': 0,
            'total_latency': 0.0,
            'complexity_results': {
                'simple': {'total': 0, 'success': 0, 'db_success': 0, 'avg_latency': 0.0},
                'medium': {'total': 0, 'success': 0, 'db_success': 0, 'avg_latency': 0.0},
                'complex': {'total': 0, 'success': 0, 'db_success': 0, 'avg_latency': 0.0}
            },
            'query_results': []
        }
        
        # Process each query
        for i, query_data in enumerate(dataset, 1):
            logger.info(f"Processing query {i}/{len(dataset)}: {query_data['vietnamese']}")
            
            # Measure latency with multiple runs
            latencies = []
            api_result = None
            
            for run in range(3):  # 3 runs for latency averaging
                start_time = time.time()
                api_result = self.api_client.query(query_data['vietnamese'])
                latency = time.time() - start_time
                latencies.append(latency)
            
            avg_latency = sum(latencies) / len(latencies)
            results['total_latency'] += avg_latency
            
            complexity = query_data['complexity']
            results['complexity_results'][complexity]['total'] += 1
            results['complexity_results'][complexity]['avg_latency'] += avg_latency
            
            # Check API success
            if api_result.get('success', False):
                results['successful_queries'] += 1
                results['complexity_results'][complexity]['success'] += 1
                
                # Test database execution
                sql_query = api_result.get('sql_query', '')
                if sql_query.strip():
                    db_result = execute_sql_query(sql_query, self.db_path)
                    if db_result['success']:
                        results['db_execution_success'] += 1
                        results['complexity_results'][complexity]['db_success'] += 1
                        logger.info(f"  ✅ SQL executed successfully ({db_result['row_count']} rows)")
                    else:
                        logger.info(f"  ❌ SQL execution failed: {db_result['error']}")
                else:
                    logger.info(f"  ⚠️ Empty SQL generated")
            else:
                results['failed_queries'] += 1
                logger.info(f"  ❌ API call failed: {api_result.get('error', 'Unknown error')}")
            
            # Store individual result
            results['query_results'].append({
                'id': query_data['id'],
                'vietnamese': query_data['vietnamese'],
                'complexity': complexity,
                'sql_query': api_result.get('sql_query', ''),
                'api_success': api_result.get('success', False),
                'api_latency': avg_latency,
                'api_error': api_result.get('error'),
                'db_execution': execute_sql_query(api_result.get('sql_query', ''), self.db_path) if api_result.get('sql_query', '').strip() else None
            })
        
        # Calculate final metrics
        self._calculate_final_metrics(results)
        
        return results
    
    def _calculate_final_metrics(self, results: Dict):
        """Calculate final evaluation metrics"""
        total = results['total_queries']
        
        # Overall metrics
        results['success_rate'] = results['successful_queries'] / total if total > 0 else 0.0
        results['db_execution_rate'] = results['db_execution_success'] / total if total > 0 else 0.0
        results['avg_latency'] = results['total_latency'] / total if total > 0 else 0.0
        
        # Complexity-specific metrics
        for complexity, stats in results['complexity_results'].items():
            if stats['total'] > 0:
                stats['success_rate'] = stats['success'] / stats['total']
                stats['db_success_rate'] = stats['db_success'] / stats['total']
                stats['avg_latency'] = stats['avg_latency'] / stats['total']
    
    def save_results(self, results: Dict, output_path: str = "./evaluation_results.json"):
        """Save evaluation results to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_path}")
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("VIETNAMESE NL2SQL EVALUATION RESULTS")
        print("="*80)
        print(f"Total Queries: {results['total_queries']}")
        print(f"API Success Rate: {results['success_rate']:.3f} ({results['successful_queries']}/{results['total_queries']})")
        print(f"DB Execution Rate: {results['db_execution_rate']:.3f} ({results['db_execution_success']}/{results['total_queries']})")
        print(f"Average Latency: {results['avg_latency']*1000:.1f}ms")
        
        print(f"\nCOMPLEXITY BREAKDOWN:")
        for complexity, stats in results['complexity_results'].items():
            print(f"{complexity.upper()}:")
            print(f"  API Success: {stats['success_rate']:.3f} ({stats['success']}/{stats['total']})")
            print(f"  DB Success: {stats['db_success_rate']:.3f} ({stats['db_success']}/{stats['total']})")
            print(f"  Avg Latency: {stats['avg_latency']*1000:.1f}ms")
        
        print("="*80)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vietnamese NL2SQL Local Evaluation")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API server URL")
    parser.add_argument("--queries-per-complexity", type=int, default=100, help="Number of queries per complexity level")
    parser.add_argument("--output", default="./evaluation_results.json", help="Output file for results")
    parser.add_argument("--db-path", default="./vietnamese_ecommerce.db", help="Database path")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = EvaluationRunner(args.api_url, args.db_path)
    results = evaluator.run_evaluation(args.queries_per_complexity)
    
    # Save and display results
    evaluator.save_results(results, args.output)
    evaluator.print_summary(results)

if __name__ == "__main__":
    main()
