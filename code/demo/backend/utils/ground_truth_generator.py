"""
Ground Truth Generator for Vietnamese NL2SQL Sample Queries
Executes all 300 sample queries and stores results in JSON for comparison
"""

import json
import sqlite3
import hashlib
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from sample_queries_data import SAMPLE_QUERIES_DATA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroundTruthGenerator:
    """
    Generate and store ground truth execution results for all sample queries
    """
    
    def __init__(self, db_path: str, output_dir: str = "batch_results"):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_query_id(self, vietnamese_query: str, sql_query: str, complexity: str) -> str:
        """
        Generate unique ID for each query based on content hash
        Format: {complexity}_{hash_prefix}
        """
        content = f"{vietnamese_query}|{sql_query}|{complexity}"
        hash_object = hashlib.md5(content.encode())
        hash_prefix = hash_object.hexdigest()[:8]
        return f"{complexity}_{hash_prefix}"
    
    def execute_sql_safely(self, sql: str, connection) -> Dict[str, Any]:
        """
        Execute SQL safely and return structured result
        """
        result = {
            'success': False,
            'data': None,
            'row_count': 0,
            'columns': [],
            'error': None,
            'execution_time_ms': 0
        }
        
        start_time = time.perf_counter()
        try:
            cursor = connection.cursor()
            cursor.execute(sql)
            
            # Get column names
            if cursor.description:
                result['columns'] = [desc[0] for desc in cursor.description]
            
            # Fetch results
            data = cursor.fetchall()
            result['data'] = [list(row) for row in data]  # Convert tuples to lists for JSON
            result['row_count'] = len(data)
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            logger.warning(f"SQL execution failed: {e}")
        
        end_time = time.perf_counter()
        result['execution_time_ms'] = (end_time - start_time) * 1000
        
        return result
    
    def process_single_query(self, query_data: Dict, complexity: str, 
                           connection, index: int) -> Dict[str, Any]:
        """
        Process a single query and return structured result
        """
        vietnamese_query = query_data['vietnamese']
        sql_query = query_data['sql']
        english_query = query_data.get('english', '')
        
        # Generate unique ID
        query_id = self.generate_query_id(vietnamese_query, sql_query, complexity)
        
        # Execute SQL and get results
        execution_result = self.execute_sql_safely(sql_query, connection)
        
        # Create structured query record
        query_record = {
            'query_id': query_id,
            'index_in_complexity': index,
            'complexity': complexity,
            'vietnamese_query': vietnamese_query,
            'english_query': english_query,
            'ground_truth_sql': sql_query,
            'execution_result': execution_result,
            'created_timestamp': time.time(),
            'created_datetime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
        
        return query_record
    
    def generate_all_ground_truth(self) -> Dict[str, Any]:
        """
        Generate ground truth for all 300 sample queries
        """
        logger.info("Starting ground truth generation for 300 sample queries...")
        
        ground_truth_data = {
            'metadata': {
                'total_queries': 0,
                'generation_timestamp': time.time(),
                'generation_datetime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                'database_path': self.db_path,
                'complexity_distribution': {}
            },
            'queries': {}
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                total_processed = 0
                
                # Process each complexity level
                for complexity, queries in SAMPLE_QUERIES_DATA.items():
                    logger.info(f"Processing {len(queries)} {complexity} queries...")
                    
                    complexity_queries = {}
                    successful_executions = 0
                    
                    for index, query_data in enumerate(queries):
                        query_record = self.process_single_query(
                            query_data, complexity, conn, index
                        )
                        
                        query_id = query_record['query_id']
                        complexity_queries[query_id] = query_record
                        
                        if query_record['execution_result']['success']:
                            successful_executions += 1
                        
                        total_processed += 1
                        
                        # Log progress every 25 queries
                        if (index + 1) % 25 == 0:
                            logger.info(f"  Processed {index + 1}/{len(queries)} {complexity} queries")
                    
                    ground_truth_data['queries'][complexity] = complexity_queries
                    ground_truth_data['metadata']['complexity_distribution'][complexity] = {
                        'total_queries': len(queries),
                        'successful_executions': successful_executions,
                        'success_rate': successful_executions / len(queries) * 100
                    }
                
                ground_truth_data['metadata']['total_queries'] = total_processed
                
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        
        logger.info(f"Ground truth generation completed. Processed {total_processed} queries.")
        return ground_truth_data
    
    def save_ground_truth(self, ground_truth_data: Dict[str, Any], 
                         filename: str = "ground_truth_results.json") -> str:
        """
        Save ground truth data to JSON file
        """
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(ground_truth_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Ground truth data saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save ground truth data: {e}")
            raise
    
    def load_ground_truth(self, filename: str = "ground_truth_results.json") -> Dict[str, Any]:
        """
        Load ground truth data from JSON file
        """
        file_path = self.output_dir / filename
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Ground truth data loaded from: {file_path}")
            return data
            
        except FileNotFoundError:
            logger.error(f"Ground truth file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load ground truth data: {e}")
            raise
    
    def get_query_by_id(self, query_id: str, 
                       ground_truth_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Retrieve a specific query by its ID
        """
        if ground_truth_data is None:
            ground_truth_data = self.load_ground_truth()
        
        # Search through all complexity levels
        for complexity, queries in ground_truth_data['queries'].items():
            if query_id in queries:
                return queries[query_id]
        
        return None
    
    def get_queries_by_complexity(self, complexity: str,
                                ground_truth_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get all queries for a specific complexity level
        """
        if ground_truth_data is None:
            ground_truth_data = self.load_ground_truth()
        
        return ground_truth_data['queries'].get(complexity, {})
    
    def generate_summary_report(self, ground_truth_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate summary report of ground truth data
        """
        if ground_truth_data is None:
            ground_truth_data = self.load_ground_truth()
        
        metadata = ground_truth_data['metadata']
        
        # Calculate overall statistics
        total_queries = metadata['total_queries']
        total_successful = sum(
            dist['successful_executions'] 
            for dist in metadata['complexity_distribution'].values()
        )
        overall_success_rate = total_successful / total_queries * 100 if total_queries > 0 else 0
        
        # Calculate execution time statistics
        all_execution_times = []
        for complexity, queries in ground_truth_data['queries'].items():
            for query_id, query_data in queries.items():
                exec_time = query_data['execution_result']['execution_time_ms']
                all_execution_times.append(exec_time)
        
        import statistics
        execution_stats = {}
        if all_execution_times:
            execution_stats = {
                'mean_execution_time_ms': statistics.mean(all_execution_times),
                'median_execution_time_ms': statistics.median(all_execution_times),
                'min_execution_time_ms': min(all_execution_times),
                'max_execution_time_ms': max(all_execution_times),
                'std_execution_time_ms': statistics.stdev(all_execution_times) if len(all_execution_times) > 1 else 0
            }
        
        summary = {
            'generation_info': {
                'timestamp': metadata['generation_datetime'],
                'database_path': metadata['database_path']
            },
            'overall_statistics': {
                'total_queries': total_queries,
                'successful_executions': total_successful,
                'overall_success_rate': overall_success_rate,
                'failed_executions': total_queries - total_successful
            },
            'complexity_breakdown': metadata['complexity_distribution'],
            'execution_time_statistics': execution_stats
        }
        
        return summary

def generate_ground_truth_main(db_path: str, output_dir: str = "batch_results"):
    """
    Main function to generate ground truth data
    """
    generator = GroundTruthGenerator(db_path, output_dir)
    
    # Generate ground truth
    ground_truth_data = generator.generate_all_ground_truth()
    
    # Save to JSON
    output_file = generator.save_ground_truth(ground_truth_data)
    
    # Generate and save summary report
    summary = generator.generate_summary_report(ground_truth_data)
    summary_file = generator.save_ground_truth(summary, "ground_truth_summary.json")
    
    # Print summary
    print("\n=== GROUND TRUTH GENERATION SUMMARY ===")
    print(f"Total Queries Processed: {summary['overall_statistics']['total_queries']}")
    print(f"Successful Executions: {summary['overall_statistics']['successful_executions']}")
    print(f"Overall Success Rate: {summary['overall_statistics']['overall_success_rate']:.2f}%")
    print(f"Ground Truth File: {output_file}")
    print(f"Summary Report: {summary_file}")
    
    for complexity, stats in summary['complexity_breakdown'].items():
        print(f"{complexity.capitalize()}: {stats['successful_executions']}/{stats['total_queries']} ({stats['success_rate']:.1f}%)")
    
    if summary['execution_time_statistics']:
        exec_stats = summary['execution_time_statistics']
        print(f"Average Execution Time: {exec_stats['mean_execution_time_ms']:.2f}ms")
    
    return ground_truth_data, summary

if __name__ == "__main__":
    # Example usage
    DB_PATH = "/path/to/your/database.db"  # Update with actual database path
    
    try:
        ground_truth_data, summary = generate_ground_truth_main(DB_PATH)
        print("Ground truth generation completed successfully!")
    except Exception as e:
        print(f"Ground truth generation failed: {e}")
