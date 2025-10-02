"""
Improved Evaluation Metrics for Vietnamese NL2SQL Pipeline
Focus on practical metrics that matter for real-world deployment
"""

import time
import statistics
import sqlite3
import psutil
from typing import Dict, List, Tuple, Any, Optional
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedEvaluationMetrics:
    """
    Improved evaluation metrics focusing on execution accuracy as primary metric
    """
    
    def __init__(self, db_connection_string: str):
        self.db_connection_string = db_connection_string
        
    def execute_sql_safely(self, sql: str, connection) -> Optional[List[Tuple]]:
        """
        Execute SQL safely and return results
        Returns None if execution fails
        """
        try:
            cursor = connection.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
        except Exception as e:
            logger.warning(f"SQL execution failed: {e}")
            return None
    
    def compare_result_sets(self, results1: Optional[List[Tuple]], 
                           results2: Optional[List[Tuple]]) -> bool:
        """
        Compare two SQL result sets for semantic equivalence
        Handles different row orders and null values
        """
        if results1 is None or results2 is None:
            return results1 == results2
        
        # Convert to sets for order-independent comparison
        try:
            set1 = set(tuple(row) for row in results1)
            set2 = set(tuple(row) for row in results2)
            return set1 == set2
        except TypeError:
            # Handle unhashable types (like lists in results)
            return sorted(results1) == sorted(results2)
    
    def measure_execution_accuracy(self, generated_sql: str, 
                                 ground_truth_sql: str) -> Dict[str, Any]:
        """
        Primary metric: Execute both SQLs and compare their result sets
        Returns detailed execution accuracy information
        """
        result = {
            'execution_accuracy': False,
            'generated_sql_valid': False,
            'ground_truth_sql_valid': False,
            'generated_results': None,
            'ground_truth_results': None,
            'error_message': None
        }
        
        try:
            with sqlite3.connect(self.db_connection_string) as conn:
                # Execute generated SQL
                generated_results = self.execute_sql_safely(generated_sql, conn)
                result['generated_sql_valid'] = generated_results is not None
                result['generated_results'] = generated_results
                
                # Execute ground truth SQL
                ground_truth_results = self.execute_sql_safely(ground_truth_sql, conn)
                result['ground_truth_sql_valid'] = ground_truth_results is not None
                result['ground_truth_results'] = ground_truth_results
                
                # Compare results if both executed successfully
                if generated_results is not None and ground_truth_results is not None:
                    result['execution_accuracy'] = self.compare_result_sets(
                        generated_results, ground_truth_results
                    )
                
        except Exception as e:
            result['error_message'] = str(e)
            logger.error(f"Database connection error: {e}")
        
        return result
    
    def measure_exact_match(self, generated_sql: str, ground_truth_sql: str) -> Dict[str, Any]:
        """
        Secondary metric: Normalized string comparison for exact match
        Less important than execution accuracy
        """
        def normalize_sql(sql: str) -> str:
            import re
            # Remove extra whitespace, convert to lowercase, remove comments
            sql = re.sub(r'\s+', ' ', sql.strip().lower())
            sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
            sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
            return sql.strip()
        
        normalized_generated = normalize_sql(generated_sql)
        normalized_ground_truth = normalize_sql(ground_truth_sql)
        
        return {
            'exact_match': normalized_generated == normalized_ground_truth,
            'normalized_generated': normalized_generated,
            'normalized_ground_truth': normalized_ground_truth
        }
    
    def measure_latency_robust(self, query: str, model_api_call, 
                             num_runs: int = 3) -> Dict[str, float]:
        """
        Measure latency with multiple runs for statistical accuracy
        Returns comprehensive latency statistics
        """
        latencies = []
        
        # Warm-up call (exclude from measurement)
        try:
            _ = model_api_call(query)
        except Exception as e:
            logger.warning(f"Warm-up call failed: {e}")
        
        # Actual measurements
        for run in range(num_runs):
            try:
                start_time = time.perf_counter()
                _ = model_api_call(query)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
            except Exception as e:
                logger.warning(f"Latency measurement run {run+1} failed: {e}")
                # Still record a high latency for failed calls
                latencies.append(float('inf'))
        
        # Filter out infinite values for statistics
        valid_latencies = [l for l in latencies if l != float('inf')]
        
        if not valid_latencies:
            return {
                'mean_latency_ms': float('inf'),
                'median_latency_ms': float('inf'),
                'std_latency_ms': 0,
                'min_latency_ms': float('inf'),
                'max_latency_ms': float('inf'),
                'success_rate': 0.0,
                'raw_measurements': latencies
            }
        
        return {
            'mean_latency_ms': statistics.mean(valid_latencies),
            'median_latency_ms': statistics.median(valid_latencies),
            'std_latency_ms': statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else 0,
            'min_latency_ms': min(valid_latencies),
            'max_latency_ms': max(valid_latencies),
            'success_rate': len(valid_latencies) / len(latencies),
            'raw_measurements': latencies
        }
    
    def measure_resource_consumption(self, query: str, model_api_call) -> Dict[str, float]:
        """
        Measure resource consumption during model inference
        Focuses on practical metrics for deployment planning
        """
        # Get initial system state
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / (1024**2)
        
        # Try to get GPU info if available
        gpu_info = {}
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                initial_gpu_memory = gpu.memoryUsed
                gpu_info['gpu_available'] = True
            else:
                gpu_info['gpu_available'] = False
        except ImportError:
            gpu_info['gpu_available'] = False
        
        # Try PyTorch GPU monitoring if available
        torch_available = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch_available = True
        except ImportError:
            pass
        
        # Execute model call and measure
        start_time = time.perf_counter()
        try:
            result = model_api_call(query)
            execution_successful = True
        except Exception as e:
            logger.warning(f"Model API call failed during resource measurement: {e}")
            execution_successful = False
        end_time = time.perf_counter()
        
        # Calculate final measurements
        final_memory_mb = process.memory_info().rss / (1024**2)
        memory_delta_mb = max(0, final_memory_mb - initial_memory_mb)
        
        resource_metrics = {
            'execution_successful': execution_successful,
            'total_time_ms': (end_time - start_time) * 1000,
            'memory_delta_mb': memory_delta_mb,
            'peak_memory_mb': final_memory_mb
        }
        
        # Add GPU metrics if available
        if torch_available:
            resource_metrics.update({
                'peak_gpu_memory_gb': torch.cuda.max_memory_allocated() / (1024**3),
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3)
            })
        
        if gpu_info.get('gpu_available'):
            try:
                gpu = GPUtil.getGPUs()[0]
                resource_metrics.update({
                    'gpu_utilization_percent': gpu.load * 100,
                    'gpu_memory_used_gb': gpu.memoryUsed / 1024
                })
            except:
                pass
        
        return resource_metrics
    
    def evaluate_single_query(self, vietnamese_query: str, generated_sql: str, 
                            ground_truth_sql: str, model_api_call) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single query
        Returns all metrics for the query
        """
        logger.info(f"Evaluating query: {vietnamese_query[:50]}...")
        
        # Primary metric: Execution Accuracy
        execution_result = self.measure_execution_accuracy(generated_sql, ground_truth_sql)
        
        # Secondary metrics
        exact_match_result = self.measure_exact_match(generated_sql, ground_truth_sql)
        latency_result = self.measure_latency_robust(vietnamese_query, model_api_call)
        resource_result = self.measure_resource_consumption(vietnamese_query, model_api_call)
        
        return {
            'query': vietnamese_query,
            'generated_sql': generated_sql,
            'ground_truth_sql': ground_truth_sql,
            'execution_accuracy': execution_result,
            'exact_match': exact_match_result,
            'latency': latency_result,
            'resources': resource_result,
            'timestamp': time.time()
        }
    
    def evaluate_query_set(self, query_data: List[Dict], model_api_call) -> Dict[str, Any]:
        """
        Evaluate a complete set of queries and return aggregated results
        """
        results = []
        
        for query_item in query_data:
            vietnamese_query = query_item['vietnamese']
            ground_truth_sql = query_item['sql']
            
            # Generate SQL using model
            try:
                generated_sql = model_api_call(vietnamese_query)
            except Exception as e:
                logger.error(f"Model API call failed for query '{vietnamese_query}': {e}")
                generated_sql = ""
            
            # Evaluate this query
            result = self.evaluate_single_query(
                vietnamese_query, generated_sql, ground_truth_sql, model_api_call
            )
            results.append(result)
        
        # Aggregate results
        return self.aggregate_results(results)
    
    def aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate individual query results into overall metrics
        """
        if not results:
            return {'error': 'No results to aggregate'}
        
        # Execution Accuracy aggregation
        execution_accuracies = [r['execution_accuracy']['execution_accuracy'] for r in results]
        execution_accuracy_rate = sum(execution_accuracies) / len(execution_accuracies) * 100
        
        # Exact Match aggregation
        exact_matches = [r['exact_match']['exact_match'] for r in results]
        exact_match_rate = sum(exact_matches) / len(exact_matches) * 100
        
        # Latency aggregation
        valid_latencies = [r['latency']['mean_latency_ms'] for r in results 
                          if r['latency']['mean_latency_ms'] != float('inf')]
        
        latency_stats = {}
        if valid_latencies:
            latency_stats = {
                'overall_mean_latency_ms': statistics.mean(valid_latencies),
                'overall_median_latency_ms': statistics.median(valid_latencies),
                'overall_std_latency_ms': statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else 0,
                'latency_success_rate': len(valid_latencies) / len(results) * 100
            }
        
        # Resource aggregation
        successful_resources = [r['resources'] for r in results 
                              if r['resources']['execution_successful']]
        
        resource_stats = {}
        if successful_resources:
            resource_stats = {
                'avg_memory_delta_mb': statistics.mean([r['memory_delta_mb'] for r in successful_resources]),
                'avg_total_time_ms': statistics.mean([r['total_time_ms'] for r in successful_resources]),
                'resource_success_rate': len(successful_resources) / len(results) * 100
            }
            
            # GPU stats if available
            gpu_memories = [r.get('peak_gpu_memory_gb', 0) for r in successful_resources]
            if any(gpu_memories):
                resource_stats['avg_peak_gpu_memory_gb'] = statistics.mean([m for m in gpu_memories if m > 0])
        
        return {
            'total_queries': len(results),
            'execution_accuracy_rate': execution_accuracy_rate,
            'exact_match_rate': exact_match_rate,
            'latency_statistics': latency_stats,
            'resource_statistics': resource_stats,
            'detailed_results': results,
            'evaluation_timestamp': time.time()
        }

# Example usage function
def run_improved_evaluation(db_path: str, query_data: List[Dict], model_api_call):
    """
    Run the improved evaluation framework
    """
    evaluator = ImprovedEvaluationMetrics(db_path)
    
    logger.info(f"Starting evaluation of {len(query_data)} queries...")
    logger.info("Primary metric: Execution Accuracy (EX)")
    logger.info("Secondary metrics: Exact Match (EM), Latency, Resource Consumption")
    
    results = evaluator.evaluate_query_set(query_data, model_api_call)
    
    # Print summary
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Total Queries: {results['total_queries']}")
    print(f"Execution Accuracy: {results['execution_accuracy_rate']:.2f}%")
    print(f"Exact Match: {results['exact_match_rate']:.2f}%")
    
    if results.get('latency_statistics'):
        lat_stats = results['latency_statistics']
        print(f"Average Latency: {lat_stats.get('overall_mean_latency_ms', 0):.2f}ms")
        print(f"Latency Success Rate: {lat_stats.get('latency_success_rate', 0):.2f}%")
    
    if results.get('resource_statistics'):
        res_stats = results['resource_statistics']
        print(f"Average Memory Delta: {res_stats.get('avg_memory_delta_mb', 0):.2f}MB")
        if 'avg_peak_gpu_memory_gb' in res_stats:
            print(f"Average Peak GPU Memory: {res_stats['avg_peak_gpu_memory_gb']:.2f}GB")
    
    return results
