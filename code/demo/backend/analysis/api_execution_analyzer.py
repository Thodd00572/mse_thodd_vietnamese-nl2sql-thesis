"""
API Execution Analyzer for Vietnamese NL2SQL Pipeline
Collects results from Colab API calls and compares with ground truth
"""

import json
import time
import requests
import psutil
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from ground_truth_generator import GroundTruthGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIExecutionAnalyzer:
    """
    Analyze API execution results and compare with ground truth
    """
    
    def __init__(self, ground_truth_file: str, output_dir: str = "batch_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.ground_truth_generator = GroundTruthGenerator("", output_dir)
        self.ground_truth_data = self.ground_truth_generator.load_ground_truth(ground_truth_file)
        
    def call_colab_api(self, vietnamese_query: str, api_endpoint: str, 
                      api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Call Colab API and measure performance metrics
        """
        # Prepare API request
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        payload = {
            'query': vietnamese_query,
            'timestamp': time.time()
        }
        
        # Monitor system resources before API call
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / (1024**2)
        
        # Measure API call performance
        api_result = {
            'success': False,
            'generated_sql': '',
            'api_response_time_ms': 0,
            'api_error': None,
            'system_metrics': {},
            'api_metadata': {}
        }
        
        start_time = time.perf_counter()
        try:
            response = requests.post(
                api_endpoint,
                json=payload,
                headers=headers,
                timeout=30  # 30 second timeout
            )
            end_time = time.perf_counter()
            
            api_result['api_response_time_ms'] = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                response_data = response.json()
                api_result['success'] = True
                api_result['generated_sql'] = response_data.get('sql', '')
                api_result['api_metadata'] = response_data.get('metadata', {})
                
                # Extract GPU metrics if provided by API
                if 'gpu_metrics' in response_data:
                    api_result['gpu_metrics'] = response_data['gpu_metrics']
                
            else:
                api_result['api_error'] = f"HTTP {response.status_code}: {response.text}"
                
        except requests.exceptions.Timeout:
            end_time = time.perf_counter()
            api_result['api_response_time_ms'] = (end_time - start_time) * 1000
            api_result['api_error'] = "API call timeout (30s)"
            
        except Exception as e:
            end_time = time.perf_counter()
            api_result['api_response_time_ms'] = (end_time - start_time) * 1000
            api_result['api_error'] = str(e)
        
        # Measure system resource usage
        final_memory_mb = process.memory_info().rss / (1024**2)
        api_result['system_metrics'] = {
            'memory_delta_mb': max(0, final_memory_mb - initial_memory_mb),
            'peak_memory_mb': final_memory_mb
        }
        
        return api_result
    
    def execute_sql_and_compare(self, generated_sql: str, query_id: str, 
                               db_connection) -> Dict[str, Any]:
        """
        Execute generated SQL and compare with ground truth
        """
        # Get ground truth for this query
        ground_truth_query = self.ground_truth_generator.get_query_by_id(
            query_id, self.ground_truth_data
        )
        
        if not ground_truth_query:
            return {
                'comparison_success': False,
                'error': f'Ground truth not found for query ID: {query_id}'
            }
        
        # Execute generated SQL
        generated_result = self.ground_truth_generator.execute_sql_safely(
            generated_sql, db_connection
        )
        
        # Compare with ground truth
        ground_truth_result = ground_truth_query['execution_result']
        
        comparison = {
            'comparison_success': True,
            'query_id': query_id,
            'execution_accuracy': False,
            'exact_match': False,
            'generated_execution': generated_result,
            'ground_truth_execution': ground_truth_result,
            'comparison_details': {}
        }
        
        # Check execution accuracy (result set comparison)
        if generated_result['success'] and ground_truth_result['success']:
            comparison['execution_accuracy'] = self._compare_result_sets(
                generated_result['data'], ground_truth_result['data']
            )
            
            # Additional comparison details
            comparison['comparison_details'] = {
                'generated_row_count': generated_result['row_count'],
                'ground_truth_row_count': ground_truth_result['row_count'],
                'row_count_match': generated_result['row_count'] == ground_truth_result['row_count'],
                'column_count_match': len(generated_result['columns']) == len(ground_truth_result['columns']),
                'columns_match': generated_result['columns'] == ground_truth_result['columns']
            }
        
        # Check exact match (SQL string comparison)
        ground_truth_sql = ground_truth_query['ground_truth_sql']
        comparison['exact_match'] = self._normalize_sql(generated_sql) == self._normalize_sql(ground_truth_sql)
        
        return comparison
    
    def _compare_result_sets(self, results1: List, results2: List) -> bool:
        """Compare two result sets for semantic equivalence"""
        if results1 is None or results2 is None:
            return results1 == results2
        
        try:
            # Convert to sets for order-independent comparison
            set1 = set(tuple(row) for row in results1)
            set2 = set(tuple(row) for row in results2)
            return set1 == set2
        except TypeError:
            # Handle unhashable types
            return sorted(results1) == sorted(results2)
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison"""
        import re
        sql = re.sub(r'\s+', ' ', sql.strip().lower())
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        return sql.strip()
    
    def analyze_single_query(self, query_id: str, api_endpoint: str, 
                           db_connection, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a single query through the complete pipeline
        """
        # Get ground truth query
        ground_truth_query = self.ground_truth_generator.get_query_by_id(
            query_id, self.ground_truth_data
        )
        
        if not ground_truth_query:
            return {
                'error': f'Query ID {query_id} not found in ground truth data',
                'query_id': query_id
            }
        
        vietnamese_query = ground_truth_query['vietnamese_query']
        
        logger.info(f"Analyzing query {query_id}: {vietnamese_query[:50]}...")
        
        # Call API
        api_result = self.call_colab_api(vietnamese_query, api_endpoint, api_key)
        
        # Execute and compare if API call succeeded
        comparison_result = {}
        if api_result['success'] and api_result['generated_sql']:
            comparison_result = self.execute_sql_and_compare(
                api_result['generated_sql'], query_id, db_connection
            )
        
        # Create comprehensive analysis record
        analysis_record = {
            'analysis_id': f"{query_id}_{int(time.time())}",
            'query_id': query_id,
            'timestamp': time.time(),
            'datetime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'ground_truth': {
                'vietnamese_query': ground_truth_query['vietnamese_query'],
                'complexity': ground_truth_query['complexity'],
                'ground_truth_sql': ground_truth_query['ground_truth_sql'],
                'ground_truth_execution': ground_truth_query['execution_result']
            },
            'api_execution': api_result,
            'comparison': comparison_result,
            'overall_success': api_result['success'] and comparison_result.get('execution_accuracy', False)
        }
        
        return analysis_record
    
    def analyze_query_batch(self, query_ids: List[str], api_endpoint: str, 
                          db_connection, api_key: Optional[str] = None,
                          batch_name: str = None) -> Dict[str, Any]:
        """
        Analyze a batch of queries
        """
        if batch_name is None:
            batch_name = f"batch_{int(time.time())}"
        
        logger.info(f"Starting batch analysis: {batch_name} ({len(query_ids)} queries)")
        
        batch_results = {
            'batch_metadata': {
                'batch_name': batch_name,
                'total_queries': len(query_ids),
                'start_timestamp': time.time(),
                'start_datetime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                'api_endpoint': api_endpoint
            },
            'individual_results': {},
            'batch_statistics': {}
        }
        
        successful_analyses = 0
        execution_accuracies = 0
        exact_matches = 0
        api_response_times = []
        
        for i, query_id in enumerate(query_ids):
            try:
                analysis_result = self.analyze_single_query(
                    query_id, api_endpoint, db_connection, api_key
                )
                
                batch_results['individual_results'][query_id] = analysis_result
                
                # Update statistics
                if not analysis_result.get('error'):
                    successful_analyses += 1
                    
                    if analysis_result['api_execution']['success']:
                        api_response_times.append(analysis_result['api_execution']['api_response_time_ms'])
                    
                    if analysis_result.get('comparison', {}).get('execution_accuracy'):
                        execution_accuracies += 1
                    
                    if analysis_result.get('comparison', {}).get('exact_match'):
                        exact_matches += 1
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{len(query_ids)} queries")
                    
            except Exception as e:
                logger.error(f"Error analyzing query {query_id}: {e}")
                batch_results['individual_results'][query_id] = {
                    'error': str(e),
                    'query_id': query_id
                }
        
        # Calculate batch statistics
        batch_results['batch_metadata']['end_timestamp'] = time.time()
        batch_results['batch_metadata']['end_datetime'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        batch_results['batch_metadata']['duration_seconds'] = (
            batch_results['batch_metadata']['end_timestamp'] - 
            batch_results['batch_metadata']['start_timestamp']
        )
        
        batch_results['batch_statistics'] = {
            'successful_analyses': successful_analyses,
            'execution_accuracy_count': execution_accuracies,
            'exact_match_count': exact_matches,
            'execution_accuracy_rate': execution_accuracies / len(query_ids) * 100 if query_ids else 0,
            'exact_match_rate': exact_matches / len(query_ids) * 100 if query_ids else 0,
            'api_success_rate': len(api_response_times) / len(query_ids) * 100 if query_ids else 0
        }
        
        if api_response_times:
            import statistics
            batch_results['batch_statistics']['api_performance'] = {
                'mean_response_time_ms': statistics.mean(api_response_times),
                'median_response_time_ms': statistics.median(api_response_times),
                'min_response_time_ms': min(api_response_times),
                'max_response_time_ms': max(api_response_times),
                'std_response_time_ms': statistics.stdev(api_response_times) if len(api_response_times) > 1 else 0
            }
        
        logger.info(f"Batch analysis completed: {successful_analyses}/{len(query_ids)} successful")
        return batch_results
    
    def save_analysis_results(self, analysis_data: Dict[str, Any], 
                            filename: str = None) -> str:
        """
        Save analysis results to JSON file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"api_analysis_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Analysis results saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
            raise
    
    def get_all_query_ids(self, complexity: Optional[str] = None) -> List[str]:
        """
        Get all query IDs, optionally filtered by complexity
        """
        query_ids = []
        
        if complexity:
            # Get IDs for specific complexity
            complexity_queries = self.ground_truth_data['queries'].get(complexity, {})
            query_ids.extend(complexity_queries.keys())
        else:
            # Get all IDs
            for comp_queries in self.ground_truth_data['queries'].values():
                query_ids.extend(comp_queries.keys())
        
        return query_ids
