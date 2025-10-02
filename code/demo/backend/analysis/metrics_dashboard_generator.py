"""
Metrics Dashboard Generator for Vietnamese NL2SQL Pipeline
Creates structured analytics JSON for metrics screen visualization
"""

import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from api_execution_analyzer import APIExecutionAnalyzer
from ground_truth_generator import GroundTruthGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsDashboardGenerator:
    """
    Generate structured analytics data for metrics dashboard
    """
    
    def __init__(self, output_dir: str = "batch_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def load_analysis_results(self, analysis_file: str) -> Dict[str, Any]:
        """Load analysis results from JSON file"""
        file_path = self.output_dir / analysis_file
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load analysis results: {e}")
            raise
    
    def generate_dashboard_metrics(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive dashboard metrics from analysis data
        """
        dashboard_data = {
            'dashboard_metadata': {
                'generated_timestamp': time.time(),
                'generated_datetime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                'source_batch': analysis_data.get('batch_metadata', {}).get('batch_name', 'unknown'),
                'total_queries_analyzed': analysis_data.get('batch_metadata', {}).get('total_queries', 0)
            },
            'overall_performance': {},
            'complexity_breakdown': {},
            'temporal_analysis': {},
            'error_analysis': {},
            'api_performance_metrics': {},
            'detailed_query_results': []
        }
        
        # Extract individual results
        individual_results = analysis_data.get('individual_results', {})
        
        # Calculate overall performance metrics
        dashboard_data['overall_performance'] = self._calculate_overall_performance(individual_results)
        
        # Breakdown by complexity
        dashboard_data['complexity_breakdown'] = self._analyze_by_complexity(individual_results)
        
        # Temporal analysis
        dashboard_data['temporal_analysis'] = self._analyze_temporal_patterns(individual_results)
        
        # Error analysis
        dashboard_data['error_analysis'] = self._analyze_errors(individual_results)
        
        # API performance metrics
        dashboard_data['api_performance_metrics'] = self._analyze_api_performance(individual_results)
        
        # Detailed query results for drill-down
        dashboard_data['detailed_query_results'] = self._prepare_detailed_results(individual_results)
        
        return dashboard_data
    
    def _calculate_overall_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance metrics"""
        total_queries = len(results)
        if total_queries == 0:
            return {'error': 'No queries to analyze'}
        
        successful_api_calls = 0
        execution_accuracies = 0
        exact_matches = 0
        api_response_times = []
        
        for query_id, result in results.items():
            if result.get('error'):
                continue
                
            # API success
            if result.get('api_execution', {}).get('success'):
                successful_api_calls += 1
                api_time = result['api_execution'].get('api_response_time_ms', 0)
                if api_time > 0:
                    api_response_times.append(api_time)
            
            # Execution accuracy
            if result.get('comparison', {}).get('execution_accuracy'):
                execution_accuracies += 1
            
            # Exact match
            if result.get('comparison', {}).get('exact_match'):
                exact_matches += 1
        
        # Calculate statistics
        import statistics
        performance_metrics = {
            'total_queries': total_queries,
            'api_success_rate': (successful_api_calls / total_queries) * 100,
            'execution_accuracy_rate': (execution_accuracies / total_queries) * 100,
            'exact_match_rate': (exact_matches / total_queries) * 100,
            'overall_success_rate': (execution_accuracies / total_queries) * 100,  # Primary metric
        }
        
        if api_response_times:
            performance_metrics['api_latency_stats'] = {
                'mean_ms': statistics.mean(api_response_times),
                'median_ms': statistics.median(api_response_times),
                'min_ms': min(api_response_times),
                'max_ms': max(api_response_times),
                'std_ms': statistics.stdev(api_response_times) if len(api_response_times) > 1 else 0
            }
        
        return performance_metrics
    
    def _analyze_by_complexity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance by query complexity"""
        complexity_stats = {}
        
        # Group results by complexity
        complexity_groups = {}
        for query_id, result in results.items():
            if result.get('error'):
                continue
                
            complexity = result.get('ground_truth', {}).get('complexity', 'unknown')
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append(result)
        
        # Calculate stats for each complexity
        for complexity, group_results in complexity_groups.items():
            total = len(group_results)
            api_successes = sum(1 for r in group_results if r.get('api_execution', {}).get('success'))
            execution_accuracies = sum(1 for r in group_results if r.get('comparison', {}).get('execution_accuracy'))
            exact_matches = sum(1 for r in group_results if r.get('comparison', {}).get('exact_match'))
            
            # API response times for this complexity
            response_times = [
                r['api_execution']['api_response_time_ms'] 
                for r in group_results 
                if r.get('api_execution', {}).get('success') and r['api_execution'].get('api_response_time_ms', 0) > 0
            ]
            
            complexity_stats[complexity] = {
                'total_queries': total,
                'api_success_rate': (api_successes / total) * 100 if total > 0 else 0,
                'execution_accuracy_rate': (execution_accuracies / total) * 100 if total > 0 else 0,
                'exact_match_rate': (exact_matches / total) * 100 if total > 0 else 0,
            }
            
            if response_times:
                import statistics
                complexity_stats[complexity]['avg_response_time_ms'] = statistics.mean(response_times)
                complexity_stats[complexity]['median_response_time_ms'] = statistics.median(response_times)
        
        return complexity_stats
    
    def _analyze_temporal_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal patterns in API performance"""
        temporal_data = []
        
        for query_id, result in results.items():
            if result.get('error') or not result.get('timestamp'):
                continue
                
            temporal_point = {
                'timestamp': result['timestamp'],
                'datetime': result.get('datetime', ''),
                'query_id': query_id,
                'complexity': result.get('ground_truth', {}).get('complexity', 'unknown'),
                'api_success': result.get('api_execution', {}).get('success', False),
                'execution_accuracy': result.get('comparison', {}).get('execution_accuracy', False),
                'response_time_ms': result.get('api_execution', {}).get('api_response_time_ms', 0)
            }
            temporal_data.append(temporal_point)
        
        # Sort by timestamp
        temporal_data.sort(key=lambda x: x['timestamp'])
        
        # Calculate moving averages and trends
        window_size = min(10, len(temporal_data))
        moving_averages = []
        
        if len(temporal_data) >= window_size:
            for i in range(window_size - 1, len(temporal_data)):
                window = temporal_data[i - window_size + 1:i + 1]
                
                avg_response_time = sum(p['response_time_ms'] for p in window if p['response_time_ms'] > 0)
                valid_responses = len([p for p in window if p['response_time_ms'] > 0])
                
                moving_avg = {
                    'timestamp': window[-1]['timestamp'],
                    'avg_response_time_ms': avg_response_time / valid_responses if valid_responses > 0 else 0,
                    'success_rate': sum(1 for p in window if p['api_success']) / len(window) * 100,
                    'accuracy_rate': sum(1 for p in window if p['execution_accuracy']) / len(window) * 100
                }
                moving_averages.append(moving_avg)
        
        return {
            'raw_temporal_data': temporal_data,
            'moving_averages': moving_averages,
            'trend_analysis': {
                'total_duration_seconds': temporal_data[-1]['timestamp'] - temporal_data[0]['timestamp'] if temporal_data else 0,
                'queries_per_minute': len(temporal_data) / ((temporal_data[-1]['timestamp'] - temporal_data[0]['timestamp']) / 60) if len(temporal_data) > 1 else 0
            }
        }
    
    def _analyze_errors(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error patterns and types"""
        error_analysis = {
            'api_errors': {},
            'sql_execution_errors': {},
            'comparison_errors': {},
            'error_by_complexity': {}
        }
        
        for query_id, result in results.items():
            complexity = result.get('ground_truth', {}).get('complexity', 'unknown')
            
            # Initialize complexity error tracking
            if complexity not in error_analysis['error_by_complexity']:
                error_analysis['error_by_complexity'][complexity] = {
                    'total_queries': 0,
                    'api_errors': 0,
                    'sql_errors': 0,
                    'comparison_errors': 0
                }
            
            error_analysis['error_by_complexity'][complexity]['total_queries'] += 1
            
            # Check for API errors
            api_error = result.get('api_execution', {}).get('api_error')
            if api_error:
                error_type = self._categorize_api_error(api_error)
                error_analysis['api_errors'][error_type] = error_analysis['api_errors'].get(error_type, 0) + 1
                error_analysis['error_by_complexity'][complexity]['api_errors'] += 1
            
            # Check for SQL execution errors
            if result.get('comparison', {}).get('generated_execution', {}).get('error'):
                sql_error = result['comparison']['generated_execution']['error']
                error_type = self._categorize_sql_error(sql_error)
                error_analysis['sql_execution_errors'][error_type] = error_analysis['sql_execution_errors'].get(error_type, 0) + 1
                error_analysis['error_by_complexity'][complexity]['sql_errors'] += 1
            
            # Check for comparison errors
            if result.get('comparison', {}).get('error'):
                comp_error = result['comparison']['error']
                error_analysis['comparison_errors'][comp_error] = error_analysis['comparison_errors'].get(comp_error, 0) + 1
                error_analysis['error_by_complexity'][complexity]['comparison_errors'] += 1
        
        return error_analysis
    
    def _categorize_api_error(self, error_message: str) -> str:
        """Categorize API error types"""
        error_lower = error_message.lower()
        
        if 'timeout' in error_lower:
            return 'timeout'
        elif 'connection' in error_lower:
            return 'connection_error'
        elif 'http 4' in error_lower:
            return 'client_error'
        elif 'http 5' in error_lower:
            return 'server_error'
        else:
            return 'other_api_error'
    
    def _categorize_sql_error(self, error_message: str) -> str:
        """Categorize SQL execution error types"""
        error_lower = error_message.lower()
        
        if 'syntax' in error_lower:
            return 'syntax_error'
        elif 'table' in error_lower and ('not' in error_lower or 'missing' in error_lower):
            return 'table_not_found'
        elif 'column' in error_lower and ('not' in error_lower or 'missing' in error_lower):
            return 'column_not_found'
        elif 'permission' in error_lower or 'access' in error_lower:
            return 'permission_error'
        else:
            return 'other_sql_error'
    
    def _analyze_api_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze detailed API performance metrics"""
        api_metrics = {
            'response_time_distribution': {},
            'gpu_metrics': {},
            'system_resource_usage': {},
            'throughput_analysis': {}
        }
        
        response_times = []
        gpu_memory_usage = []
        system_memory_deltas = []
        
        for query_id, result in results.items():
            if result.get('error'):
                continue
            
            # Collect response times
            api_exec = result.get('api_execution', {})
            if api_exec.get('success') and api_exec.get('api_response_time_ms', 0) > 0:
                response_times.append(api_exec['api_response_time_ms'])
            
            # Collect GPU metrics if available
            if 'gpu_metrics' in api_exec:
                gpu_data = api_exec['gpu_metrics']
                if 'peak_memory_gb' in gpu_data:
                    gpu_memory_usage.append(gpu_data['peak_memory_gb'])
            
            # Collect system metrics
            sys_metrics = api_exec.get('system_metrics', {})
            if 'memory_delta_mb' in sys_metrics:
                system_memory_deltas.append(sys_metrics['memory_delta_mb'])
        
        # Analyze response time distribution
        if response_times:
            import statistics
            api_metrics['response_time_distribution'] = {
                'count': len(response_times),
                'mean_ms': statistics.mean(response_times),
                'median_ms': statistics.median(response_times),
                'std_ms': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                'percentiles': {
                    'p50': statistics.median(response_times),
                    'p90': sorted(response_times)[int(0.9 * len(response_times))] if len(response_times) > 10 else max(response_times),
                    'p95': sorted(response_times)[int(0.95 * len(response_times))] if len(response_times) > 20 else max(response_times),
                    'p99': sorted(response_times)[int(0.99 * len(response_times))] if len(response_times) > 100 else max(response_times)
                }
            }
        
        # Analyze GPU metrics
        if gpu_memory_usage:
            import statistics
            api_metrics['gpu_metrics'] = {
                'count': len(gpu_memory_usage),
                'mean_memory_gb': statistics.mean(gpu_memory_usage),
                'max_memory_gb': max(gpu_memory_usage),
                'min_memory_gb': min(gpu_memory_usage)
            }
        
        # Analyze system resource usage
        if system_memory_deltas:
            import statistics
            api_metrics['system_resource_usage'] = {
                'mean_memory_delta_mb': statistics.mean(system_memory_deltas),
                'max_memory_delta_mb': max(system_memory_deltas),
                'total_memory_used_mb': sum(system_memory_deltas)
            }
        
        return api_metrics
    
    def _prepare_detailed_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare detailed results for drill-down analysis"""
        detailed_results = []
        
        for query_id, result in results.items():
            if result.get('error'):
                detailed_item = {
                    'query_id': query_id,
                    'status': 'error',
                    'error_message': result['error']
                }
            else:
                detailed_item = {
                    'query_id': query_id,
                    'timestamp': result.get('timestamp', 0),
                    'datetime': result.get('datetime', ''),
                    'vietnamese_query': result.get('ground_truth', {}).get('vietnamese_query', ''),
                    'complexity': result.get('ground_truth', {}).get('complexity', 'unknown'),
                    'ground_truth_sql': result.get('ground_truth', {}).get('ground_truth_sql', ''),
                    'generated_sql': result.get('api_execution', {}).get('generated_sql', ''),
                    'api_success': result.get('api_execution', {}).get('success', False),
                    'api_response_time_ms': result.get('api_execution', {}).get('api_response_time_ms', 0),
                    'execution_accuracy': result.get('comparison', {}).get('execution_accuracy', False),
                    'exact_match': result.get('comparison', {}).get('exact_match', False),
                    'overall_success': result.get('overall_success', False),
                    'status': 'success' if result.get('overall_success') else 'failed'
                }
                
                # Add error information if present
                if result.get('api_execution', {}).get('api_error'):
                    detailed_item['api_error'] = result['api_execution']['api_error']
                
                if result.get('comparison', {}).get('generated_execution', {}).get('error'):
                    detailed_item['sql_error'] = result['comparison']['generated_execution']['error']
            
            detailed_results.append(detailed_item)
        
        # Sort by timestamp
        detailed_results.sort(key=lambda x: x.get('timestamp', 0))
        
        return detailed_results
    
    def save_dashboard_data(self, dashboard_data: Dict[str, Any], 
                          filename: str = None) -> str:
        """Save dashboard data to JSON file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"metrics_dashboard_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dashboard data saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save dashboard data: {e}")
            raise
    
    def generate_summary_report(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate human-readable summary report"""
        overall = dashboard_data.get('overall_performance', {})
        complexity = dashboard_data.get('complexity_breakdown', {})
        
        report = []
        report.append("=== VIETNAMESE NL2SQL PIPELINE METRICS SUMMARY ===")
        report.append(f"Generated: {dashboard_data.get('dashboard_metadata', {}).get('generated_datetime', 'Unknown')}")
        report.append(f"Total Queries: {overall.get('total_queries', 0)}")
        report.append("")
        
        report.append("OVERALL PERFORMANCE:")
        report.append(f"  API Success Rate: {overall.get('api_success_rate', 0):.2f}%")
        report.append(f"  Execution Accuracy: {overall.get('execution_accuracy_rate', 0):.2f}%")
        report.append(f"  Exact Match Rate: {overall.get('exact_match_rate', 0):.2f}%")
        
        if 'api_latency_stats' in overall:
            latency = overall['api_latency_stats']
            report.append(f"  Average Latency: {latency.get('mean_ms', 0):.2f}ms")
            report.append(f"  Median Latency: {latency.get('median_ms', 0):.2f}ms")
        
        report.append("")
        report.append("PERFORMANCE BY COMPLEXITY:")
        for comp, stats in complexity.items():
            report.append(f"  {comp.upper()}:")
            report.append(f"    Queries: {stats.get('total_queries', 0)}")
            report.append(f"    Execution Accuracy: {stats.get('execution_accuracy_rate', 0):.2f}%")
            report.append(f"    Avg Response Time: {stats.get('avg_response_time_ms', 0):.2f}ms")
        
        return "\n".join(report)

def generate_metrics_dashboard(analysis_file: str, output_dir: str = "batch_results") -> Dict[str, Any]:
    """
    Main function to generate metrics dashboard from analysis results
    """
    generator = MetricsDashboardGenerator(output_dir)
    
    # Load analysis data
    analysis_data = generator.load_analysis_results(analysis_file)
    
    # Generate dashboard metrics
    dashboard_data = generator.generate_dashboard_metrics(analysis_data)
    
    # Save dashboard data
    dashboard_file = generator.save_dashboard_data(dashboard_data)
    
    # Generate and print summary
    summary_report = generator.generate_summary_report(dashboard_data)
    print(summary_report)
    
    # Save summary report
    summary_file = generator.save_dashboard_data(
        {'summary_report': summary_report}, 
        f"metrics_summary_{int(time.time())}.json"
    )
    
    logger.info(f"Dashboard generated: {dashboard_file}")
    logger.info(f"Summary saved: {summary_file}")
    
    return dashboard_data

if __name__ == "__main__":
    # Example usage
    ANALYSIS_FILE = "api_analysis_results_1234567890.json"  # Update with actual file
    
    try:
        dashboard_data = generate_metrics_dashboard(ANALYSIS_FILE)
        print("Metrics dashboard generated successfully!")
    except Exception as e:
        print(f"Dashboard generation failed: {e}")
