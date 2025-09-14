"""
Colab API Client for Vietnamese NL2SQL System
Handles communication with Google Colab notebooks running Pipeline 1 and Pipeline 2
"""

import requests
import logging
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

class ColabAPIClient:
    def __init__(self):
        # These URLs will be provided by the Colab notebooks
        self.pipeline1_url = None
        self.pipeline2_url = None
        self.timeout = 60  # 60 seconds timeout for API calls
        self.base_url = None  # Store the base ngrok URL
        
    def set_pipeline_urls(self, pipeline1_url: str = None, pipeline2_url: str = None):
        """Set the Colab API URLs for pipelines"""
        if pipeline1_url:
            self.pipeline1_url = pipeline1_url
            logger.info(f"Pipeline 1 URL set: {pipeline1_url}")
        
        if pipeline2_url:
            self.pipeline2_url = pipeline2_url
            logger.info(f"Pipeline 2 URL set: {pipeline2_url}")
    
    def set_base_url(self, base_url: str):
        """Set the base ngrok URL and automatically configure pipeline URLs"""
        self.base_url = base_url.rstrip('/')
        self.pipeline1_url = f"{self.base_url}/pipeline1"
        self.pipeline2_url = f"{self.base_url}/pipeline2"
        logger.info(f"Base URL set: {self.base_url}")
        logger.info(f"Pipeline 1 URL: {self.pipeline1_url}")
        logger.info(f"Pipeline 2 URL: {self.pipeline2_url}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Colab server and return detailed status"""
        status = {
            "base_url": self.base_url,
            "pipeline1_url": self.pipeline1_url,
            "pipeline2_url": self.pipeline2_url,
            "health_check": None,
            "pipeline1_status": None,
            "pipeline2_status": None,
            "connection_errors": []
        }
        
        if not self.base_url:
            status["connection_errors"].append("Base URL not configured")
            return status
        
        # Test health endpoint
        try:
            health_url = f"{self.base_url}/health"
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                status["health_check"] = response.json()
                logger.info(f" Health check passed: {health_url}")
            else:
                status["connection_errors"].append(f"Health check failed: HTTP {response.status_code}")
        except Exception as e:
            status["connection_errors"].append(f"Health check error: {str(e)}")
        
        # Test Pipeline 1
        status["pipeline1_status"] = self.check_pipeline_health("pipeline1")
        
        # Test Pipeline 2  
        status["pipeline2_status"] = self.check_pipeline_health("pipeline2")
        
        return status
    
    def check_pipeline_health(self, pipeline: str) -> bool:
        """Check if a pipeline is healthy and ready"""
        try:
            url = self.pipeline1_url if pipeline == "pipeline1" else self.pipeline2_url
            if not url:
                logger.warning(f"{pipeline} URL not configured")
                return False
            
            # Replace /pipeline1 or /pipeline2 with /health
            health_url = url.replace(f"/{pipeline}", "/health")
            
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy" and data.get("models_ready", False)
            
            return False
            
        except Exception as e:
            logger.error(f"Health check failed for {pipeline}: {e}")
            return False
    
    def call_pipeline1(self, vietnamese_query: str) -> Dict[str, Any]:
        """Call Pipeline 1: Vietnamese → PhoBERT-SQL → SQL"""
        try:
            if not self.pipeline1_url:
                raise Exception("Pipeline 1 URL not configured")
            
            logger.info(f"[ColabClient] Calling Pipeline 1: {vietnamese_query}")
            start_time = time.time()
            
            response = requests.post(
                self.pipeline1_url,
                json={"query": vietnamese_query},
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                total_time = time.time() - start_time
                
                logger.info(f"[ColabClient] Pipeline 1 success: {total_time:.2f}s")
                
                # Format response to match local pipeline format
                return {
                    'success': True,
                    'vietnamese_query': vietnamese_query,
                    'sql_query': result.get('sql_query', ''),
                    'execution_time': total_time,
                    'timings': result.get('timings', {}),
                    'model_info': result.get('model_info', {}),
                    'source': 'colab_pipeline1'
                }
            else:
                logger.error(f"[ColabClient] Pipeline 1 error: {response.status_code}")
                return {
                    'success': False,
                    'error': f"API error: {response.status_code}",
                    'vietnamese_query': vietnamese_query,
                    'sql_query': 'SELECT * FROM products LIMIT 10',
                    'execution_time': 0.0,
                    'source': 'colab_pipeline1'
                }
                
        except Exception as e:
            logger.error(f"[ColabClient] Pipeline 1 exception: {e}")
            return {
                'success': False,
                'error': str(e),
                'vietnamese_query': vietnamese_query,
                'sql_query': 'SELECT * FROM products LIMIT 10',
                'execution_time': 0.0,
                'source': 'colab_pipeline1'
            }
    
    def call_pipeline2(self, vietnamese_query: str) -> Dict[str, Any]:
        """Call Pipeline 2: Vietnamese → English → SQLCoder → SQL"""
        try:
            if not self.pipeline2_url:
                raise Exception("Pipeline 2 URL not configured")
            
            logger.info(f"[ColabClient] Calling Pipeline 2: {vietnamese_query}")
            start_time = time.time()
            
            response = requests.post(
                self.pipeline2_url,
                json={"query": vietnamese_query},
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                total_time = time.time() - start_time
                
                logger.info(f"[ColabClient] Pipeline 2 success: {total_time:.2f}s")
                
                # Format response to match local pipeline format
                return {
                    'success': True,
                    'vietnamese_query': vietnamese_query,
                    'english_query': result.get('english_query', ''),
                    'sql_query': result.get('sql_query', ''),
                    'execution_time': total_time,
                    'timings': result.get('timings', {}),
                    'model_info': result.get('model_info', {}),
                    'source': 'colab_pipeline2'
                }
            else:
                logger.error(f"[ColabClient] Pipeline 2 error: {response.status_code}")
                return {
                    'success': False,
                    'error': f"API error: {response.status_code}",
                    'vietnamese_query': vietnamese_query,
                    'english_query': '',
                    'sql_query': 'SELECT * FROM products LIMIT 10',
                    'execution_time': 0.0,
                    'source': 'colab_pipeline2'
                }
                
        except Exception as e:
            logger.error(f"[ColabClient] Pipeline 2 exception: {e}")
            return {
                'success': False,
                'error': str(e),
                'vietnamese_query': vietnamese_query,
                'english_query': '',
                'sql_query': 'SELECT * FROM products LIMIT 10',
                'execution_time': 0.0,
                'source': 'colab_pipeline2'
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of both pipelines"""
        return {
            'pipeline1': {
                'url': self.pipeline1_url,
                'configured': self.pipeline1_url is not None,
                'healthy': self.check_pipeline_health('pipeline1') if self.pipeline1_url else False
            },
            'pipeline2': {
                'url': self.pipeline2_url,
                'configured': self.pipeline2_url is not None,
                'healthy': self.check_pipeline_health('pipeline2') if self.pipeline2_url else False
            }
        }

# Global instance
colab_client = ColabAPIClient()
