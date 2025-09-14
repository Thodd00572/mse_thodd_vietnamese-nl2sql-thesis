import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import time
import httpx
import json
from typing import Dict, Any, List
from .model_config import model_loader

logger = logging.getLogger(__name__)

class VietnameseSQLPipeline:
    """Pipeline 1: Vietnamese → PhoBERT-SQL → SQL execution"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = model_loader.device
        self.colab_url = None
        
    def load_model(self):
        """Load PhoBERT-SQL model"""
        self.model = model_loader.get_model('phobert_sql')
        self.tokenizer = model_loader.get_tokenizer('phobert_sql')
        return self.model is not None
    
    def set_colab_url(self, url: str):
        """Set the Colab endpoint URL"""
        self.colab_url = url.rstrip('/') if url else None
        logger.info(f"Pipeline 1 Colab URL set to: {self.colab_url}")
    
    async def _call_colab_api(self, vietnamese_query: str) -> Dict[str, Any]:
        """Call the Colab API for Pipeline 1 - single query"""
        if not self.colab_url:
            raise Exception("Colab URL not configured for Pipeline 1")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.colab_url}/pipeline1",
                    json={"query": vietnamese_query},
                    headers={"ngrok-skip-browser-warning": "true"}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Colab API call failed: {e}")
            raise
    
    async def _call_colab_batch_api(self, queries: List[str]) -> Dict[str, Any]:
        """Call the Colab API for Pipeline 1 - batch queries"""
        if not self.colab_url:
            raise Exception("Colab URL not configured for Pipeline 1")
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:  # Longer timeout for batch
                response = await client.post(
                    f"{self.colab_url}/batch/pipeline1",
                    json={"queries": queries, "batch_size": len(queries)},
                    headers={"ngrok-skip-browser-warning": "true"}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Colab batch API call failed: {e}")
            raise
    
    async def vietnamese_to_sql(self, vietnamese_query: str, schema_context: str = "") -> Dict[str, Any]:
        """Convert Vietnamese query to SQL using Colab API only - no fallback"""
        start_time = time.time()
        
        # Check if Colab URL is configured
        if not self.colab_url:
            return {
                'sql_query': '',
                'execution_time': 0.0,
                'success': False,
                'error': 'Colab server not configured. Please connect to Colab server first.',
                'requires_colab': True,
                'source': 'no_colab'
            }
        
        try:
            # Call Colab API - no local fallback
            colab_result = await self._call_colab_api(vietnamese_query)
            execution_time = time.time() - start_time
            
            return {
                'sql_query': colab_result.get('sql_query', ''),
                'execution_time': colab_result.get('execution_time', execution_time),
                'success': colab_result.get('success', False),
                'error': colab_result.get('error'),
                'source': 'colab_api',
                'metrics': colab_result.get('metrics', {}),
                'processing_time': colab_result.get('execution_time', execution_time)
            }
            
        except Exception as e:
            logger.error(f"Colab API call failed: {e}")
            execution_time = time.time() - start_time
            return {
                'sql_query': '',
                'execution_time': execution_time,
                'success': False,
                'error': f'Colab connection failed: {str(e)}. Please check Colab server status.',
                'requires_colab': True,
                'source': 'colab_error'
            }
    
    async def vietnamese_to_sql_batch(self, queries: List[str], batch_size: int = 10) -> List[Dict[str, Any]]:
        """Convert multiple Vietnamese queries to SQL using Colab batch API for optimization"""
        if not self.colab_url:
            return [{
                'sql_query': '',
                'execution_time': 0.0,
                'success': False,
                'error': 'Colab server not configured. Please connect to Colab server first.',
                'requires_colab': True,
                'source': 'no_colab'
            } for _ in queries]
        
        results = []
        
        # Process queries in batches
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            start_time = time.time()
            
            try:
                logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} queries")
                batch_result = await self._call_colab_batch_api(batch)
                
                # Extract individual results from batch response
                batch_results = batch_result.get('results', [])
                
                for j, result in enumerate(batch_results):
                    results.append({
                        'sql_query': result.get('sql_query', ''),
                        'execution_time': result.get('execution_time', 0.0),
                        'success': result.get('success', False),
                        'error': result.get('error'),
                        'source': 'colab_batch_api',
                        'batch_index': i + j,
                        'batch_total_time': batch_result.get('total_execution_time', 0.0)
                    })
                    
            except Exception as e:
                logger.error(f"Batch API call failed for batch {i//batch_size + 1}: {e}")
                execution_time = time.time() - start_time
                
                # Add error results for all queries in failed batch
                for j in range(len(batch)):
                    results.append({
                        'sql_query': '',
                        'execution_time': execution_time / len(batch),
                        'success': False,
                        'error': f'Batch processing failed: {str(e)}',
                        'requires_colab': True,
                        'source': 'colab_batch_error',
                        'batch_index': i + j
                    })
        
        return results
    
    def _enhanced_vietnamese_to_sql(self, query: str) -> str:
        """Rule-based Vietnamese to SQL translation with comprehensive patterns"""
        query_lower = query.lower()
        
        # Laptop/Computer queries
        if "laptop" in query_lower or "máy tính" in query_lower:
            return "SELECT * FROM products WHERE name LIKE '%laptop%' OR name LIKE '%máy tính%' LIMIT 10"
        
        # Bag/Purse queries
        elif "túi" in query_lower or "balo" in query_lower:
            if "nữ" in query_lower:
                return "SELECT * FROM products WHERE (name LIKE '%túi%' OR name LIKE '%balo%') AND (name LIKE '%nữ%' OR category LIKE '%women%' OR category LIKE '%nữ%') LIMIT 10"
            elif "nam" in query_lower:
                return "SELECT * FROM products WHERE (name LIKE '%túi%' OR name LIKE '%balo%') AND (name LIKE '%nam%' OR category LIKE '%men%' OR category LIKE '%nam%') LIMIT 10"
            else:
                return "SELECT * FROM products WHERE name LIKE '%túi%' OR name LIKE '%balo%' LIMIT 10"
        
        # Shoe queries
        elif "giày" in query_lower or "dép" in query_lower:
            if "nữ" in query_lower:
                return "SELECT * FROM products WHERE (name LIKE '%giày%' OR name LIKE '%dép%') AND (name LIKE '%nữ%' OR category LIKE '%women%' OR category LIKE '%nữ%') LIMIT 10"
            elif "nam" in query_lower:
                return "SELECT * FROM products WHERE (name LIKE '%giày%' OR name LIKE '%dép%') AND (name LIKE '%nam%' OR category LIKE '%men%' OR category LIKE '%nam%') LIMIT 10"
            else:
                return "SELECT * FROM products WHERE name LIKE '%giày%' OR name LIKE '%dép%' LIMIT 10"
        
        # Clothing queries
        elif "áo" in query_lower:
            if "thun" in query_lower:
                return "SELECT * FROM products WHERE name LIKE '%áo thun%' OR name LIKE '%áo%' AND name LIKE '%thun%' LIMIT 10"
            elif "khoác" in query_lower:
                return "SELECT * FROM products WHERE name LIKE '%áo khoác%' OR name LIKE '%khoác%' LIMIT 10"
            else:
                return "SELECT * FROM products WHERE name LIKE '%áo%' LIMIT 10"
        
        # Price-based queries
        elif "giá" in query_lower and ("dưới" in query_lower or "nhỏ hơn" in query_lower):
            if "500k" in query_lower or "500" in query_lower:
                return "SELECT * FROM products WHERE price < 500000 ORDER BY price ASC LIMIT 10"
            elif "1 triệu" in query_lower or "1000k" in query_lower:
                return "SELECT * FROM products WHERE price < 1000000 ORDER BY price ASC LIMIT 10"
            else:
                return "SELECT * FROM products WHERE price < 1000000 ORDER BY price ASC LIMIT 10"
        
        elif "giá" in query_lower and ("trên" in query_lower or "lớn hơn" in query_lower):
            return "SELECT * FROM products WHERE price > 1000000 ORDER BY price DESC LIMIT 10"
        
        # Brand queries
        elif "nike" in query_lower:
            return "SELECT * FROM products WHERE brand LIKE '%Nike%' OR name LIKE '%Nike%' LIMIT 10"
        elif "adidas" in query_lower:
            return "SELECT * FROM products WHERE brand LIKE '%Adidas%' OR name LIKE '%Adidas%' LIMIT 10"
        
        # Search queries
        elif "tìm" in query_lower:
            search_term = query_lower.replace("tìm", "").strip()
            return f"SELECT * FROM products WHERE name LIKE '%{search_term}%' LIMIT 10"
        
        # Default fallback
        else:
            return "SELECT * FROM products LIMIT 10"

class VietnameseEnglishSQLPipeline:
    """Pipeline 2: Vietnamese → PhoBERT (Vietnamese-to-English) → SQLCoder (English-to-SQL) → SQL execution"""
    
    def __init__(self):
        # No local models - Colab API only
        self.device = model_loader.device
        self.colab_url = None
        
    def set_colab_url(self, url: str):
        """Set the Colab endpoint URL"""
        self.colab_url = url.rstrip('/') if url else None
        logger.info(f"Pipeline 2 Colab URL set to: {self.colab_url}")
    
    async def _call_colab_api(self, vietnamese_query: str) -> Dict[str, Any]:
        """Call the Colab API for Pipeline 2 - single query"""
        if not self.colab_url:
            raise Exception("Colab URL not configured for Pipeline 2")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.colab_url}/pipeline2",
                    json={"query": vietnamese_query},
                    headers={"ngrok-skip-browser-warning": "true"}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Colab API call failed: {e}")
            raise
    
    async def _call_colab_batch_api(self, queries: List[str]) -> Dict[str, Any]:
        """Call the Colab API for Pipeline 2 - batch queries"""
        if not self.colab_url:
            raise Exception("Colab URL not configured for Pipeline 2")
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:  # Longer timeout for batch
                response = await client.post(
                    f"{self.colab_url}/batch/pipeline2",
                    json={"queries": queries, "batch_size": len(queries)},
                    headers={"ngrok-skip-browser-warning": "true"}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Colab batch API call failed: {e}")
            raise
    
    def load_models(self):
        """Skip local model loading - use Colab API only"""
        logger.info("[Pipeline2] Skipping local model loading - using Colab API only")
        return True  # Always return True since we don't load models locally
    
    def vietnamese_to_english(self, vietnamese_query: str) -> Dict[str, Any]:
        """Translate Vietnamese to English - Colab API only, no local processing"""
        logger.info(f"[Pipeline2-VN2EN] Vietnamese to English translation (Colab API only)")
        logger.info(f"[Pipeline2-VN2EN] Input query: '{vietnamese_query}'")
        start_time = time.time()
        
        try:
            # No local model processing - use simple rule-based fallback only
            logger.info("[Pipeline2-VN2EN] Using rule-based translation fallback")
            
            # Simple rule-based Vietnamese to English translation
            english_query = self._rule_based_vn_to_en_translation(vietnamese_query)
            
            execution_time = time.time() - start_time
            logger.info(f"[Pipeline2-VN2EN] Translation completed in {execution_time:.3f}s")
            
            return {
                "english_query": english_query,
                "execution_time": execution_time,
                "success": True,
                "error": None,
                "translation_method": "Rule-based-fallback"
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[Pipeline2-VN2EN] Translation failed: {str(e)}")
            return {
                "english_query": vietnamese_query,  # Fallback to original query
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "translation_method": "Failed"
            }
  
    def english_to_sql(self, english_query: str, schema_context: str = "") -> Dict[str, Any]:
        """Convert English query to SQL - Colab API only, no local processing"""
        logger.info(f"[Pipeline2-EN2SQL] Starting English to SQL conversion (Colab API only)")
        logger.info(f"[Pipeline2-EN2SQL] Input query: '{english_query}'")
        start_time = time.time()
        
        try:
            # No local model processing - use rule-based fallback only
            logger.info("[Pipeline2-EN2SQL] Using rule-based SQL generation fallback")
            
            sql_query = self._rule_based_en_to_sql_translation(english_query)
            
            execution_time = time.time() - start_time
            logger.info(f"[Pipeline2-EN2SQL] SQL generation completed in {execution_time:.3f}s")
            
            return {
                "sql_query": sql_query,
                "execution_time": execution_time,
                "success": True,
                "error": None,
                "translation_method": "Rule-based-fallback"
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[Pipeline2-EN2SQL] SQL generation failed: {str(e)}")
            return {
                "sql_query": "SELECT * FROM products LIMIT 10",  # Safe fallback
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "translation_method": "Failed"
            }
    
    async def full_pipeline_batch(self, queries: List[str], batch_size: int = 10) -> List[Dict[str, Any]]:
        """Execute multiple Vietnamese -> English -> SQL pipelines using Colab batch API for optimization"""
        if not self.colab_url:
            return [{
                "vietnamese_query": query,
                "english_query": "",
                "sql_query": "",
                "execution_time": 0.0,
                "vn_en_time": 0.0,
                "en_sql_time": 0.0,
                "success": False,
                "error": "Colab server not configured. Please connect to Colab server first.",
                "requires_colab": True,
                "source": "no_colab"
            } for query in queries]
        
        results = []
        
        # Process queries in batches
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            start_time = time.time()
            
            try:
                logger.info(f"[Pipeline2] Processing batch {i//batch_size + 1} with {len(batch)} queries")
                batch_result = await self._call_colab_batch_api(batch)
                
                # Extract individual results from batch response
                batch_results = batch_result.get('results', [])
                
                for j, result in enumerate(batch_results):
                    results.append({
                        "vietnamese_query": batch[j],
                        "english_query": result.get('english_translation', ''),
                        "sql_query": result.get('sql_query', ''),
                        "execution_time": result.get('execution_time', 0.0),
                        "vn_en_time": result.get('translation_time', 0.0),
                        "en_sql_time": result.get('sql_generation_time', 0.0),
                        "success": result.get('success', False),
                        "error": result.get('error'),
                        "source": "colab_batch_api",
                        "batch_index": i + j,
                        "batch_total_time": batch_result.get('total_execution_time', 0.0)
                    })
                    
            except Exception as e:
                logger.error(f"[Pipeline2] Batch API call failed for batch {i//batch_size + 1}: {e}")
                execution_time = time.time() - start_time
                
                # Add error results for all queries in failed batch
                for j in range(len(batch)):
                    results.append({
                        "vietnamese_query": batch[j],
                        "english_query": "",
                        "sql_query": "",
                        "execution_time": execution_time / len(batch),
                        "vn_en_time": 0.0,
                        "en_sql_time": 0.0,
                        "success": False,
                        "error": f"Batch processing failed: {str(e)}",
                        "requires_colab": True,
                        "source": "colab_batch_error",
                        "batch_index": i + j
                    })
        
        return results
    
    async def full_pipeline(self, vietnamese_query: str) -> Dict[str, Any]:
        """Execute Vietnamese -> English -> SQL pipeline using Colab API only - no fallback"""
        logger.info(f"[Pipeline2] Starting Vietnamese NL2SQL processing via Colab")
        logger.info(f"[Pipeline2] Input Vietnamese query: '{vietnamese_query}'")
        start_time = time.time()
        
        # Check if Colab URL is configured
        if not self.colab_url:
            return {
                "vietnamese_query": vietnamese_query,
                "english_query": "",
                "sql_query": "",
                "execution_time": 0.0,
                "vn_en_time": 0.0,
                "en_sql_time": 0.0,
                "success": False,
                "error": "Colab server not configured. Please connect to Colab server first.",
                "requires_colab": True,
                "source": "no_colab"
            }
        
        try:
            # Call Colab API for Pipeline 2 - no local processing
            colab_result = await self._call_colab_api(vietnamese_query)
            total_time = time.time() - start_time
            
            logger.info(f"[Pipeline2] Colab processing completed in {total_time:.3f}s")
            
            return {
                "vietnamese_query": vietnamese_query,
                "english_query": colab_result.get('english_translation', ''),
                "sql_query": colab_result.get('sql_query', ''),
                "success": colab_result.get('success', False),
                "error": colab_result.get('error'),
                "total_time": colab_result.get('execution_time', total_time),
                "translation_time": colab_result.get('translation_time', 0),
                "sql_generation_time": colab_result.get('sql_generation_time', 0),
                "source": "colab_api",
                "metrics": colab_result.get('metrics', {})
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"[Pipeline2] Colab API error: {str(e)}")
            return {
                "vietnamese_query": vietnamese_query,
                "english_query": "",
                "sql_query": "",
                "execution_time": total_time,
                "vn_en_time": 0.0,
                "en_sql_time": 0.0,
                "success": False,
                "error": f"Colab connection failed: {str(e)}. Please check Colab server status.",
                "requires_colab": True,
                "source": "colab_error"
            }
    
    def _rule_based_vn_to_en_translation(self, query: str) -> str:
        """Enhanced rule-based Vietnamese to English translation"""
        translations = {
            # Bags and accessories
            "túi xách": "handbag",
            "túi": "bag",
            "balo": "backpack",
            "ví": "wallet",
            
            # Clothing
            "áo thun": "t-shirt",
            "áo": "shirt",
            "áo khoác": "jacket",
            "váy": "dress",
            "quần": "pants",
            
            # Shoes
            "giày": "shoes",
            "dép": "sandals",
            "giày thể thao": "sneakers",
            "giày cao gót": "high heels",
            
            # Gender
            "nữ": "women",
            "nam": "men",
            
            # Colors
            "đen": "black",
            "trắng": "white",
            "đỏ": "red",
            "xanh": "blue",
            "nâu": "brown",
            
            # Price terms
            "giá": "price",
            "rẻ": "cheap",
            "đắt": "expensive",
            "dưới": "under",
            "trên": "over",
            
            # Actions
            "tìm": "find",
            "tìm kiếm": "search",
            
            # Technology
            "laptop": "laptop",
            "máy tính": "computer",
            "máy tính xách tay": "laptop",
            
            # Brands
            "Apple": "Apple",
            "MacBook": "MacBook",
            "Dell": "Dell",
            "HP": "HP",
            "Lenovo": "Lenovo",
            "Nike": "Nike",
            "Adidas": "Adidas"
        }
        
        english_query = query
        for vn_word, en_word in translations.items():
            english_query = english_query.replace(vn_word, en_word)
        
        return english_query
    
    def _rule_based_en_to_sql_translation(self, query: str) -> str:
        """Enhanced rule-based English to SQL translation"""
        query_lower = query.lower()
        
        # Bag/handbag queries
        if "handbag" in query_lower or "bag" in query_lower:
            if "women" in query_lower:
                return "SELECT * FROM products WHERE (name LIKE '%túi%' OR name LIKE '%bag%') AND (name LIKE '%nữ%' OR category LIKE '%women%' OR category LIKE '%nữ%') LIMIT 10"
            elif "men" in query_lower:
                return "SELECT * FROM products WHERE (name LIKE '%túi%' OR name LIKE '%bag%') AND (name LIKE '%nam%' OR category LIKE '%men%' OR category LIKE '%nam%') LIMIT 10"
            else:
                return "SELECT * FROM products WHERE name LIKE '%túi%' OR name LIKE '%bag%' LIMIT 10"
        
        # Shoe queries
        elif "shoes" in query_lower or "sandals" in query_lower or "sneakers" in query_lower:
            if "women" in query_lower:
                return "SELECT * FROM products WHERE (name LIKE '%giày%' OR name LIKE '%dép%') AND (name LIKE '%nữ%' OR category LIKE '%women%' OR category LIKE '%nữ%') LIMIT 10"
            elif "men" in query_lower:
                return "SELECT * FROM products WHERE (name LIKE '%giày%' OR name LIKE '%dép%') AND (name LIKE '%nam%' OR category LIKE '%men%' OR category LIKE '%nam%') LIMIT 10"
            else:
                return "SELECT * FROM products WHERE name LIKE '%giày%' OR name LIKE '%dép%' LIMIT 10"
        
        # Clothing queries
        elif "shirt" in query_lower or "t-shirt" in query_lower:
            return "SELECT * FROM products WHERE name LIKE '%áo%' LIMIT 10"
        elif "jacket" in query_lower:
            return "SELECT * FROM products WHERE name LIKE '%áo khoác%' OR name LIKE '%khoác%' LIMIT 10"
        elif "dress" in query_lower:
            return "SELECT * FROM products WHERE name LIKE '%váy%' LIMIT 10"
        
        # Technology queries
        elif "laptop" in query_lower or "computer" in query_lower:
            return "SELECT * FROM products WHERE name LIKE '%laptop%' OR name LIKE '%computer%' OR name LIKE '%máy tính%' LIMIT 10"
        elif "apple" in query_lower and "macbook" in query_lower:
            return "SELECT * FROM products WHERE name LIKE '%Apple%' AND name LIKE '%MacBook%' LIMIT 10"
        
        # Price queries
        elif "price" in query_lower and ("less" in query_lower or "under" in query_lower or "cheap" in query_lower):
            return "SELECT * FROM products WHERE price < 1000000 ORDER BY price ASC LIMIT 10"
        elif "price" in query_lower and ("more" in query_lower or "over" in query_lower or "expensive" in query_lower):
            return "SELECT * FROM products WHERE price > 1000000 ORDER BY price DESC LIMIT 10"
        
        # Brand queries
        elif "nike" in query_lower:
            return "SELECT * FROM products WHERE brand LIKE '%Nike%' OR name LIKE '%Nike%' LIMIT 10"
        elif "adidas" in query_lower:
            return "SELECT * FROM products WHERE brand LIKE '%Adidas%' OR name LIKE '%Adidas%' LIMIT 10"
        
        # Search queries
        elif "find" in query_lower or "search" in query_lower:
            search_term = query_lower.replace("find", "").replace("search", "").strip()
            return f"SELECT * FROM products WHERE name LIKE '%{search_term}%' LIMIT 10"
        
        # Default fallback
        else:
            return "SELECT * FROM products LIMIT 10"

# Create pipeline instances
pipeline1 = VietnameseSQLPipeline()
pipeline2 = VietnameseEnglishSQLPipeline()
