from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import asyncio
from datetime import datetime
import logging

from sample_queries import SAMPLE_QUERIES
from models.pipelines import pipeline1, pipeline2
from shared_models import db_manager, execute_sql_query

logger = logging.getLogger(__name__)
router = APIRouter()

class QueryResult(BaseModel):
    query_id: int
    complexity: str
    vietnamese_query: str
    challenge: str
    expected_sql_type: str
    pipeline1_result: Optional[Dict[str, Any]] = None
    pipeline2_result: Optional[Dict[str, Any]] = None
    execution_timestamp: str

class SampleQueryResponse(BaseModel):
    total_queries: int
    results: List[QueryResult]
    summary: Dict[str, Any]
    execution_time: float

class BatchExecutionRequest(BaseModel):
    complexity: str  # "simple", "medium", "complex", or "all"
    pipeline: str = "both"  # "pipeline1", "pipeline2", or "both"

class BatchExecutionResponse(BaseModel):
    complexity: str
    total_batches: int
    total_queries: int
    batch_results: List[Dict[str, Any]]
    overall_stats: Dict[str, Any]
    total_execution_time: float

class ComplexityReport(BaseModel):
    complexity: str
    total_queries: int
    pipeline1_stats: Dict[str, Any]
    pipeline2_stats: Dict[str, Any]
    comparison: Dict[str, Any]

@router.post("/execute-samples-batch", response_model=BatchExecutionResponse)
async def execute_sample_queries_batch(request: BatchExecutionRequest):
    """Execute sample queries in batches for optimal performance"""
    start_time = time.time()
    
    # Get queries based on complexity
    if request.complexity == "all":
        all_queries = []
        for complexity, queries in SAMPLE_QUERIES.items():
            for query_data in queries:
                all_queries.append({**query_data, "complexity": complexity})
        selected_queries = all_queries
    else:
        selected_queries = [{**query_data, "complexity": request.complexity} 
                          for query_data in SAMPLE_QUERIES.get(request.complexity, [])]
    
    if not selected_queries:
        raise HTTPException(status_code=400, detail=f"No queries found for complexity: {request.complexity}")
    
    # Calculate batching: 10 queries per batch
    batch_size = 10
    total_queries = len(selected_queries)
    total_batches = (total_queries + batch_size - 1) // batch_size  # Ceiling division
    
    logger.info(f"Executing {total_queries} {request.complexity} queries in {total_batches} batches of {batch_size}")
    
    batch_results = []
    overall_stats = {
        "pipeline1": {"success": 0, "errors": 0, "total_time": 0, "batches_processed": 0},
        "pipeline2": {"success": 0, "errors": 0, "total_time": 0, "batches_processed": 0}
    }
    
    # Process queries in batches
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_queries)
        batch_queries = selected_queries[start_idx:end_idx]
        
        batch_start_time = time.time()
        batch_result = {
            "batch_number": batch_num + 1,
            "queries_in_batch": len(batch_queries),
            "pipeline1_results": [],
            "pipeline2_results": [],
            "batch_execution_time": 0
        }
        
        # Extract just the query strings for batch processing
        query_strings = [q["query"] for q in batch_queries]
        
        # Execute Pipeline 1 batch if requested
        if request.pipeline in ["pipeline1", "both"]:
            try:
                logger.info(f"Processing batch {batch_num + 1}/{total_batches} with Pipeline 1")
                p1_batch_results = await pipeline1.vietnamese_to_sql_batch(query_strings, batch_size)
                
                for i, result in enumerate(p1_batch_results):
                    if result["success"]:
                        # Execute SQL and get results
                        try:
                            db_results, sql_error = execute_sql_query(result["sql_query"])
                            if not sql_error:
                                batch_result["pipeline1_results"].append({
                                    "query": batch_queries[i]["query"],
                                    "complexity": batch_queries[i]["complexity"],
                                    "sql_query": result["sql_query"],
                                    "results": db_results[:5],  # Limit for display
                                    "result_count": len(db_results),
                                    "execution_time": result["execution_time"],
                                    "success": True,
                                    "error": None
                                })
                                overall_stats["pipeline1"]["success"] += 1
                            else:
                                batch_result["pipeline1_results"].append({
                                    "query": batch_queries[i]["query"],
                                    "complexity": batch_queries[i]["complexity"],
                                    "sql_query": result["sql_query"],
                                    "results": [],
                                    "result_count": 0,
                                    "execution_time": result["execution_time"],
                                    "success": False,
                                    "error": f"SQL execution error: {sql_error}"
                                })
                                overall_stats["pipeline1"]["errors"] += 1
                        except Exception as e:
                            batch_result["pipeline1_results"].append({
                                "query": batch_queries[i]["query"],
                                "complexity": batch_queries[i]["complexity"],
                                "sql_query": result["sql_query"],
                                "results": [],
                                "result_count": 0,
                                "execution_time": result["execution_time"],
                                "success": False,
                                "error": f"Database error: {str(e)}"
                            })
                            overall_stats["pipeline1"]["errors"] += 1
                    else:
                        batch_result["pipeline1_results"].append({
                            "query": batch_queries[i]["query"],
                            "complexity": batch_queries[i]["complexity"],
                            "sql_query": "",
                            "results": [],
                            "result_count": 0,
                            "execution_time": result["execution_time"],
                            "success": False,
                            "error": result["error"]
                        })
                        overall_stats["pipeline1"]["errors"] += 1
                
                overall_stats["pipeline1"]["batches_processed"] += 1
                overall_stats["pipeline1"]["total_time"] += sum(r["execution_time"] for r in p1_batch_results)
                
            except Exception as e:
                logger.error(f"Pipeline 1 batch {batch_num + 1} failed: {e}")
                for i in range(len(batch_queries)):
                    batch_result["pipeline1_results"].append({
                        "query": batch_queries[i]["query"],
                        "complexity": batch_queries[i]["complexity"],
                        "sql_query": "",
                        "results": [],
                        "result_count": 0,
                        "execution_time": 0,
                        "success": False,
                        "error": f"Batch processing error: {str(e)}"
                    })
                    overall_stats["pipeline1"]["errors"] += 1
        
        # Execute Pipeline 2 batch if requested
        if request.pipeline in ["pipeline2", "both"]:
            try:
                logger.info(f"Processing batch {batch_num + 1}/{total_batches} with Pipeline 2")
                p2_batch_results = await pipeline2.full_pipeline_batch(query_strings, batch_size)
                
                for i, result in enumerate(p2_batch_results):
                    if result["success"]:
                        # Execute SQL and get results
                        try:
                            db_results, sql_error = execute_sql_query(result["sql_query"])
                            if not sql_error:
                                batch_result["pipeline2_results"].append({
                                    "query": batch_queries[i]["query"],
                                    "complexity": batch_queries[i]["complexity"],
                                    "english_query": result["english_query"],
                                    "sql_query": result["sql_query"],
                                    "results": db_results[:5],  # Limit for display
                                    "result_count": len(db_results),
                                    "execution_time": result["execution_time"],
                                    "vn_en_time": result["vn_en_time"],
                                    "en_sql_time": result["en_sql_time"],
                                    "success": True,
                                    "error": None
                                })
                                overall_stats["pipeline2"]["success"] += 1
                            else:
                                batch_result["pipeline2_results"].append({
                                    "query": batch_queries[i]["query"],
                                    "complexity": batch_queries[i]["complexity"],
                                    "english_query": result["english_query"],
                                    "sql_query": result["sql_query"],
                                    "results": [],
                                    "result_count": 0,
                                    "execution_time": result["execution_time"],
                                    "vn_en_time": result["vn_en_time"],
                                    "en_sql_time": result["en_sql_time"],
                                    "success": False,
                                    "error": f"SQL execution error: {sql_error}"
                                })
                                overall_stats["pipeline2"]["errors"] += 1
                        except Exception as e:
                            batch_result["pipeline2_results"].append({
                                "query": batch_queries[i]["query"],
                                "complexity": batch_queries[i]["complexity"],
                                "english_query": result["english_query"],
                                "sql_query": result["sql_query"],
                                "results": [],
                                "result_count": 0,
                                "execution_time": result["execution_time"],
                                "vn_en_time": result["vn_en_time"],
                                "en_sql_time": result["en_sql_time"],
                                "success": False,
                                "error": f"Database error: {str(e)}"
                            })
                            overall_stats["pipeline2"]["errors"] += 1
                    else:
                        batch_result["pipeline2_results"].append({
                            "query": batch_queries[i]["query"],
                            "complexity": batch_queries[i]["complexity"],
                            "english_query": result["english_query"],
                            "sql_query": "",
                            "results": [],
                            "result_count": 0,
                            "execution_time": result["execution_time"],
                            "vn_en_time": result["vn_en_time"],
                            "en_sql_time": result["en_sql_time"],
                            "success": False,
                            "error": result["error"]
                        })
                        overall_stats["pipeline2"]["errors"] += 1
                
                overall_stats["pipeline2"]["batches_processed"] += 1
                overall_stats["pipeline2"]["total_time"] += sum(r["execution_time"] for r in p2_batch_results)
                
            except Exception as e:
                logger.error(f"Pipeline 2 batch {batch_num + 1} failed: {e}")
                for i in range(len(batch_queries)):
                    batch_result["pipeline2_results"].append({
                        "query": batch_queries[i]["query"],
                        "complexity": batch_queries[i]["complexity"],
                        "english_query": "",
                        "sql_query": "",
                        "results": [],
                        "result_count": 0,
                        "execution_time": 0,
                        "vn_en_time": 0,
                        "en_sql_time": 0,
                        "success": False,
                        "error": f"Batch processing error: {str(e)}"
                    })
                    overall_stats["pipeline2"]["errors"] += 1
        
        batch_result["batch_execution_time"] = time.time() - batch_start_time
        batch_results.append(batch_result)
        
        logger.info(f"Batch {batch_num + 1}/{total_batches} completed in {batch_result['batch_execution_time']:.2f}s")
    
    total_execution_time = time.time() - start_time
    
    logger.info(f"All batches completed: {total_queries} queries in {total_batches} batches, {total_execution_time:.2f}s total")
    
    return BatchExecutionResponse(
        complexity=request.complexity,
        total_batches=total_batches,
        total_queries=total_queries,
        batch_results=batch_results,
        overall_stats=overall_stats,
        total_execution_time=total_execution_time
    )

@router.post("/execute-samples")
async def execute_sample_queries():
    """Execute all sample queries and return comprehensive results (Legacy endpoint)"""
    start_time = time.time()
    results = []
    query_id = 1
    
    # Statistics tracking
    stats = {
        "simple": {"pipeline1": {"success": 0, "errors": 0, "total_time": 0}, 
                  "pipeline2": {"success": 0, "errors": 0, "total_time": 0}},
        "medium": {"pipeline1": {"success": 0, "errors": 0, "total_time": 0}, 
                  "pipeline2": {"success": 0, "errors": 0, "total_time": 0}},
        "complex": {"pipeline1": {"success": 0, "errors": 0, "total_time": 0}, 
                   "pipeline2": {"success": 0, "errors": 0, "total_time": 0}}
    }
    
    for complexity, queries in SAMPLE_QUERIES.items():
        logger.info(f"Executing {len(queries)} {complexity} queries...")
        
        for query_data in queries:
            query_result = QueryResult(
                query_id=query_id,
                complexity=complexity,
                vietnamese_query=query_data["query"],
                challenge=query_data["challenge"],
                expected_sql_type=query_data["expected_sql_type"],
                execution_timestamp=datetime.now().isoformat()
            )
            
            # Execute Pipeline 1
            try:
                p1_start = time.time()
                sql_result = pipeline1.vietnamese_to_sql(query_data["query"])
                
                if sql_result["success"]:
                    db_results = db_manager.execute_query(sql_result["sql_query"])
                    p1_time = time.time() - p1_start
                    
                    query_result.pipeline1_result = {
                        "sql_query": sql_result["sql_query"],
                        "results": db_results[:5],  # Limit to first 5 results for display
                        "result_count": len(db_results),
                        "execution_time": p1_time,
                        "success": True,
                        "error": None
                    }
                    
                    stats[complexity]["pipeline1"]["success"] += 1
                    stats[complexity]["pipeline1"]["total_time"] += p1_time
                else:
                    query_result.pipeline1_result = {
                        "sql_query": "",
                        "results": [],
                        "result_count": 0,
                        "execution_time": time.time() - p1_start,
                        "success": False,
                        "error": sql_result["error"]
                    }
                    stats[complexity]["pipeline1"]["errors"] += 1
                    
            except Exception as e:
                logger.error(f"Pipeline 1 error for query {query_id}: {e}")
                query_result.pipeline1_result = {
                    "sql_query": "",
                    "results": [],
                    "result_count": 0,
                    "execution_time": 0,
                    "success": False,
                    "error": str(e)
                }
                stats[complexity]["pipeline1"]["errors"] += 1
            
            # Execute Pipeline 2
            try:
                p2_start = time.time()
                pipeline_result = pipeline2.full_pipeline(query_data["query"])
                
                if pipeline_result["success"]:
                    db_results = db_manager.execute_query(pipeline_result["sql_query"])
                    p2_time = time.time() - p2_start
                    
                    query_result.pipeline2_result = {
                        "sql_query": pipeline_result["sql_query"],
                        "english_query": pipeline_result["english_query"],
                        "results": db_results[:5],  # Limit to first 5 results for display
                        "result_count": len(db_results),
                        "execution_time": p2_time,
                        "vn_en_time": pipeline_result["vn_en_time"],
                        "en_sql_time": pipeline_result["en_sql_time"],
                        "success": True,
                        "error": None
                    }
                    
                    stats[complexity]["pipeline2"]["success"] += 1
                    stats[complexity]["pipeline2"]["total_time"] += p2_time
                else:
                    query_result.pipeline2_result = {
                        "sql_query": "",
                        "english_query": pipeline_result.get("english_query", ""),
                        "results": [],
                        "result_count": 0,
                        "execution_time": time.time() - p2_start,
                        "success": False,
                        "error": pipeline_result["error"]
                    }
                    stats[complexity]["pipeline2"]["errors"] += 1
                    
            except Exception as e:
                logger.error(f"Pipeline 2 error for query {query_id}: {e}")
                query_result.pipeline2_result = {
                    "sql_query": "",
                    "english_query": "",
                    "results": [],
                    "result_count": 0,
                    "execution_time": 0,
                    "success": False,
                    "error": str(e)
                }
                stats[complexity]["pipeline2"]["errors"] += 1
            
            results.append(query_result)
            query_id += 1
    
    # Generate summary statistics
    total_execution_time = time.time() - start_time
    total_queries = len(results)
    
    summary = {
        "total_queries": total_queries,
        "total_execution_time": total_execution_time,
        "complexity_breakdown": {
            complexity: len(queries) for complexity, queries in SAMPLE_QUERIES.items()
        },
        "overall_stats": stats,
        "timestamp": datetime.now().isoformat()
    }
    
    return SampleQueryResponse(
        total_queries=total_queries,
        results=results,
        summary=summary,
        execution_time=total_execution_time
    )

@router.get("/complexity-report")
async def get_complexity_report():
    """Generate detailed report for each complexity level"""
    # This would typically load from stored results, but for now we'll execute fresh
    sample_results = await execute_sample_queries()
    
    reports = []
    
    for complexity in ["simple", "medium", "complex"]:
        complexity_queries = [r for r in sample_results.results if r.complexity == complexity]
        
        # Pipeline 1 analysis
        p1_successes = [q for q in complexity_queries if q.pipeline1_result and q.pipeline1_result["success"]]
        p1_errors = [q for q in complexity_queries if q.pipeline1_result and not q.pipeline1_result["success"]]
        p1_avg_time = sum(q.pipeline1_result["execution_time"] for q in p1_successes) / len(p1_successes) if p1_successes else 0
        
        # Pipeline 2 analysis
        p2_successes = [q for q in complexity_queries if q.pipeline2_result and q.pipeline2_result["success"]]
        p2_errors = [q for q in complexity_queries if q.pipeline2_result and not q.pipeline2_result["success"]]
        p2_avg_time = sum(q.pipeline2_result["execution_time"] for q in p2_successes) / len(p2_successes) if p2_successes else 0
        
        # Comparison analysis
        both_success = [q for q in complexity_queries 
                       if q.pipeline1_result and q.pipeline1_result["success"] 
                       and q.pipeline2_result and q.pipeline2_result["success"]]
        
        sql_matches = sum(1 for q in both_success 
                         if q.pipeline1_result["sql_query"].strip() == q.pipeline2_result["sql_query"].strip())
        
        report = ComplexityReport(
            complexity=complexity,
            total_queries=len(complexity_queries),
            pipeline1_stats={
                "success_count": len(p1_successes),
                "error_count": len(p1_errors),
                "success_rate": (len(p1_successes) / len(complexity_queries)) * 100,
                "average_time_ms": p1_avg_time * 1000,
                "common_errors": [q.pipeline1_result["error"] for q in p1_errors if q.pipeline1_result["error"]][:3]
            },
            pipeline2_stats={
                "success_count": len(p2_successes),
                "error_count": len(p2_errors),
                "success_rate": (len(p2_successes) / len(complexity_queries)) * 100,
                "average_time_ms": p2_avg_time * 1000,
                "common_errors": [q.pipeline2_result["error"] for q in p2_errors if q.pipeline2_result["error"]][:3]
            },
            comparison={
                "sql_exact_match_rate": (sql_matches / len(both_success)) * 100 if both_success else 0,
                "both_success_rate": (len(both_success) / len(complexity_queries)) * 100,
                "faster_pipeline": "Pipeline 1" if p1_avg_time < p2_avg_time else "Pipeline 2",
                "time_difference_ms": abs(p1_avg_time - p2_avg_time) * 1000,
                "accuracy_difference": abs(len(p1_successes) - len(p2_successes))
            }
        )
        
        reports.append(report)
    
    return {
        "complexity_reports": reports,
        "overall_summary": {
            "total_sample_queries": sum(r.total_queries for r in reports),
            "best_performing_complexity": {
                "pipeline1": max(reports, key=lambda r: r.pipeline1_stats["success_rate"]).complexity,
                "pipeline2": max(reports, key=lambda r: r.pipeline2_stats["success_rate"]).complexity
            },
            "most_challenging_complexity": {
                "pipeline1": min(reports, key=lambda r: r.pipeline1_stats["success_rate"]).complexity,
                "pipeline2": min(reports, key=lambda r: r.pipeline2_stats["success_rate"]).complexity
            }
        },
        "timestamp": datetime.now().isoformat()
    }

@router.get("/sample-queries")
async def get_sample_queries():
    """Get all sample queries without execution"""
    return {
        "sample_queries": SAMPLE_QUERIES,
        "total_count": sum(len(queries) for queries in SAMPLE_QUERIES.values()),
        "complexity_counts": {
            complexity: len(queries) for complexity, queries in SAMPLE_QUERIES.items()
        }
    }
