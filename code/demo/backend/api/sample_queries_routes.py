from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sample_queries_data import SAMPLE_QUERIES_DATA
except ImportError:
    # Fallback data if import fails
    SAMPLE_QUERIES_DATA = {
        'simple': [],
        'medium': [],
        'complex': []
    }

router = APIRouter()

@router.get("/sample-queries")
async def get_sample_queries() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all sample queries organized by complexity level.
    Returns 100 queries per complexity level with Vietnamese-SQL pairs.
    """
    try:
        return {
            "simple": SAMPLE_QUERIES_DATA.get('simple', []),
            "medium": SAMPLE_QUERIES_DATA.get('medium', []),
            "complex": SAMPLE_QUERIES_DATA.get('complex', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sample queries: {str(e)}")

@router.get("/sample-queries/{complexity}")
async def get_sample_queries_by_complexity(complexity: str) -> List[Dict[str, Any]]:
    """
    Get sample queries for a specific complexity level.
    
    Args:
        complexity: One of 'simple', 'medium', 'complex'
    """
    if complexity not in ['simple', 'medium', 'complex']:
        raise HTTPException(status_code=400, detail="Complexity must be one of: simple, medium, complex")
    
    try:
        queries = SAMPLE_QUERIES_DATA.get(complexity, [])
        return queries
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load {complexity} queries: {str(e)}")

@router.get("/sample-queries-stats")
async def get_sample_queries_stats() -> Dict[str, Any]:
    """
    Get statistics about the sample queries dataset.
    """
    try:
        stats = {
            "total_queries": sum(len(queries) for queries in SAMPLE_QUERIES_DATA.values()),
            "by_complexity": {
                complexity: len(queries) 
                for complexity, queries in SAMPLE_QUERIES_DATA.items()
            },
            "has_sql": {
                complexity: sum(1 for query in queries if 'sql' in query and query['sql'])
                for complexity, queries in SAMPLE_QUERIES_DATA.items()
            }
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sample queries stats: {str(e)}")
