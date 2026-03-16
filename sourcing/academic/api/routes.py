from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from ..academic_sourcing import AcademicSourcingEngine
from ..therapy_dataset_sourcing import find_therapy_datasets

router = APIRouter()

# Initialize engine (Note: For prod, use dependency injection)
engine = AcademicSourcingEngine()


@router.get("/search")
async def search_literature(
    q: str = Query(..., min_length=3, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    sources: Optional[List[str]] = Query(None, description="Filter by source type"),
) -> Dict[str, Any]:
    """
    Search for academic literature across multiple sources.
    """
    try:
        results = engine.search_literature(q, limit=limit, sources=sources)
        return {"results": results, "total": len(results), "facets": {}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets")
async def search_datasets(
    q: Optional[str] = Query(None, min_length=3, description="Search query"),
    min_turns: int = 20,
    min_quality: float = 0.5,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Search for therapy conversation datasets.
    """
    try:
        # Note: limit param is not directly supported in the convenient function wrapper
        # but the underlying method does.
        # For now, we slice the result.
        datasets = find_therapy_datasets(
            query=q, min_turns=min_turns, min_quality=min_quality
        )
        return {"results": datasets[:limit], "total": len(datasets)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
