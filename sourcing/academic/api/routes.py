import logging
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from ai.sourcing.academic.academic_sourcing import AcademicSourcingEngine
from ai.sourcing.academic.therapy_dataset_sourcing import find_therapy_datasets

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize engine (Note: For prod, use dependency injection)
engine = AcademicSourcingEngine()


@router.get("/search")
async def search_literature(
    q: Annotated[str, Query(min_length=3, description="Search query")],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    sources: Annotated[list[str] | None, Query(description="Filter by source type")] = None,
) -> dict[str, Any]:
    """
    Search for academic literature across multiple sources.
    """
    try:
        results = engine.search_literature(q, limit=limit, sources=sources)
        return {"results": results, "total": len(results), "facets": {}}
    except Exception as e:
        logger.exception("Error while searching literature.")
        raise HTTPException(status_code=500, detail="Internal server error occurred while searching literature.") from e


@router.get("/datasets")
async def search_datasets(
    q: Annotated[str | None, Query(min_length=3, description="Search query")] = None,
    min_turns: int = 20,
    min_quality: float = 0.5,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Search for therapy conversation datasets.
    """
    try:
        # Note: limit param is not directly supported in the convenient function wrapper
        # but the underlying method does.
        # For now, we slice the result.
        datasets = find_therapy_datasets(query=q, min_turns=min_turns, min_quality=min_quality)
        return {"results": datasets[:limit], "total": len(datasets)}
    except Exception as e:
        logger.exception("Error while searching datasets.")
        raise HTTPException(status_code=500, detail="Internal server error occurred while searching datasets.") from e
