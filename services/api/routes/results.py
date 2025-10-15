"""
Analysis Results Routes
DATABASE DISABLED - Using in-memory storage
"""

from fastapi import APIRouter, HTTPException, Depends
# DATABASE DISABLED
# from sqlalchemy.orm import Session
from typing import List, Optional

# DATABASE DISABLED - Using in-memory storage instead
# from ..database import get_db
# from ..models import schemas, models
from ..storage import get_storage
from ..models import schemas

router = APIRouter()

@router.get("/{firmware_id}", response_model=schemas.AnalysisResultDetail)
async def get_results(
    firmware_id: int,
    # DATABASE DISABLED
    # db: Session = Depends(get_db)
):
    """
    Get analysis results for a specific firmware
    """
    storage = get_storage()
    result = storage.get_latest_analysis_by_firmware(firmware_id)

    if not result:
        raise HTTPException(status_code=404, detail="Analysis results not found")

    return result.to_dict()

@router.get("/list/all", response_model=List[schemas.AnalysisResultSummary])
async def list_all_results(
    skip: int = 0,
    limit: int = 100,
    # DATABASE DISABLED
    # db: Session = Depends(get_db)
):
    """
    List all analysis results
    """
    storage = get_storage()
    results = storage.list_all_analysis_results(skip=skip, limit=limit)
    return [r.to_dict() for r in results]

@router.get("/export/{firmware_id}")
async def export_results(
    firmware_id: int,
    format: str = "json",
    # DATABASE DISABLED
    # db: Session = Depends(get_db)
):
    """
    Export analysis results in various formats (JSON, CSV, PDF)
    """
    storage = get_storage()
    result = storage.get_latest_analysis_by_firmware(firmware_id)

    if not result:
        raise HTTPException(status_code=404, detail="Analysis results not found")

    if format == "json":
        return {
            "firmware_id": result.firmware_id,
            "analysis_time": result.analysis_time.isoformat(),
            "detected_functions": result.detected_functions,
            "confidence_scores": result.confidence_scores,
            "explanations": result.explanations,
            "metadata": result.metadata
        }
    elif format == "csv":
        # CSV export logic
        raise HTTPException(status_code=501, detail="CSV export not yet implemented")
    elif format == "pdf":
        # PDF export logic
        raise HTTPException(status_code=501, detail="PDF export not yet implemented")
    else:
        raise HTTPException(status_code=400, detail="Invalid format")
