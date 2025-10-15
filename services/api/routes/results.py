"""
Analysis Results Routes
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional

from ..database import get_db
from ..models import schemas, models

router = APIRouter()

@router.get("/{firmware_id}", response_model=schemas.AnalysisResultDetail)
async def get_results(firmware_id: int, db: Session = Depends(get_db)):
    """
    Get analysis results for a specific firmware
    """
    result = db.query(models.AnalysisResult).filter(
        models.AnalysisResult.firmware_id == firmware_id
    ).order_by(models.AnalysisResult.analysis_time.desc()).first()

    if not result:
        raise HTTPException(status_code=404, detail="Analysis results not found")

    return result

@router.get("/list/all", response_model=List[schemas.AnalysisResultSummary])
async def list_all_results(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List all analysis results
    """
    results = db.query(models.AnalysisResult).offset(skip).limit(limit).all()
    return results

@router.get("/export/{firmware_id}")
async def export_results(
    firmware_id: int,
    format: str = "json",
    db: Session = Depends(get_db)
):
    """
    Export analysis results in various formats (JSON, CSV, PDF)
    """
    result = db.query(models.AnalysisResult).filter(
        models.AnalysisResult.firmware_id == firmware_id
    ).order_by(models.AnalysisResult.analysis_time.desc()).first()

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
