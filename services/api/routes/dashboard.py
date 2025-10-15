"""
Dashboard Statistics and Metrics Routes
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta

from ..database import get_db
from ..models import schemas, models

router = APIRouter()

@router.get("/stats", response_model=schemas.DashboardStats)
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """
    Get dashboard statistics
    """
    # Total firmware count
    total_firmware = db.query(func.count(models.Firmware.id)).scalar()

    # Analyzed firmware count
    analyzed_count = db.query(func.count(models.Firmware.id)).filter(
        models.Firmware.status == "analyzed"
    ).scalar()

    # Analyzing firmware count
    analyzing_count = db.query(func.count(models.Firmware.id)).filter(
        models.Firmware.status == "analyzing"
    ).scalar()

    # Total crypto functions detected
    total_crypto_functions = db.query(func.count(models.AnalysisResult.id)).scalar()

    # Recent uploads (last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_uploads = db.query(func.count(models.Firmware.id)).filter(
        models.Firmware.upload_time >= seven_days_ago
    ).scalar()

    # Average confidence score
    avg_confidence = db.query(func.avg(models.AnalysisResult.confidence_scores)).scalar() or 0

    return {
        "total_firmware": total_firmware,
        "analyzed_count": analyzed_count,
        "analyzing_count": analyzing_count,
        "pending_count": total_firmware - analyzed_count - analyzing_count,
        "total_crypto_functions": total_crypto_functions,
        "recent_uploads": recent_uploads,
        "average_confidence": float(avg_confidence),
        "timestamp": datetime.utcnow()
    }

@router.get("/recent-activity", response_model=schemas.RecentActivity)
async def get_recent_activity(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get recent firmware uploads and analyses
    """
    recent_uploads = db.query(models.Firmware).order_by(
        models.Firmware.upload_time.desc()
    ).limit(limit).all()

    recent_analyses = db.query(models.AnalysisResult).order_by(
        models.AnalysisResult.analysis_time.desc()
    ).limit(limit).all()

    return {
        "recent_uploads": recent_uploads,
        "recent_analyses": recent_analyses,
        "timestamp": datetime.utcnow()
    }

@router.get("/crypto-functions", response_model=schemas.CryptoFunctionStats)
async def get_crypto_function_stats(db: Session = Depends(get_db)):
    """
    Get statistics about detected cryptographic functions
    """
    # This would be customized based on your specific crypto function types
    # Example: AES, RSA, SHA256, etc.

    results = db.query(models.AnalysisResult).all()

    function_types = {}
    for result in results:
        if result.detected_functions:
            for func in result.detected_functions:
                func_type = func.get("type", "unknown")
                if func_type in function_types:
                    function_types[func_type] += 1
                else:
                    function_types[func_type] = 1

    return {
        "function_types": function_types,
        "total_detections": sum(function_types.values()),
        "timestamp": datetime.utcnow()
    }
