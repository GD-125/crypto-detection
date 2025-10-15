"""
Dashboard Statistics and Metrics Routes
DATABASE DISABLED - Using in-memory storage
"""

from fastapi import APIRouter, Depends
# DATABASE DISABLED
# from sqlalchemy.orm import Session
# from sqlalchemy import func
from datetime import datetime, timedelta

# DATABASE DISABLED - Using in-memory storage instead
# from ..database import get_db
# from ..models import schemas, models
from ..storage import get_storage
from ..models import schemas

router = APIRouter()

@router.get("/stats", response_model=schemas.DashboardStats)
async def get_dashboard_stats(
    # DATABASE DISABLED
    # db: Session = Depends(get_db)
):
    """
    Get dashboard statistics
    """
    storage = get_storage()

    # Total firmware count
    total_firmware = storage.get_total_firmware_count()

    # Analyzed firmware count
    analyzed_count = storage.get_firmware_count_by_status("analyzed")

    # Analyzing firmware count
    analyzing_count = storage.get_firmware_count_by_status("analyzing")

    # Total crypto functions detected
    total_crypto_functions = storage.get_total_analysis_count()

    # Recent uploads (last 7 days)
    recent_uploads = len(storage.get_recent_firmware(days=7, limit=1000))

    # Average confidence score (simplified calculation)
    all_results = storage.list_all_analysis_results(limit=1000)
    avg_confidence = 0.0
    if all_results:
        confidence_sum = 0
        count = 0
        for result in all_results:
            if result.confidence_scores:
                if isinstance(result.confidence_scores, dict):
                    scores = list(result.confidence_scores.values())
                    if scores:
                        confidence_sum += sum(scores) / len(scores)
                        count += 1
        if count > 0:
            avg_confidence = confidence_sum / count

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
    # DATABASE DISABLED
    # db: Session = Depends(get_db)
):
    """
    Get recent firmware uploads and analyses
    """
    storage = get_storage()

    recent_uploads = storage.list_firmware(skip=0, limit=limit)
    recent_analyses = storage.get_recent_analyses(limit=limit)

    return {
        "recent_uploads": [f.to_dict() for f in recent_uploads],
        "recent_analyses": [a.to_dict() for a in recent_analyses],
        "timestamp": datetime.utcnow()
    }

@router.get("/crypto-functions", response_model=schemas.CryptoFunctionStats)
async def get_crypto_function_stats(
    # DATABASE DISABLED
    # db: Session = Depends(get_db)
):
    """
    Get statistics about detected cryptographic functions
    """
    storage = get_storage()
    results = storage.list_all_analysis_results(limit=1000)

    function_types = {}
    for result in results:
        if result.detected_functions:
            for func in result.detected_functions:
                if isinstance(func, dict):
                    func_type = func.get("type", "unknown")
                else:
                    func_type = str(func)

                if func_type in function_types:
                    function_types[func_type] += 1
                else:
                    function_types[func_type] = 1

    return {
        "function_types": function_types,
        "total_detections": sum(function_types.values()),
        "timestamp": datetime.utcnow()
    }
