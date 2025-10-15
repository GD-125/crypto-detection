"""
Analysis Triggering and Management Routes
DATABASE DISABLED - Using in-memory storage
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
# DATABASE DISABLED
# from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

# DATABASE DISABLED - Using in-memory storage instead
# from ..database import get_db
# from ..models import schemas, models
from ..storage import get_storage
from ..models import schemas
# Lazy imports - these modules will be imported when actually needed
# from ...binary_analyzer.analyzer import BinaryAnalyzer
# from ...ai_engine.inference import CryptoDetector
# from ...feature_extractor.extractor import FeatureExtractor

router = APIRouter()

def run_analysis_task(firmware_id: int):
    """
    Background task for running firmware analysis
    """
    storage = get_storage()

    try:
        # Get firmware
        firmware = storage.get_firmware_by_id(firmware_id)

        if not firmware:
            return

        # Update status
        storage.update_firmware_status(firmware_id, "analyzing")

        # Lazy import of analysis modules
        # NOTE: Import these when your analyzer modules are ready
        # from ...binary_analyzer.analyzer import BinaryAnalyzer
        # from ...ai_engine.inference import CryptoDetector
        # from ...feature_extractor.extractor import FeatureExtractor

        # Initialize analyzers
        # binary_analyzer = BinaryAnalyzer()
        # feature_extractor = FeatureExtractor()
        # crypto_detector = CryptoDetector()

        # Step 1: Binary disassembly
        # disassembly_result = binary_analyzer.disassemble(firmware.file_path)

        # Step 2: Feature extraction
        # features = feature_extractor.extract(disassembly_result)

        # Step 3: AI inference
        # predictions = crypto_detector.detect(features)

        # TEMPORARY: Mock predictions for testing
        predictions = {
            "functions": [],
            "confidences": {},
            "explanations": {},
            "metadata": {"note": "Analysis modules not yet integrated"}
        }

        # Create analysis result
        storage.add_analysis_result(
            firmware_id=firmware_id,
            detected_functions=predictions.get("functions", []),
            confidence_scores=predictions.get("confidences", {}),
            explanations=predictions.get("explanations", {}),
            metadata=predictions.get("metadata", {})
        )

        # Update firmware status
        storage.update_firmware_status(firmware_id, "analyzed")

    except Exception as e:
        storage.update_firmware_status(firmware_id, "error", str(e))

@router.post("/start/{firmware_id}", response_model=schemas.AnalysisResponse)
async def start_analysis(
    firmware_id: int,
    background_tasks: BackgroundTasks,
    options: Optional[schemas.AnalysisOptions] = None,
    # DATABASE DISABLED
    # db: Session = Depends(get_db)
):
    """
    Start analysis for a specific firmware
    """
    storage = get_storage()

    # Check if firmware exists
    firmware = storage.get_firmware_by_id(firmware_id)

    if not firmware:
        raise HTTPException(status_code=404, detail="Firmware not found")

    if firmware.status == "analyzing":
        raise HTTPException(status_code=400, detail="Analysis already in progress")

    # Add background task
    background_tasks.add_task(run_analysis_task, firmware_id)

    return {
        "firmware_id": firmware_id,
        "status": "started",
        "message": "Analysis started in background",
        "timestamp": datetime.utcnow()
    }

@router.get("/status/{firmware_id}", response_model=schemas.AnalysisStatus)
async def get_analysis_status(
    firmware_id: int,
    # DATABASE DISABLED
    # db: Session = Depends(get_db)
):
    """
    Get analysis status for a specific firmware
    """
    storage = get_storage()
    firmware = storage.get_firmware_by_id(firmware_id)

    if not firmware:
        raise HTTPException(status_code=404, detail="Firmware not found")

    return {
        "firmware_id": firmware_id,
        "status": firmware.status,
        "last_analysis": firmware.last_analysis,
        "error_message": firmware.error_message
    }

@router.post("/batch", response_model=schemas.BatchAnalysisResponse)
async def start_batch_analysis(
    firmware_ids: List[int],
    background_tasks: BackgroundTasks,
    # DATABASE DISABLED
    # db: Session = Depends(get_db)
):
    """
    Start batch analysis for multiple firmware files
    """
    storage = get_storage()
    results = []

    for firmware_id in firmware_ids:
        firmware = storage.get_firmware_by_id(firmware_id)

        if firmware and firmware.status != "analyzing":
            background_tasks.add_task(run_analysis_task, firmware_id)
            results.append({
                "firmware_id": firmware_id,
                "status": "started"
            })
        elif firmware:
            results.append({
                "firmware_id": firmware_id,
                "status": "already_analyzing"
            })
        else:
            results.append({
                "firmware_id": firmware_id,
                "status": "not_found"
            })

    return {
        "total": len(firmware_ids),
        "started": len([r for r in results if r["status"] == "started"]),
        "results": results,
        "timestamp": datetime.utcnow()
    }
