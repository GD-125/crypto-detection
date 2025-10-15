"""
Analysis Triggering and Management Routes
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from ..database import get_db
from ..models import schemas, models
from ...binary_analyzer.analyzer import BinaryAnalyzer
from ...ai_engine.inference import CryptoDetector
from ...feature_extractor.extractor import FeatureExtractor

router = APIRouter()

def run_analysis_task(firmware_id: int, db: Session):
    """
    Background task for running firmware analysis
    """
    try:
        # Get firmware
        firmware = db.query(models.Firmware).filter(
            models.Firmware.id == firmware_id
        ).first()

        if not firmware:
            return

        # Update status
        firmware.status = "analyzing"
        db.commit()

        # Initialize analyzers
        binary_analyzer = BinaryAnalyzer()
        feature_extractor = FeatureExtractor()
        crypto_detector = CryptoDetector()

        # Step 1: Binary disassembly
        disassembly_result = binary_analyzer.disassemble(firmware.file_path)

        # Step 2: Feature extraction
        features = feature_extractor.extract(disassembly_result)

        # Step 3: AI inference
        predictions = crypto_detector.detect(features)

        # Create analysis result
        result = models.AnalysisResult(
            firmware_id=firmware_id,
            analysis_time=datetime.utcnow(),
            status="completed",
            detected_functions=predictions["functions"],
            confidence_scores=predictions["confidences"],
            explanations=predictions["explanations"],
            metadata=predictions["metadata"]
        )

        db.add(result)

        # Update firmware status
        firmware.status = "analyzed"
        firmware.last_analysis = datetime.utcnow()

        db.commit()

    except Exception as e:
        firmware.status = "error"
        firmware.error_message = str(e)
        db.commit()

@router.post("/start/{firmware_id}", response_model=schemas.AnalysisResponse)
async def start_analysis(
    firmware_id: int,
    background_tasks: BackgroundTasks,
    options: Optional[schemas.AnalysisOptions] = None,
    db: Session = Depends(get_db)
):
    """
    Start analysis for a specific firmware
    """
    # Check if firmware exists
    firmware = db.query(models.Firmware).filter(
        models.Firmware.id == firmware_id
    ).first()

    if not firmware:
        raise HTTPException(status_code=404, detail="Firmware not found")

    if firmware.status == "analyzing":
        raise HTTPException(status_code=400, detail="Analysis already in progress")

    # Add background task
    background_tasks.add_task(run_analysis_task, firmware_id, db)

    return {
        "firmware_id": firmware_id,
        "status": "started",
        "message": "Analysis started in background",
        "timestamp": datetime.utcnow()
    }

@router.get("/status/{firmware_id}", response_model=schemas.AnalysisStatus)
async def get_analysis_status(firmware_id: int, db: Session = Depends(get_db)):
    """
    Get analysis status for a specific firmware
    """
    firmware = db.query(models.Firmware).filter(
        models.Firmware.id == firmware_id
    ).first()

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
    db: Session = Depends(get_db)
):
    """
    Start batch analysis for multiple firmware files
    """
    results = []

    for firmware_id in firmware_ids:
        firmware = db.query(models.Firmware).filter(
            models.Firmware.id == firmware_id
        ).first()

        if firmware and firmware.status != "analyzing":
            background_tasks.add_task(run_analysis_task, firmware_id, db)
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
