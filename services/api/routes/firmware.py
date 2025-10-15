"""
Firmware Upload and Management Routes
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List
import hashlib
import os
from datetime import datetime

from ..database import get_db
from ..models import schemas, models
from ...binary_analyzer.analyzer import BinaryAnalyzer
from ...ai_engine.inference import CryptoDetector

router = APIRouter()

UPLOAD_DIR = "data/uploads"

@router.post("/upload", response_model=schemas.FirmwareUploadResponse)
async def upload_firmware(
    file: UploadFile = File(...),
    architecture: str = "auto",
    db: Session = Depends(get_db)
):
    """
    Upload firmware binary for analysis
    """
    try:
        # Read file content
        content = await file.read()

        # Calculate hash
        file_hash = hashlib.sha256(content).hexdigest()

        # Check if file already exists
        existing = db.query(models.Firmware).filter(
            models.Firmware.file_hash == file_hash
        ).first()

        if existing:
            return {
                "id": existing.id,
                "filename": existing.filename,
                "file_hash": existing.file_hash,
                "status": "already_exists",
                "message": "File already uploaded",
                "upload_time": existing.upload_time
            }

        # Save file
        file_path = os.path.join(UPLOAD_DIR, f"{file_hash}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(content)

        # Create database entry
        firmware = models.Firmware(
            filename=file.filename,
            file_path=file_path,
            file_hash=file_hash,
            file_size=len(content),
            architecture=architecture,
            upload_time=datetime.utcnow(),
            status="uploaded"
        )

        db.add(firmware)
        db.commit()
        db.refresh(firmware)

        return {
            "id": firmware.id,
            "filename": firmware.filename,
            "file_hash": firmware.file_hash,
            "status": "success",
            "message": "File uploaded successfully",
            "upload_time": firmware.upload_time
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/list", response_model=List[schemas.FirmwareInfo])
async def list_firmware(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List all uploaded firmware files
    """
    firmware_list = db.query(models.Firmware).offset(skip).limit(limit).all()
    return firmware_list

@router.get("/{firmware_id}", response_model=schemas.FirmwareInfo)
async def get_firmware(firmware_id: int, db: Session = Depends(get_db)):
    """
    Get firmware details by ID
    """
    firmware = db.query(models.Firmware).filter(
        models.Firmware.id == firmware_id
    ).first()

    if not firmware:
        raise HTTPException(status_code=404, detail="Firmware not found")

    return firmware

@router.delete("/{firmware_id}")
async def delete_firmware(firmware_id: int, db: Session = Depends(get_db)):
    """
    Delete firmware by ID
    """
    firmware = db.query(models.Firmware).filter(
        models.Firmware.id == firmware_id
    ).first()

    if not firmware:
        raise HTTPException(status_code=404, detail="Firmware not found")

    # Delete file
    if os.path.exists(firmware.file_path):
        os.remove(firmware.file_path)

    # Delete database entry
    db.delete(firmware)
    db.commit()

    return {"status": "success", "message": "Firmware deleted"}
