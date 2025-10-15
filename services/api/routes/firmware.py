"""
Firmware Upload and Management Routes
DATABASE DISABLED - Using in-memory storage
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
# DATABASE DISABLED
# from sqlalchemy.orm import Session
from typing import List
import hashlib
import os
from datetime import datetime

# DATABASE DISABLED - Using in-memory storage instead
# from ..database import get_db
# from ..models import schemas, models
from ..storage import get_storage, InMemoryStorage
from ..models import schemas
# Lazy imports for analysis modules (only needed when running analysis)
# from ...binary_analyzer.analyzer import BinaryAnalyzer
# from ...ai_engine.inference import CryptoDetector

router = APIRouter()

UPLOAD_DIR = "data/uploads"

@router.post("/upload", response_model=schemas.FirmwareUploadResponse)
async def upload_firmware(
    file: UploadFile = File(...),
    architecture: str = "auto",
    # DATABASE DISABLED
    # db: Session = Depends(get_db)
):
    """
    Upload firmware binary for analysis
    """
    try:
        storage = get_storage()

        # Read file content
        content = await file.read()

        # Calculate hash
        file_hash = hashlib.sha256(content).hexdigest()

        # Check if file already exists
        existing = storage.get_firmware_by_hash(file_hash)

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
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, f"{file_hash}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(content)

        # Create storage entry
        firmware = storage.add_firmware(
            filename=file.filename,
            file_path=file_path,
            file_hash=file_hash,
            file_size=len(content),
            architecture=architecture
        )

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
    # DATABASE DISABLED
    # db: Session = Depends(get_db)
):
    """
    List all uploaded firmware files
    """
    storage = get_storage()
    firmware_list = storage.list_firmware(skip=skip, limit=limit)
    return [f.to_dict() for f in firmware_list]

@router.get("/{firmware_id}", response_model=schemas.FirmwareInfo)
async def get_firmware(
    firmware_id: int,
    # DATABASE DISABLED
    # db: Session = Depends(get_db)
):
    """
    Get firmware details by ID
    """
    storage = get_storage()
    firmware = storage.get_firmware_by_id(firmware_id)

    if not firmware:
        raise HTTPException(status_code=404, detail="Firmware not found")

    return firmware.to_dict()

@router.delete("/{firmware_id}")
async def delete_firmware(
    firmware_id: int,
    # DATABASE DISABLED
    # db: Session = Depends(get_db)
):
    """
    Delete firmware by ID
    """
    storage = get_storage()
    firmware = storage.get_firmware_by_id(firmware_id)

    if not firmware:
        raise HTTPException(status_code=404, detail="Firmware not found")

    # Delete file
    if os.path.exists(firmware.file_path):
        os.remove(firmware.file_path)

    # Delete storage entry
    storage.delete_firmware(firmware_id)

    return {"status": "success", "message": "Firmware deleted"}
