"""
Pydantic Schemas for Request/Response Validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Firmware Schemas
class FirmwareUploadResponse(BaseModel):
    id: int
    filename: str
    file_hash: str
    status: str
    message: str
    upload_time: datetime

    class Config:
        from_attributes = True

class FirmwareInfo(BaseModel):
    id: int
    filename: str
    file_hash: str
    file_size: int
    architecture: str
    upload_time: datetime
    last_analysis: Optional[datetime] = None
    status: str
    error_message: Optional[str] = None

    class Config:
        from_attributes = True

# Analysis Schemas
class AnalysisOptions(BaseModel):
    architecture: Optional[str] = "auto"
    deep_scan: Optional[bool] = False
    xai_enabled: Optional[bool] = True

class AnalysisResponse(BaseModel):
    firmware_id: int
    status: str
    message: str
    timestamp: datetime

class AnalysisStatus(BaseModel):
    firmware_id: int
    status: str
    last_analysis: Optional[datetime] = None
    error_message: Optional[str] = None

class BatchAnalysisResponse(BaseModel):
    total: int
    started: int
    results: List[Dict[str, Any]]
    timestamp: datetime

# Results Schemas
class AnalysisResultDetail(BaseModel):
    id: int
    firmware_id: int
    analysis_time: datetime
    status: str
    detected_functions: Optional[List[Dict[str, Any]]] = []
    confidence_scores: Optional[Dict[str, float]] = {}
    explanations: Optional[Dict[str, Any]] = {}
    metadata: Optional[Dict[str, Any]] = {}

    class Config:
        from_attributes = True

class AnalysisResultSummary(BaseModel):
    id: int
    firmware_id: int
    analysis_time: datetime
    status: str

    class Config:
        from_attributes = True

# Dashboard Schemas
class DashboardStats(BaseModel):
    total_firmware: int
    analyzed_count: int
    analyzing_count: int
    pending_count: int
    total_crypto_functions: int
    recent_uploads: int
    average_confidence: float
    timestamp: datetime

class RecentActivity(BaseModel):
    recent_uploads: List[FirmwareInfo]
    recent_analyses: List[AnalysisResultSummary]
    timestamp: datetime

class CryptoFunctionStats(BaseModel):
    function_types: Dict[str, int]
    total_detections: int
    timestamp: datetime

# Crypto Function Schemas
class CryptoFunctionBase(BaseModel):
    name: str
    category: str
    description: Optional[str] = None
    patterns: Optional[Dict[str, Any]] = None

class CryptoFunctionCreate(CryptoFunctionBase):
    pass

class CryptoFunctionResponse(CryptoFunctionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True
