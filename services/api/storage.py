"""
In-Memory Storage Module (Database Replacement)
This module provides in-memory storage for firmware and analysis data
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
import json


@dataclass
class FirmwareData:
    """In-memory firmware data structure"""
    id: int
    filename: str
    file_path: str
    file_hash: str
    file_size: int
    architecture: str = "auto"
    upload_time: datetime = field(default_factory=datetime.utcnow)
    last_analysis: Optional[datetime] = None
    status: str = "uploaded"
    error_message: Optional[str] = None

    def to_dict(self):
        data = asdict(self)
        data['upload_time'] = self.upload_time.isoformat() if self.upload_time else None
        data['last_analysis'] = self.last_analysis.isoformat() if self.last_analysis else None
        return data


@dataclass
class AnalysisResultData:
    """In-memory analysis result data structure"""
    id: int
    firmware_id: int
    analysis_time: datetime = field(default_factory=datetime.utcnow)
    status: str = "completed"
    detected_functions: Optional[List] = None
    confidence_scores: Optional[Dict] = None
    explanations: Optional[Dict] = None
    metadata: Optional[Dict] = None

    def to_dict(self):
        data = asdict(self)
        data['analysis_time'] = self.analysis_time.isoformat() if self.analysis_time else None
        return data


class InMemoryStorage:
    """In-memory storage for firmware and analysis results"""

    def __init__(self):
        self.firmware_store: Dict[int, FirmwareData] = {}
        self.analysis_store: Dict[int, AnalysisResultData] = {}
        self.firmware_counter = 0
        self.analysis_counter = 0
        self.hash_index: Dict[str, int] = {}  # file_hash -> firmware_id

    # Firmware operations
    def add_firmware(self, filename: str, file_path: str, file_hash: str,
                     file_size: int, architecture: str = "auto") -> FirmwareData:
        """Add new firmware entry"""
        self.firmware_counter += 1
        firmware = FirmwareData(
            id=self.firmware_counter,
            filename=filename,
            file_path=file_path,
            file_hash=file_hash,
            file_size=file_size,
            architecture=architecture
        )
        self.firmware_store[firmware.id] = firmware
        self.hash_index[file_hash] = firmware.id
        return firmware

    def get_firmware_by_id(self, firmware_id: int) -> Optional[FirmwareData]:
        """Get firmware by ID"""
        return self.firmware_store.get(firmware_id)

    def get_firmware_by_hash(self, file_hash: str) -> Optional[FirmwareData]:
        """Get firmware by hash"""
        firmware_id = self.hash_index.get(file_hash)
        if firmware_id:
            return self.firmware_store.get(firmware_id)
        return None

    def list_firmware(self, skip: int = 0, limit: int = 100) -> List[FirmwareData]:
        """List all firmware with pagination"""
        all_firmware = list(self.firmware_store.values())
        return all_firmware[skip:skip + limit]

    def update_firmware_status(self, firmware_id: int, status: str,
                               error_message: Optional[str] = None):
        """Update firmware status"""
        if firmware_id in self.firmware_store:
            self.firmware_store[firmware_id].status = status
            if error_message:
                self.firmware_store[firmware_id].error_message = error_message

    def delete_firmware(self, firmware_id: int) -> bool:
        """Delete firmware entry"""
        if firmware_id in self.firmware_store:
            firmware = self.firmware_store[firmware_id]
            del self.hash_index[firmware.file_hash]
            del self.firmware_store[firmware_id]
            # Also delete associated analysis results
            self.analysis_store = {
                k: v for k, v in self.analysis_store.items()
                if v.firmware_id != firmware_id
            }
            return True
        return False

    # Analysis operations
    def add_analysis_result(self, firmware_id: int, detected_functions: List,
                           confidence_scores: Dict, explanations: Dict,
                           metadata: Dict) -> AnalysisResultData:
        """Add analysis result"""
        self.analysis_counter += 1
        result = AnalysisResultData(
            id=self.analysis_counter,
            firmware_id=firmware_id,
            detected_functions=detected_functions,
            confidence_scores=confidence_scores,
            explanations=explanations,
            metadata=metadata
        )
        self.analysis_store[result.id] = result

        # Update firmware last_analysis time
        if firmware_id in self.firmware_store:
            self.firmware_store[firmware_id].last_analysis = datetime.utcnow()

        return result

    def get_latest_analysis_by_firmware(self, firmware_id: int) -> Optional[AnalysisResultData]:
        """Get latest analysis result for firmware"""
        results = [r for r in self.analysis_store.values() if r.firmware_id == firmware_id]
        if results:
            return max(results, key=lambda x: x.analysis_time)
        return None

    def list_all_analysis_results(self, skip: int = 0, limit: int = 100) -> List[AnalysisResultData]:
        """List all analysis results"""
        all_results = sorted(self.analysis_store.values(),
                           key=lambda x: x.analysis_time, reverse=True)
        return all_results[skip:skip + limit]

    # Statistics
    def get_total_firmware_count(self) -> int:
        """Get total firmware count"""
        return len(self.firmware_store)

    def get_firmware_count_by_status(self, status: str) -> int:
        """Get firmware count by status"""
        return sum(1 for f in self.firmware_store.values() if f.status == status)

    def get_total_analysis_count(self) -> int:
        """Get total analysis count"""
        return len(self.analysis_store)

    def get_recent_firmware(self, days: int = 7, limit: int = 10) -> List[FirmwareData]:
        """Get recent firmware uploads"""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent = [f for f in self.firmware_store.values() if f.upload_time >= cutoff]
        return sorted(recent, key=lambda x: x.upload_time, reverse=True)[:limit]

    def get_recent_analyses(self, limit: int = 10) -> List[AnalysisResultData]:
        """Get recent analyses"""
        all_results = sorted(self.analysis_store.values(),
                           key=lambda x: x.analysis_time, reverse=True)
        return all_results[:limit]


# Global storage instance
storage = InMemoryStorage()


def get_storage():
    """Get storage instance (replaces get_db dependency)"""
    return storage
