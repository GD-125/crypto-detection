"""
Database Models for ERP-Integrated Cryptographic Detection System
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from ..database import Base

class Firmware(Base):
    """Firmware binary model"""
    __tablename__ = "firmware"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_hash = Column(String(64), unique=True, index=True, nullable=False)
    file_size = Column(Integer, nullable=False)
    architecture = Column(String(50), default="auto")
    upload_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_analysis = Column(DateTime, nullable=True)
    status = Column(String(50), default="uploaded", nullable=False)  # uploaded, analyzing, analyzed, error
    error_message = Column(Text, nullable=True)

    # Relationship
    analysis_results = relationship("AnalysisResult", back_populates="firmware", cascade="all, delete-orphan")

class AnalysisResult(Base):
    """Analysis result model"""
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    firmware_id = Column(Integer, ForeignKey("firmware.id"), nullable=False)
    analysis_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String(50), default="completed", nullable=False)
    detected_functions = Column(JSON, nullable=True)  # List of detected crypto functions
    confidence_scores = Column(JSON, nullable=True)  # Confidence scores for each detection
    explanations = Column(JSON, nullable=True)  # XAI explanations
    metadata = Column(JSON, nullable=True)  # Additional metadata

    # Relationship
    firmware = relationship("Firmware", back_populates="analysis_results")

class CryptoFunction(Base):
    """Cryptographic function catalog"""
    __tablename__ = "crypto_functions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    category = Column(String(50), nullable=False)  # e.g., symmetric, asymmetric, hash
    description = Column(Text, nullable=True)
    patterns = Column(JSON, nullable=True)  # Known patterns for detection
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class TrainingDataset(Base):
    """Training dataset entries"""
    __tablename__ = "training_datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    file_path = Column(String(512), nullable=False)
    architecture = Column(String(50), nullable=False)
    label = Column(String(100), nullable=True)  # Ground truth label
    features = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
