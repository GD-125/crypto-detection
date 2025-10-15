"""
Configuration Module for Cryptographic Function Detection System
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Database
    database_url: str = "sqlite:///./data/crypto_detection.db"

    # Ghidra
    ghidra_home: str = "/opt/ghidra"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # Model
    model_path: str = "data/models/crypto_detector.pth"
    feature_dim: int = 512

    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"

    # Logging
    log_level: str = "INFO"

    # Paths
    upload_dir: str = "data/uploads"
    dataset_dir: str = "data/datasets"
    model_dir: str = "data/models"

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
