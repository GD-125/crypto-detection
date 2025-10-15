"""
Main FastAPI Application for ERP-Integrated Cryptographic Function Detection System
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
import logging
from datetime import datetime
import os

from .routes import firmware, analysis, results, dashboard
from .models import schemas
from .database import engine, Base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Cryptographic Function Detection API",
    description="ERP-integrated system for detecting cryptographic functions in firmware binaries",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(firmware.router, prefix="/api/firmware", tags=["firmware"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(results.router, prefix="/api/results", tags=["results"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Cryptographic Function Detection API",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "running",
            "database": "connected",
            "ai_engine": "ready",
            "binary_analyzer": "ready"
        }
    }

@app.on_event("startup")
async def startup_event():
    """Actions to perform on application startup"""
    logger.info("Starting Cryptographic Function Detection API...")
    logger.info("Initializing microservices...")

    # Create necessary directories
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/datasets", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)

    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Actions to perform on application shutdown"""
    logger.info("Shutting down Cryptographic Function Detection API...")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
