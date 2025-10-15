# Multi-stage Dockerfile for Cryptographic Function Detection System

# Stage 1: Backend API Service
FROM python:3.10-slim as backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    file \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/ ./services/
COPY config/ ./config/

# Create data directories
RUN mkdir -p data/uploads data/datasets data/models data/ghidra_projects

# Expose API port
EXPOSE 8000

# Run API service
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Stage 2: Frontend Dashboard
FROM node:18-alpine as frontend

WORKDIR /app

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy frontend code
COPY frontend/ .

# Build frontend
RUN npm run build

# Expose frontend port
EXPOSE 3000

# Serve frontend
CMD ["npm", "start"]
