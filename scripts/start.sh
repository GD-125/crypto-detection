#!/bin/bash

# Startup script for Cryptographic Primitives Detection System

echo "=========================================="
echo "Cryptographic Primitives Detection System"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/installed" ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    touch venv/installed
fi

# Create data directories
echo "Creating data directories..."
mkdir -p data/uploads
mkdir -p data/datasets
mkdir -p data/models
mkdir -p data/ghidra_projects

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "Please edit .env file with your configuration"
fi

# Start backend API
echo ""
echo "Starting backend API on port 8000..."
cd services
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait a bit for API to start
sleep 3

# Start frontend
echo "Starting frontend on port 3000..."
cd ../frontend

# Install frontend dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

npm start &
FRONTEND_PID=$!

echo ""
echo "=========================================="
echo "Services started successfully!"
echo "=========================================="
echo "API: http://localhost:8000"
echo "API Docs: http://localhost:8000/api/docs"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=========================================="

# Wait for user interrupt
trap "kill $API_PID $FRONTEND_PID; exit" INT
wait
