@echo off

echo ==========================================
echo Cryptographic Function Detection System
echo ==========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies if needed
if not exist "venv\installed" (
    echo Installing Python dependencies...
    pip install -r requirements.txt
    echo. > venv\installed
)

REM Create data directories
echo Creating data directories...
if not exist "data\uploads" mkdir data\uploads
if not exist "data\datasets" mkdir data\datasets
if not exist "data\models" mkdir data\models
if not exist "data\ghidra_projects" mkdir data\ghidra_projects

REM Check if .env exists
if not exist ".env" (
    echo Creating .env from .env.example...
    copy .env.example .env
    echo Please edit .env file with your configuration
)

echo.
echo ==========================================
echo Starting services...
echo ==========================================
echo.

REM Start backend API
echo Starting backend API on port 8000...
start "Crypto Detection API" cmd /k "cd services && uvicorn api.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait a bit for API to start
timeout /t 3 /nobreak > nul

REM Start frontend
echo Starting frontend on port 3000...
cd frontend

REM Install frontend dependencies if needed
if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
)

start "Crypto Detection Frontend" cmd /k "npm start"

cd ..

echo.
echo ==========================================
echo Services started successfully!
echo ==========================================
echo API: http://localhost:8000
echo API Docs: http://localhost:8000/api/docs
echo Frontend: http://localhost:3000
echo.
echo Close the terminal windows to stop services
echo ==========================================
echo.

pause
