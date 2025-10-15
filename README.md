# ERP-Integrated Cryptographic Function Detection System

A comprehensive system for detecting cryptographic functions in firmware binaries using AI-powered analysis, multi-architecture binary disassembly, and explainable AI.

## Overview

This system provides an end-to-end solution for analyzing firmware binaries to detect cryptographic implementations. It combines:

- **Multi-Architecture Binary Analysis** using Ghidra
- **Deep Learning Detection** with PyTorch
- **Explainable AI (XAI)** using Captum for transparent decision-making
- **RESTful API** built with FastAPI
- **Interactive Dashboard** for firmware upload, analysis, and results visualization
- **Containerized Deployment** with Docker for scalability

## Features

- **Firmware Upload & Management**: Upload firmware binaries in various formats (ELF, BIN, HEX, FW)
- **Multi-ISA Support**: Automatic detection and analysis of x86, ARM, MIPS, PowerPC architectures
- **AI-Powered Detection**: Deep learning model trained to identify cryptographic functions
- **Explainable Results**: Understand why the AI made specific predictions with feature importance analysis
- **Real-time Analysis**: Background task processing with status monitoring
- **Results Visualization**: Interactive dashboard showing detected crypto functions with confidence scores
- **Batch Processing**: Analyze multiple firmware files simultaneously
- **Export Capabilities**: Export results in JSON, CSV, and PDF formats

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Dashboard                       │
│                    (React + REST API)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                     FastAPI Backend                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Firmware    │  │   Analysis   │  │   Results    │     │
│  │  Management  │  │  Controller  │  │   Manager    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────┬────────────────┬────────────────┬─────────────┘
             │                │                │
    ┌────────▼────────┐  ┌───▼──────┐  ┌─────▼─────────┐
    │  Binary Analyzer │  │ Feature  │  │  AI Engine    │
    │    (Ghidra)      │  │Extractor │  │  (PyTorch)    │
    └──────────────────┘  └──────────┘  └───────────────┘
```

## Directory Structure

```
Project/
├── services/
│   ├── api/                    # FastAPI backend service
│   │   ├── main.py            # Main application entry
│   │   ├── routes/            # API route handlers
│   │   │   ├── firmware.py    # Firmware upload/management
│   │   │   ├── analysis.py    # Analysis triggering
│   │   │   ├── results.py     # Results retrieval
│   │   │   └── dashboard.py   # Dashboard statistics
│   │   ├── models/            # Database and Pydantic models
│   │   │   ├── models.py      # SQLAlchemy ORM models
│   │   │   └── schemas.py     # Pydantic schemas
│   │   └── database.py        # Database configuration
│   ├── ai-engine/             # AI/ML inference engine
│   │   ├── inference.py       # CryptoDetector with XAI
│   │   └── trainer.py         # Model training utilities
│   ├── binary-analyzer/       # Ghidra integration
│   │   └── analyzer.py        # Binary disassembly
│   └── feature-extractor/     # Feature extraction
│       └── extractor.py       # ISA-agnostic features
├── frontend/                  # React dashboard
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── Dashboard.jsx
│   │   │   ├── FirmwareUpload.jsx
│   │   │   ├── AnalysisResults.jsx
│   │   │   └── Navigation.jsx
│   │   ├── services/          # API client
│   │   │   └── api.js
│   │   └── App.jsx            # Main app component
│   ├── public/
│   └── package.json
├── data/                      # Data storage
│   ├── uploads/              # Uploaded firmware files
│   ├── datasets/             # Training datasets
│   ├── models/               # Trained model weights
│   └── ghidra_projects/      # Ghidra analysis projects
├── config/                   # Configuration
│   └── config.py             # Application settings
├── docker/                   # Docker configuration
├── tests/                    # Test suites
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
├── scripts/                  # Utility scripts
├── docs/                     # Documentation
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker build file
├── docker-compose.yml       # Docker orchestration
├── .env.example             # Environment variables template
└── README.md                # This file
```

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (for containerized deployment)
- Ghidra 10.0+ (for binary analysis)
- PostgreSQL 15+ (for production) or SQLite (for development)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Project
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install Ghidra**
   - Download from https://ghidra-sre.org/
   - Extract to `/opt/ghidra` or set `GHIDRA_HOME` environment variable

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Set up frontend**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

6. **Create data directories**
   ```bash
   mkdir -p data/uploads data/datasets data/models data/ghidra_projects
   ```

### Docker Deployment

1. **Build and start all services**
   ```bash
   docker-compose up -d
   ```

2. **Access the application**
   - Frontend Dashboard: http://localhost:3000
   - API Documentation: http://localhost:8000/api/docs
   - API Alternative Docs: http://localhost:8000/api/redoc

3. **Stop services**
   ```bash
   docker-compose down
   ```

## Usage

### Starting the Backend API

```bash
# Development mode
cd services
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Or use the configured command
python -m uvicorn services.api.main:app --reload
```

### Starting the Frontend

```bash
cd frontend
npm start
```

The frontend will be available at http://localhost:3000

### API Endpoints

#### Firmware Management
- `POST /api/firmware/upload` - Upload firmware binary
- `GET /api/firmware/list` - List all firmware files
- `GET /api/firmware/{id}` - Get firmware details
- `DELETE /api/firmware/{id}` - Delete firmware

#### Analysis
- `POST /api/analysis/start/{firmware_id}` - Start analysis
- `GET /api/analysis/status/{firmware_id}` - Check analysis status
- `POST /api/analysis/batch` - Batch analysis

#### Results
- `GET /api/results/{firmware_id}` - Get analysis results
- `GET /api/results/list/all` - List all results
- `GET /api/results/export/{firmware_id}` - Export results

#### Dashboard
- `GET /api/dashboard/stats` - Get statistics
- `GET /api/dashboard/recent-activity` - Get recent activity
- `GET /api/dashboard/crypto-functions` - Get crypto function statistics

### Using the Dashboard

1. **Upload Firmware**
   - Navigate to "Upload" page
   - Select firmware binary file
   - Choose architecture (or use auto-detect)
   - Click "Upload and Analyze"

2. **Monitor Analysis**
   - View progress on Dashboard
   - Check status of analyzing firmware
   - Results appear automatically when complete

3. **View Results**
   - Click on analyzed firmware
   - See detected cryptographic functions
   - Review confidence scores
   - Examine XAI explanations

## Model Training

To train the AI model with your own dataset:

```python
from services.ai_engine.trainer import ModelTrainer
import numpy as np

# Load your training data
features = np.load('data/datasets/features.npy')
labels = np.load('data/datasets/labels.npy')

# Initialize trainer
trainer = ModelTrainer(
    input_dim=512,
    hidden_dim=256,
    num_classes=10
)

# Train model
history = trainer.train(
    features=features,
    labels=labels,
    epochs=50,
    batch_size=32
)

# Save training history
trainer.save_history()
```

## Configuration

### Environment Variables

Edit `.env` file to configure:

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/crypto_detection

# Ghidra
GHIDRA_HOME=/opt/ghidra

# API
API_HOST=0.0.0.0
API_PORT=8000

# Model
MODEL_PATH=data/models/crypto_detector.pth
FEATURE_DIM=512
```

### Supported Architectures

- x86 (32-bit)
- x86_64 (64-bit)
- ARM (32-bit)
- ARM64 (AArch64)
- MIPS
- PowerPC

### Detected Crypto Functions

The system can detect:
- AES (Advanced Encryption Standard)
- RSA (Rivest-Shamir-Adleman)
- SHA-256, SHA-512 (Secure Hash Algorithms)
- DES, 3DES (Data Encryption Standard)
- ECC (Elliptic Curve Cryptography)
- HMAC (Hash-based Message Authentication Code)
- MD5 (Message Digest 5)

## Testing

### Run Unit Tests
```bash
pytest tests/unit/
```

### Run Integration Tests
```bash
pytest tests/integration/
```

### Run All Tests with Coverage
```bash
pytest --cov=services tests/
```

## Performance

- **Analysis Time**: 2-5 minutes per firmware (depends on size)
- **Inference Time**: < 100ms per binary
- **Supported File Size**: Up to 100MB per firmware
- **Concurrent Analysis**: 10 simultaneous analyses (configurable)

## Security Considerations

- All uploaded files are scanned and validated
- SQL injection protection via SQLAlchemy ORM
- CORS configuration for production deployment
- Secure credential storage with environment variables
- Input validation using Pydantic models

## Troubleshooting

### Ghidra Not Found
```bash
export GHIDRA_HOME=/path/to/ghidra
```

### Database Connection Error
- Check PostgreSQL is running
- Verify DATABASE_URL in .env
- For development, use SQLite: `DATABASE_URL=sqlite:///./data/crypto_detection.db`

### Model Not Found Warning
- The system will run with an untrained model
- Train your own model or download pre-trained weights
- Place model file at `data/models/crypto_detector.pth`

### Port Already in Use
```bash
# Change port in .env
API_PORT=8001

# Or use docker-compose port mapping
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Ghidra by NSA for binary analysis capabilities
- PyTorch team for the deep learning framework
- Captum for explainable AI tools
- FastAPI for the excellent web framework

## Contact & Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Email: m.guruchandhran@gmail.com

## Citation

If you use this system in your research, please cite:

```bibtex
@software{crypto_detection_system,
  title={ERP-Integrated Cryptographic Function Detection System},
  author={DevQueens},
  year={2025},
  url={https://github.com/GD-125/crypto-detection}
}
```

---

**Built with ❤️ for SIH 2025**
