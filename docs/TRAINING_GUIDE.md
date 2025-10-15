# Complete Training, Evaluation & Testing Guide

## Table of Contents
1. [Dataset Preparation](#dataset-preparation)
2. [Training the Model](#training-the-model)
3. [Evaluation](#evaluation)
4. [Testing](#testing)
5. [Running the System](#running-the-system)
6. [Troubleshooting](#troubleshooting)

---

## Dataset Preparation

### Real-World Datasets (Recommended)

#### 1. **Firmware Analysis Dataset**
- **Source**: [IoT Firmware Dataset](https://github.com/OWASP/IoTGoat)
- **Description**: Collection of IoT firmware samples with various architectures
- **Download**:
  ```bash
  git clone https://github.com/OWASP/IoTGoat.git
  ```

#### 2. **Cryptographic Binary Dataset**
- **Source**: [Crypto Benchmark Suite](https://github.com/ANSSI-FR/crypto-condor)
- **Description**: Cryptographic implementations for testing
- **Download**:
  ```bash
  git clone https://github.com/ANSSI-FR/crypto-condor.git
  ```

#### 3. **Binary Analysis Datasets**
- **Source**: [DARPA CGC Binaries](https://github.com/CyberGrandChallenge/samples)
- **Description**: Challenge binaries with known vulnerabilities and crypto functions
- **Download**:
  ```bash
  git clone https://github.com/CyberGrandChallenge/samples.git
  ```

#### 4. **OpenSSL Compiled Binaries**
- **Source**: Build from source
- **Description**: Known crypto implementations
- **Download**:
  ```bash
  git clone https://github.com/openssl/openssl.git
  cd openssl
  ./Configure
  make
  # Binaries will be in apps/ directory
  ```

#### 5. **IoT Security Dataset**
- **Source**: [Firmadyne Dataset](https://github.com/firmadyne/firmadyne)
- **Description**: Large collection of firmware images
- **Download**:
  ```bash
  git clone https://github.com/firmadyne/firmadyne.git
  ```

### Generate Sample Dataset (For Testing)

Run the sample dataset generator:

```bash
python scripts/generate_dataset.py --output data/datasets/samples --num-samples 1000
```

This creates annotated firmware samples with known crypto functions.

---

## Step-by-Step Setup

### 1. Environment Setup

```bash
# Navigate to project directory
cd "F:\SIH 2025\Project"

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac

# Edit .env with your configuration
```

### 2. Install Ghidra

```bash
# Download Ghidra from https://ghidra-sre.org/
# Extract to C:\ghidra (Windows) or /opt/ghidra (Linux)

# Set environment variable
# Windows:
set GHIDRA_HOME=C:\ghidra

# Linux/Mac:
export GHIDRA_HOME=/opt/ghidra

# Or add to .env file
echo "GHIDRA_HOME=C:\ghidra" >> .env
```

### 3. Database Setup

**Option A: SQLite (Development)**
```bash
# Already configured in .env.example
# Database will be created automatically at: data/crypto_detection.db
```

**Option B: PostgreSQL (Production)**
```bash
# Install PostgreSQL
# Create database
createdb crypto_detection

# Update .env
DATABASE_URL=postgresql://user:password@localhost:5432/crypto_detection

# Run migrations
cd services
alembic upgrade head
```

---

## Training the Model

### Step 1: Prepare Training Data

```bash
# Download real datasets (choose one or more)
cd data/datasets

# Option 1: IoT Firmware
git clone https://github.com/OWASP/IoTGoat.git

# Option 2: Generate sample data
cd ../..
python scripts/generate_dataset.py --num-samples 1000

# Option 3: Use your own firmware samples
# Place them in: data/datasets/firmware/
```

### Step 2: Analyze Firmware and Extract Features

```bash
# Run feature extraction on your dataset
python scripts/extract_features.py \ --input data/datasets/firmware/ \ --output data/datasets/features/ \ --labels data/datasets/labels.json
```

This will:
1. Disassemble binaries using Ghidra
2. Extract features from each binary
3. Create feature vectors and labels

### Step 3: Train the Model

```bash
# Basic training
python scripts/train_model.py \ --features data/datasets/features/features.npy \ --labels data/datasets/features/labels.npy \ --epochs 50 \ --batch-size 32

# Advanced training with custom parameters
python scripts/train_model.py \ --features data/datasets/features/features.npy \ --labels data/datasets/features/labels.npy \ --epochs 100 \ --batch-size 64 \ --learning-rate 0.001 \ --hidden-dim 256 \ --validation-split 0.2 \ --model-output data/models/crypto_detector_custom.pth
```

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--hidden-dim`: Hidden layer dimension (default: 256)
- `--validation-split`: Validation split ratio (default: 0.2)

**Expected Output:**
```
Epoch 1/50 - Train Loss: 2.3456, Train Acc: 0.4521 - Val Loss: 2.1234, Val Acc: 0.5123
Epoch 2/50 - Train Loss: 2.0123, Train Acc: 0.5234 - Val Loss: 1.9876, Val Acc: 0.5567
...
Epoch 50/50 - Train Loss: 0.2345, Train Acc: 0.9234 - Val Loss: 0.3456, Val Acc: 0.8976
Best model saved!
Training complete!
```

### Step 4: Monitor Training

```bash
# View training progress
tensorboard --logdir=data/logs

# Open browser: http://localhost:6006
```

---

## Evaluation

### Step 1: Evaluate on Test Set

```bash
# Run evaluation
python scripts/evaluate_model.py \ --model data/models/crypto_detector.pth \ --test-features data/datasets/test/features.npy \ --test-labels data/datasets/test/labels.npy
```

**Output:**
```
=== Model Evaluation Results ===
Accuracy: 0.9234
Precision: 0.9123
Recall: 0.9345
F1-Score: 0.9234

Per-Class Performance:
  AES:    Precision: 0.95, Recall: 0.93, F1: 0.94
  RSA:    Precision: 0.92, Recall: 0.94, F1: 0.93
  SHA256: Precision: 0.91, Recall: 0.92, F1: 0.91
  ...
```

### Step 2: Cross-Validation

```bash
python scripts/cross_validate.py \ --features data/datasets/features/features.npy \ --labels data/datasets/features/labels.npy \ --folds 5
```

### Step 3: Generate Confusion Matrix

```bash
python scripts/generate_metrics.py \ --model data/models/crypto_detector.pth \ --test-data data/datasets/test/ \ --output reports/evaluation_report.pdf
```

---

## Testing

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_binary_analyzer.py -v

# Run with coverage
pytest tests/unit/ --cov=services --cov-report=html
```

### Integration Tests

```bash
# Test end-to-end workflow
pytest tests/integration/ -v

# Test API endpoints
pytest tests/integration/test_api_endpoints.py -v
```

### Manual Testing

```bash
# Test single firmware analysis
python scripts/test_analysis.py \ --firmware data/datasets/test_samples/sample.bin \ --architecture x86_64
```

---

## Running the System

### Method 1: Docker (Recommended for Production)

```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Access Points:**
- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/api/docs

### Method 2: Manual Startup (Development)

**Windows:**
```bash
scripts\start.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

**Or manually:**

```bash
# Terminal 1: Start Backend
cd services
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Frontend
cd frontend
npm install
npm start
```

---

## Complete Workflow Example

### 1. Initial Setup
```bash
# Clone or navigate to project
cd "F:\SIH 2025\Project"

# Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Install Ghidra
# Download from https://ghidra-sre.org/
# Extract to C:\ghidra
set GHIDRA_HOME=C:\ghidra
```

### 2. Generate/Collect Dataset
```bash
# Generate sample dataset
python scripts/generate_dataset.py --num-samples 1000

# Or download real datasets
cd data/datasets
git clone https://github.com/OWASP/IoTGoat.git
```

### 3. Extract Features
```bash
python scripts/extract_features.py \ --input data/datasets/IoTGoat/ \ --output data/datasets/features/
```

### 4. Train Model
```bash
python scripts/train_model.py \ --features data/datasets/features/features.npy \ --labels data/datasets/features/labels.npy \ --epochs 50
```

### 5. Evaluate Model
```bash
python scripts/evaluate_model.py \ --model data/models/crypto_detector.pth \ --test-features data/datasets/test/features.npy \ --test-labels data/datasets/test/labels.npy
```

### 6. Start System
```bash
# Using Docker
docker-compose up -d

# Or manual
scripts\start.bat  # Windows
```

### 7. Test via UI
1. Open http://localhost:3000
2. Upload firmware file
3. Wait for analysis
4. View results with XAI explanations

---

## Performance Benchmarks

### Expected Training Time
- 1000 samples: ~10-15 minutes (CPU), ~2-3 minutes (GPU)
- 10000 samples: ~1-2 hours (CPU), ~15-20 minutes (GPU)

### Expected Inference Time
- Single firmware: 2-5 minutes (includes disassembly)
- AI inference only: <100ms per sample

### Resource Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 10GB for system, 50GB+ for large datasets
- **CPU**: Multi-core recommended for parallel processing
- **GPU**: Optional, but speeds up training 5-10x

---

## Troubleshooting

### Issue: Ghidra Not Found
```bash
# Solution: Set GHIDRA_HOME
export GHIDRA_HOME=/path/to/ghidra  # Linux/Mac
set GHIDRA_HOME=C:\ghidra           # Windows
```

### Issue: Model Not Loading
```bash
# Solution: Check model file exists
ls data/models/crypto_detector.pth

# If missing, train model first
python scripts/train_model.py --features data/datasets/features/features.npy --labels data/datasets/features/labels.npy
```

### Issue: Out of Memory During Training
```bash
# Solution: Reduce batch size
python scripts/train_model.py --batch-size 16

# Or use gradient accumulation
python scripts/train_model.py --batch-size 8 --accumulation-steps 4
```

### Issue: Low Accuracy
```bash
# Solutions:
# 1. More training data
# 2. More epochs
# 3. Adjust learning rate
# 4. Data augmentation
python scripts/train_model.py --epochs 100 --learning-rate 0.0001
```

### Issue: API Not Responding
```bash
# Check if API is running
curl http://localhost:8000/health

# Check logs
docker-compose logs api

# Restart
docker-compose restart api
```

---

## Advanced Features

### Fine-tuning Pre-trained Model
```bash
python scripts/fine_tune.py \ --pretrained data/models/crypto_detector.pth \ --new-data data/datasets/new_samples/ \ --epochs 20
```

### Batch Processing
```bash
python scripts/batch_analyze.py \ --input-dir data/datasets/firmware/ \ --output-dir data/results/ \ --parallel 4
```

### Export Results
```bash
# Via API
curl -X GET "http://localhost:8000/api/results/export/1?format=json"

# Via CLI
python scripts/export_results.py \ --firmware-id 1 \ --format pdf \ --output reports/analysis_report.pdf
```

---

## Next Steps

1. ✅ Setup environment and dependencies
2. ✅ Prepare/download datasets
3. ✅ Extract features from firmware
4. ✅ Train the model
5. ✅ Evaluate performance
6. ✅ Start the system
7. ✅ Test with real firmware
8. 🎯 Deploy to production
9. 🎯 Monitor and improve

---

## Support & Resources

- **Documentation**: See README.md
- **API Docs**: http://localhost:8000/api/docs
- **Issue Tracker**: Create issues for bugs
- **Training Logs**: data/logs/
- **Model Checkpoints**: data/models/

**Happy Training! 🚀**
