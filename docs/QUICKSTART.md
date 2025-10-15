# Quick Start Guide

Get up and running with the Cryptographic Function Detection System in **under 30 minutes**!

## Prerequisites

- Python 3.10+
- pip
- 4GB RAM minimum
- 10GB free disk space

## Option 1: Automated Quick Start (Recommended)

### Single Command Setup

```bash
# Navigate to project
cd "F:/SIH 2025/Project"

# Run automated setup
python scripts/quickstart.py --num-samples 1000 --epochs 50
```

This will automatically:
1. ✓ Generate synthetic dataset (1000 samples)
2. ✓ Extract features using Ghidra
3. ✓ Train the AI model (50 epochs)
4. ✓ Evaluate model performance
5. ✓ Prepare system for deployment

**Expected time**: 15-30 minutes (depending on hardware)

---

## Option 2: Manual Step-by-Step

### Step 1: Environment Setup (5 minutes)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Generate Dataset (2 minutes)

```bash
# Generate 1000 synthetic samples
python scripts/generate_dataset.py \
    --output data/datasets/synthetic \
    --num-samples 1000
```

**Output**:
```
Generating 1000 samples...
Generated 100/1000 samples
Generated 200/1000 samples
...
✓ Dataset generation complete!
  Total samples: 1000
  Train samples: 800
  Test samples: 200
```

### Step 3: Extract Features (5-10 minutes)

```bash
# Extract features from binaries
python scripts/extract_features.py \ --input data/datasets/synthetic/firmware \ --output data/datasets/features \ --labels data/datasets/synthetic/metadata.json
```

**Output**:
```
Found 1000 binary files
Extracting features: 100%|████████████| 1000/1000
✓ Feature extraction complete!
  Processed: 1000 files
  Features: (1000, 512)
```

### Step 4: Train Model (10-15 minutes)

```bash
# Train the model
python scripts/train_model.py \ --features data/datasets/features/features.npy \ --labels data/datasets/features/labels.npy \ --epochs 50 \ --batch-size 32
```

**Output**:
```
Epoch 1/50 - Train Loss: 2.3456, Train Acc: 0.4521 - Val Loss: 2.1234, Val Acc: 0.5123
Epoch 2/50 - Train Loss: 2.0123, Train Acc: 0.5234 - Val Loss: 1.9876, Val Acc: 0.5567
...
Epoch 50/50 - Train Loss: 0.2345, Train Acc: 0.9234 - Val Loss: 0.3456, Val Acc: 0.8976
TRAINING COMPLETE!
Final Val Accuracy: 0.8976
```

### Step 5: Evaluate Model (2 minutes)

```bash
# Evaluate on test set
python scripts/evaluate_model.py \ --model data/models/crypto_detector.pth \ --test-features data/datasets/test/features.npy \ --test-labels data/datasets/test/labels.npy
```

**Output**:
```
EVALUATION RESULTS
==================
Accuracy:  0.8976 (89.76%)
Precision: 0.8923
Recall:    0.8945
F1-Score:  0.8934
```

### Step 6: Start the System

**Option A: Using Docker (Recommended)**

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

**Option B: Manual Startup**

```bash
# Terminal 1: Backend API
cd services
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend (new terminal)
cd frontend
npm install
npm start
```

### Step 7: Access the Application

Open your browser:
- **Frontend Dashboard**: http://localhost:3000
- **API Documentation**: http://localhost:8000/api/docs
- **API Alternative Docs**: http://localhost:8000/api/redoc

---

## Testing the System

### Upload Test Firmware

1. Navigate to http://localhost:3000
2. Click "Upload" in the navigation
3. Select a firmware binary file
4. Choose architecture (or use auto-detect)
5. Click "Upload and Analyze"
6. Wait for analysis (2-5 minutes)
7. View results with XAI explanations

### API Testing

```bash
# Health check
curl http://localhost:8000/health

# Upload firmware via API
curl -X POST "http://localhost:8000/api/firmware/upload" \ -F "file=@data/datasets/synthetic/firmware/sample_00001_AES.bin" \ -F "architecture=auto"

# Get analysis results
curl http://localhost:8000/api/results/1
```

---

## Using Real Datasets

### Download IoTGoat Dataset

```bash
cd data/datasets
git clone https://github.com/OWASP/IoTGoat.git

# Extract features
cd ../..
python scripts/extract_features.py \  --input data/datasets/IoTGoat/ \ --output data/datasets/iotgoat_features/

# Train on real data
python scripts/train_model.py \ --features data/datasets/iotgoat_features/features.npy \ --labels data/datasets/iotgoat_features/labels.npy \ --epochs 100 \ --output data/models/crypto_detector_iotgoat.pth
```

### Download Crypto-Condor

```bash
cd data/datasets
git clone https://github.com/ANSSI-FR/crypto-condor.git

# Extract features
cd ../..
python scripts/extract_features.py \ --input data/datasets/crypto-condor/ \ --output data/datasets/crypto_condor_features/
```

For more datasets, see **DATASETS.md**

---

## Complete Workflow Example

### From Zero to Working System

```bash
# 1. Setup environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Quick start (automated)
python scripts/quickstart.py

# 3. Start services
docker-compose up -d

# 4. Open browser
# Navigate to http://localhost:3000

# 5. Upload firmware and analyze!
```

**Total time**: ~30 minutes

---

## Performance Expectations

### With Synthetic Data (1000 samples)

| Metric | Value |
|--------|-------|
| Training Time | 10-15 min (CPU) |
| Model Accuracy | 85-90% |
| Inference Time | <100ms |
| Dataset Size | 10 MB |

### With Real Data (10,000+ samples)

| Metric | Value |
|--------|-------|
| Training Time | 1-2 hours (CPU) |
| Model Accuracy | 90-95% |
| Inference Time | <100ms |
| Dataset Size | 1+ GB |

---

## Troubleshooting

### Issue: Module not found

```bash
# Solution: Install missing packages
pip install -r requirements.txt

# Or specific package
pip install fastapi uvicorn torch
```

### Issue: Port already in use

```bash
# Solution: Change port
# Edit .env file:
API_PORT=8001

# Or kill existing process
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux:
lsof -ti:8000 | xargs kill -9
```

### Issue: Out of memory during training

```bash
# Solution: Reduce batch size
python scripts/train_model.py \ --features data/datasets/features/features.npy \ --labels data/datasets/features/labels.npy \ --batch-size 16  # Reduced from 32
```

### Issue: Model accuracy too low

```bash
# Solutions:
# 1. More training data
python scripts/generate_dataset.py --num-samples 5000

# 2. More epochs
python scripts/train_model.py --epochs 100

# 3. Use real datasets
cd data/datasets
git clone https://github.com/OWASP/IoTGoat.git
```

### Issue: Ghidra not found

```bash
# Solution: Install Ghidra
# 1. Download from https://ghidra-sre.org/
# 2. Extract to C:\ghidra (Windows) or /opt/ghidra (Linux)
# 3. Set environment variable:

# Windows:
set GHIDRA_HOME=C:\ghidra

# Linux/Mac:
export GHIDRA_HOME=/opt/ghidra

# Or add to .env file:
echo "GHIDRA_HOME=C:\ghidra" >> .env
```

---

## Next Steps

### 1. Improve Model Performance

```bash
# Collect more data
cd data/datasets
git clone https://github.com/OWASP/IoTGoat.git
git clone https://github.com/ANSSI-FR/crypto-condor.git

# Re-train with combined data
python scripts/train_model.py \ --features data/datasets/combined/features.npy \ --labels data/datasets/combined/labels.npy \ --epochs 100
```

### 2. Deploy to Production

```bash
# Use PostgreSQL instead of SQLite
# Update .env:
DATABASE_URL=postgresql://user:pass@localhost/crypto_detection

# Run migrations
cd services
alembic upgrade head

# Deploy with Docker
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Customize for Your Needs

- Add custom crypto function detectors
- Train on proprietary firmware
- Integrate with existing ERP systems
- Export results to your format

---

## Verification Checklist

After quick start, verify:

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Dataset generated (data/datasets/synthetic/)
- [ ] Features extracted (data/datasets/features/)
- [ ] Model trained (data/models/crypto_detector.pth)
- [ ] Model evaluated (reports/evaluation_results.json)
- [ ] Services running (docker-compose ps)
- [ ] Frontend accessible (http://localhost:3000)
- [ ] API accessible (http://localhost:8000/api/docs)
- [ ] Test firmware analyzed successfully

---

## Command Reference

### Most Common Commands

```bash
# Generate dataset
python scripts/generate_dataset.py --num-samples 1000

# Extract features
python scripts/extract_features.py --input <dir> --output <dir>

# Train model
python scripts/train_model.py --features <file> --labels <file>

# Evaluate model
python scripts/evaluate_model.py --model <file> --test-features <file> --test-labels <file>

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Getting Help

- **Documentation**: See README.md and TRAINING_GUIDE.md
- **Datasets**: See DATASETS.md
- **API Reference**: http://localhost:8000/api/docs
- **Logs**: docker-compose logs or data/logs/

---

## Success!

If you see this output, you're ready to go:

```
✓ Dataset generation complete!
✓ Feature extraction complete!
✓ Model training complete!
✓ Model evaluation complete!
✓ Services running!

Frontend: http://localhost:3000
API: http://localhost:8000/api/docs
```

**Happy analyzing! 🚀**
