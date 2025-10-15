# Complete Guide - # CRYPTERA: AI BASED CRYPTOGRAPHIC PRIMITIVES DETECTION IN FIRMWARE

## 📚 Documentation Index

This guide provides a complete roadmap for using the system. Choose your path:

### 🚀 Quick Start (New Users)
→ **[QUICKSTART.md](QUICKSTART.md)** - Get running in 30 minutes

### 📖 Full Documentation
→ **[README.md](README.md)** - System overview and architecture

### 🎓 Training & Evaluation
→ **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training instructions

### 📊 Datasets
→ **[DATASETS.md](DATASETS.md)** - Dataset links and preparation

---

## 🎯 Choose Your Path

### Path 1: Quick Demo (30 minutes)
**Goal**: See the system working ASAP

```bash
# 1. Automated setup
python scripts/quickstart.py

# 2. Start services
docker-compose up -d

# 3. Open http://localhost:3000
```

**Follow**: [QUICKSTART.md](QUICKSTART.md)

---

### Path 2: Research/Development (1 day)
**Goal**: Understand the system, train on real data

**Day 1 Schedule**:

**Morning (3 hours)**
- ✓ Read README.md (30 min)
- ✓ Setup environment (30 min)
- ✓ Download real datasets (1 hour)
- ✓ Extract features (1 hour)

**Afternoon (4 hours)**
- ✓ Train model (2 hours)
- ✓ Evaluate model (30 min)
- ✓ Test with real firmware (1 hour)
- ✓ Review results (30 min)

**Follow**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md) + [DATASETS.md](DATASETS.md)

---

### Path 3: Production Deployment (1 week)
**Goal**: Deploy to production with optimal performance

**Week 1 Schedule**:

**Day 1-2**: Data Collection
- Collect 10,000+ real firmware samples
- Label data accurately
- Validate dataset quality

**Day 3-4**: Training
- Train on large dataset
- Cross-validation
- Hyperparameter tuning

**Day 5**: Evaluation & Testing
- Comprehensive evaluation
- Integration testing
- Performance benchmarking

**Day 6**: Deployment
- Docker containerization
- Database setup (PostgreSQL)
- Load balancing configuration

**Day 7**: Monitoring & Optimization
- Setup monitoring
- Performance optimization
- Documentation

**Follow**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md) + Custom deployment guide

---

## 📋 Complete Workflow

### Phase 1: Setup (30 minutes)

```bash
# 1. Clone/navigate to project
cd "F:\SIH 2025\Project"

# 2. Create environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac
```

---

### Phase 2: Data Preparation (2-24 hours)

**Option A: Synthetic Data (Quick - 10 minutes)**
```bash
python scripts/generate_dataset.py --num-samples 1000
```

**Option B: Real Datasets (Recommended - 2-4 hours)**
```bash
# Download datasets
cd data/datasets
git clone https://github.com/OWASP/IoTGoat.git
git clone https://github.com/ANSSI-FR/crypto-condor.git

# Extract features
cd ../..
python scripts/extract_features.py \ --input data/datasets/IoTGoat/ \ --output data/datasets/features/
```

**See**: [DATASETS.md](DATASETS.md) for all dataset options

---

### Phase 3: Training (1-4 hours)

```bash
# Basic training
python scripts/train_model.py \ --features data/datasets/features/features.npy \ --labels data/datasets/features/labels.npy \ --epochs 50

# Advanced training
python scripts/train_model.py \ --features data/datasets/features/features.npy \  --labels data/datasets/features/labels.npy \ --epochs 100 \ --batch-size 64 \ --learning-rate 0.0001 \ --hidden-dim 512
```

**See**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for optimization tips

---

### Phase 4: Evaluation (15 minutes)

```bash
python scripts/evaluate_model.py \ --model data/models/crypto_detector.pth \ --test-features data/datasets/test/features.npy \ --test-labels data/datasets/test/labels.npy
```

**Expected Output**:
- Accuracy: 85-95%
- Precision: 85-95%
- F1-Score: 85-95%

---

### Phase 5: Deployment (30 minutes)

**Option A: Docker (Recommended)**
```bash
docker-compose up -d
```

**Option B: Manual**
```bash
# Terminal 1: Backend
cd services
uvicorn api.main:app --reload

# Terminal 2: Frontend
cd frontend
npm install && npm start
```

---

### Phase 6: Testing & Usage

**Web Interface**:
1. Open http://localhost:3000
2. Upload firmware
3. View results

**API Testing**:
```bash
# Upload firmware
curl -X POST "http://localhost:8000/api/firmware/upload" \
    -F "file=@firmware.bin"

# Get results
curl http://localhost:8000/api/results/1
```

**CLI Testing**:
```bash
python scripts/test_inference.py \ --firmware data/datasets/test/sample.bin
```

---

## 🔧 Key Scripts Reference

| Script | Purpose | Typical Usage |
|--------|---------|---------------|
| `quickstart.py` | Complete automation | One-time setup |
| `generate_dataset.py` | Create synthetic data | Initial testing |
| `extract_features.py` | Extract from binaries | Data preparation |
| `train_model.py` | Train AI model | Model creation |
| `evaluate_model.py` | Test performance | Validation |
| `test_inference.py` | Single file test | Quick testing |

---

## 📊 Dataset Recommendations

### For Quick Testing
- **Synthetic Dataset**: 1,000 samples
- **Time**: 10 minutes
- **Accuracy**: 85-90%

### For Development
- **IoTGoat** + **Crypto-Condor** + Synthetic
- **Total**: 2,000-5,000 samples
- **Time**: 2-4 hours
- **Accuracy**: 90-93%

### For Production
- **Firmadyne** + Multiple crypto libraries
- **Total**: 10,000+ samples
- **Time**: 1 week
- **Accuracy**: 93-96%

**See**: [DATASETS.md](DATASETS.md) for download links

---

## 🎯 Performance Targets

### Minimum Acceptable
- Accuracy: >80%
- Inference: <500ms
- Training: <4 hours (10K samples)

### Good
- Accuracy: >90%
- Inference: <200ms
- Training: <2 hours (10K samples)

### Excellent
- Accuracy: >95%
- Inference: <100ms
- Training: <1 hour (10K samples)

---

## 🐛 Troubleshooting Quick Reference

| Issue | Solution | Reference |
|-------|----------|-----------|
| Module not found | `pip install -r requirements.txt` | QUICKSTART.md |
| Ghidra not found | Install + set GHIDRA_HOME | TRAINING_GUIDE.md |
| Low accuracy | More data + epochs | TRAINING_GUIDE.md |
| Out of memory | Reduce batch size | QUICKSTART.md |
| Port in use | Change port in .env | README.md |

---

## 📚 Documentation Files

### Core Documentation
- **README.md** - System overview, architecture, installation
- **QUICKSTART.md** - 30-minute getting started guide
- **TRAINING_GUIDE.md** - Complete training instructions
- **DATASETS.md** - Dataset sources and preparation

### Code Documentation
- **services/api/** - API endpoints documentation
- **services/ai-engine/** - AI model documentation
- **services/binary-analyzer/** - Ghidra integration docs
- **frontend/** - React dashboard documentation

---

## 🎓 Learning Path

### Beginner (1 day)
1. Read README.md
2. Follow QUICKSTART.md
3. Test with synthetic data
4. Explore web interface

### Intermediate (1 week)
1. Study TRAINING_GUIDE.md
2. Download real datasets (DATASETS.md)
3. Train on real data
4. Customize for your needs

### Advanced (2+ weeks)
1. Collect proprietary firmware
2. Optimize hyperparameters
3. Deploy to production
4. Integrate with ERP systems

---

## 🚀 Common Use Cases

### Use Case 1: Quick Demo
**Scenario**: Show system capabilities in 30 minutes

```bash
python scripts/quickstart.py
docker-compose up -d
# Open http://localhost:3000
```

---

### Use Case 2: Research Project
**Scenario**: Evaluate for research paper

**Steps**:
1. Download multiple datasets
2. Train with cross-validation
3. Generate comprehensive metrics
4. Create evaluation reports

**Commands**:
```bash
# Download datasets
cd data/datasets
git clone https://github.com/OWASP/IoTGoat.git
git clone https://github.com/ANSSI-FR/crypto-condor.git

# Extract features
python scripts/extract_features.py --input data/datasets/IoTGoat/ --output data/datasets/features/

# Train
python scripts/train_model.py --features data/datasets/features/features.npy --labels data/datasets/features/labels.npy --epochs 100

# Evaluate
python scripts/evaluate_model.py --model data/models/crypto_detector.pth --test-features data/datasets/test/features.npy --test-labels data/datasets/test/labels.npy
```

---

### Use Case 3: Production Deployment
**Scenario**: Deploy for enterprise use

**Requirements**:
- PostgreSQL database
- 10,000+ training samples
- Docker deployment
- Monitoring setup

**See**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md) Production section

---

## 📞 Support Resources

### Documentation
- System overview: README.md
- Quick start: QUICKSTART.md
- Training: TRAINING_GUIDE.md
- Datasets: DATASETS.md

### API Documentation
- Interactive docs: http://localhost:8000/api/docs
- Alternative docs: http://localhost:8000/api/redoc

### Code Examples
- Scripts: `scripts/` directory
- Tests: `tests/` directory
- Services: `services/` directory

---

## ✅ Checklist for Success

### Initial Setup
- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] .env configured
- [ ] Ghidra installed (optional but recommended)

### Data Preparation
- [ ] Dataset downloaded or generated
- [ ] Features extracted
- [ ] Labels verified
- [ ] Train/test split created

### Training
- [ ] Model trained successfully
- [ ] Training metrics logged
- [ ] Model saved
- [ ] Validation accuracy >80%

### Deployment
- [ ] Services start without errors
- [ ] Frontend accessible
- [ ] API responding
- [ ] Test firmware analyzed successfully

### Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] End-to-end test successful
- [ ] Performance meets targets

---

## 🎯 Next Steps After Setup

1. **Improve Model**
   - Collect more data
   - Train for more epochs
   - Tune hyperparameters

2. **Customize**
   - Add custom crypto functions
   - Modify UI
   - Integrate with systems

3. **Deploy**
   - Production database
   - Load balancing
   - Monitoring

4. **Scale**
   - Distributed training
   - Model optimization
   - Caching layer

---

## 📈 Success Metrics

Track these to ensure system effectiveness:

- **Accuracy**: >90% on test set
- **Precision**: >88% per class
- **Recall**: >87% per class
- **Inference Time**: <200ms per firmware
- **System Uptime**: >99%
- **User Satisfaction**: Positive feedback on results

---

## 🏁 Quick Command Reference

```bash
# Setup
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt

# Quick start
python scripts/quickstart.py

# Manual workflow
python scripts/generate_dataset.py --num-samples 1000
python scripts/extract_features.py --input data/datasets/synthetic/firmware --output data/datasets/features --labels data/datasets/synthetic/metadata.json
python scripts/train_model.py --features data/datasets/features/features.npy --labels data/datasets/features/labels.npy
python scripts/evaluate_model.py --model data/models/crypto_detector.pth --test-features data/datasets/test/features.npy --test-labels data/datasets/test/labels.npy

# Deploy
docker-compose up -d

# Test
python scripts/test_inference.py --firmware data/datasets/test/sample.bin
```

---

## 🎓 Conclusion

You now have everything needed to:
- ✅ Set up the system
- ✅ Train models
- ✅ Evaluate performance
- ✅ Deploy to production
- ✅ Customize for your needs

**Start with**: [QUICKSTART.md](QUICKSTART.md) for immediate results!

**Questions?** Check the specific guides above or review the code documentation.

**Good luck! 🚀**
