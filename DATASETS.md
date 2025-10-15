# Dataset Resources for Cryptographic Function Detection

## Overview

This document provides comprehensive information about datasets suitable for training and testing the cryptographic function detection system.

---

## Real-World Datasets (Recommended)

### 1. **IoT Firmware Dataset - IoTGoat**

**Description**: OWASP IoTGoat is a deliberately insecure firmware based on OpenWrt and designed for learning IoT security and firmware analysis.

**Features**:
- Multiple architectures (ARM, MIPS, x86)
- Known cryptographic implementations
- Well-documented vulnerabilities
- Real-world IoT firmware structure

**Download**:
```bash
cd data/datasets
git clone https://github.com/OWASP/IoTGoat.git
```

**Link**: https://github.com/OWASP/IoTGoat

**Size**: ~500 MB

**Usage**:
```bash
python scripts/extract_features.py \
    --input data/datasets/IoTGoat/ \
    --output data/datasets/iotgoat_features/
```

---

### 2. **Crypto-Condor (ANSSI)**

**Description**: Automated testing tool and dataset for cryptographic implementations with various crypto functions.

**Features**:
- Multiple crypto algorithms (AES, RSA, ECC, etc.)
- Test vectors and implementations
- Different programming languages
- Compliance testing

**Download**:
```bash
cd data/datasets
git clone https://github.com/ANSSI-FR/crypto-condor.git
```

**Link**: https://github.com/ANSSI-FR/crypto-condor

**Size**: ~100 MB

**Usage**:
```bash
python scripts/extract_features.py \
    --input data/datasets/crypto-condor/test-vectors/ \
    --output data/datasets/crypto_condor_features/
```

---

### 3. **DARPA CGC Binaries**

**Description**: Cyber Grand Challenge binaries with various security-related functions including cryptography.

**Features**:
- Multiple architectures
- Annotated binaries
- Known vulnerabilities
- Challenge-quality code

**Download**:
```bash
cd data/datasets
git clone https://github.com/CyberGrandChallenge/samples.git
```

**Link**: https://github.com/CyberGrandChallenge/samples

**Size**: ~2 GB

**Usage**:
```bash
python scripts/extract_features.py \
    --input data/datasets/samples/examples/ \
    --output data/datasets/cgc_features/
```

---

### 4. **Firmadyne Dataset**

**Description**: Large-scale analysis platform for embedded Linux firmware with 20,000+ firmware images.

**Features**:
- Massive dataset (20,000+ images)
- Various vendors and architectures
- Real-world firmware samples
- Automated extraction tools

**Download**:
```bash
cd data/datasets
git clone https://github.com/firmadyne/firmadyne.git

# Download firmware database (requires registration)
# Visit: https://github.com/firmadyne/firmadyne
```

**Link**: https://github.com/firmadyne/firmadyne

**Size**: 100+ GB (full database)

**Usage**:
```bash
# Extract firmware first
cd firmadyne
./download.sh

# Then extract features
python scripts/extract_features.py \
    --input data/datasets/firmadyne/images/ \
    --output data/datasets/firmadyne_features/
```

---

### 5. **OpenSSL Compiled Binaries**

**Description**: Build OpenSSL from source to get binaries with known crypto implementations.

**Features**:
- Gold-standard crypto implementations
- Multiple crypto algorithms
- Well-tested code
- Various compilation options

**Download & Build**:
```bash
cd data/datasets
git clone https://github.com/openssl/openssl.git
cd openssl

# Build for different architectures
./Configure
make

# Binaries will be in apps/ directory
```

**Link**: https://github.com/openssl/openssl

**Size**: ~50 MB (source)

**Usage**:
```bash
python scripts/extract_features.py \
    --input data/datasets/openssl/apps/ \
    --output data/datasets/openssl_features/ \
    --labels data/datasets/openssl_labels.json
```

---

### 6. **Mbed TLS Test Suite**

**Description**: ARM's crypto library with comprehensive test suite.

**Features**:
- Production-quality crypto
- Extensive test coverage
- ARM optimizations
- Various crypto primitives

**Download**:
```bash
cd data/datasets
git clone https://github.com/Mbed-TLS/mbedtls.git
cd mbedtls
make
```

**Link**: https://github.com/Mbed-TLS/mbedtls

**Size**: ~30 MB

---

### 7. **LibTomCrypt**

**Description**: Comprehensive cryptographic library with many algorithms.

**Features**:
- 20+ crypto algorithms
- Public domain code
- Portable implementations
- Test vectors

**Download**:
```bash
cd data/datasets
git clone https://github.com/libtom/libtomcrypt.git
cd libtomcrypt
make
```

**Link**: https://github.com/libtom/libtomcrypt

**Size**: ~10 MB

---

### 8. **Binary Analysis Datasets (Kaggle)**

**Description**: Various malware and binary analysis datasets on Kaggle.

**Links**:
- PE Malware Detection: https://www.kaggle.com/datasets/amauricio/pe-executables-malware-detection
- Binary Classification: https://www.kaggle.com/datasets/jonathanoheix/binary-file-classification

**Download**: Via Kaggle CLI or web interface

---

## Synthetic Dataset Generation

### Generate Sample Dataset

For quick testing and prototyping, use our synthetic dataset generator:

```bash
# Generate 1000 samples
python scripts/generate_dataset.py \
    --output data/datasets/synthetic \
    --num-samples 1000 \
    --min-size 2048 \
    --max-size 8192
```

**Features**:
- Controllable size and complexity
- Known ground truth labels
- Fast generation
- Good for initial testing

**Limitations**:
- Not representative of real firmware
- Simplified crypto patterns
- Use only for testing, not production

---

## Dataset Preparation Workflow

### Complete Pipeline

```bash
# 1. Create directories
mkdir -p data/datasets/firmware
mkdir -p data/datasets/features
mkdir -p data/datasets/test

# 2. Download datasets (choose one or more)
cd data/datasets
git clone https://github.com/OWASP/IoTGoat.git
git clone https://github.com/ANSSI-FR/crypto-condor.git

# 3. Generate synthetic data (optional)
cd ../..
python scripts/generate_dataset.py --num-samples 1000

# 4. Extract features
python scripts/extract_features.py \
    --input data/datasets/synthetic/firmware/ \
    --output data/datasets/features/ \
    --labels data/datasets/synthetic/metadata.json

# 5. Split train/test (automatic in extract_features.py)
# Or manually:
python scripts/split_dataset.py \
    --features data/datasets/features/features.npy \
    --labels data/datasets/features/labels.npy \
    --train-split 0.8
```

---

## Recommended Dataset Combinations

### For Quick Testing (1-2 hours)
```bash
# Generate synthetic data + OpenSSL binaries
python scripts/generate_dataset.py --num-samples 500

# Download OpenSSL
git clone https://github.com/openssl/openssl.git
cd openssl && make

# Total: ~1000 samples
```

### For Research/Development (1 day)
```bash
# Synthetic + IoTGoat + Crypto-Condor
# Total: ~2000-5000 samples
git clone https://github.com/OWASP/IoTGoat.git
git clone https://github.com/ANSSI-FR/crypto-condor.git
python scripts/generate_dataset.py --num-samples 1000
```

### For Production (1 week)
```bash
# Firmadyne + Multiple crypto libraries + Real firmware
# Total: 10,000+ samples
git clone https://github.com/firmadyne/firmadyne.git
# Download firmware database
# Compile multiple crypto libraries
# Collect proprietary firmware (if available)
```

---

## Dataset Statistics

### Recommended Minimum Sizes

| Purpose | Minimum Samples | Recommended |
|---------|----------------|-------------|
| Proof of Concept | 100 | 500 |
| Research/Testing | 500 | 2,000 |
| Development | 2,000 | 10,000 |
| Production | 10,000 | 50,000+ |

### Per-Class Distribution

Aim for balanced classes or at least:
- Minimum 50 samples per class
- Maximum 10:1 class imbalance ratio

---

## Labeling Guidelines

### Manual Labeling

Create `labels.json`:
```json
{
  "samples": [
    {
      "filename": "sample_001.bin",
      "label": 0,
      "crypto_type": "AES",
      "architecture": "x86_64"
    },
    {
      "filename": "sample_002.bin",
      "label": 1,
      "crypto_type": "RSA",
      "architecture": "arm"
    }
  ]
}
```

### Automated Labeling

Use our labeling script:
```bash
python scripts/auto_label.py \
    --input data/datasets/firmware/ \
    --output data/datasets/labels.json \
    --method heuristic
```

---

## Data Augmentation

### Techniques

1. **Architecture Variation**: Compile for different targets
2. **Optimization Levels**: -O0, -O1, -O2, -O3
3. **Compiler Variation**: GCC, Clang, MSVC
4. **Binary Obfuscation**: Strip symbols, pack, obfuscate

```bash
# Example: Compile with different optimizations
gcc -O0 crypto.c -o crypto_O0
gcc -O2 crypto.c -o crypto_O2
gcc -O3 crypto.c -o crypto_O3
```

---

## Quality Assurance

### Validation Checklist

- [ ] All files are valid binaries
- [ ] Labels are correctly assigned
- [ ] No corrupted files
- [ ] Balanced class distribution
- [ ] Train/test split is stratified
- [ ] Features extracted successfully
- [ ] No data leakage between train/test

### Validation Script

```bash
python scripts/validate_dataset.py \
    --features data/datasets/features/features.npy \
    --labels data/datasets/features/labels.npy
```

---

## Storage Requirements

| Dataset | Size | Samples | Time to Process |
|---------|------|---------|-----------------|
| Synthetic (1K) | 10 MB | 1,000 | 5 min |
| IoTGoat | 500 MB | ~500 | 30 min |
| Crypto-Condor | 100 MB | ~200 | 15 min |
| CGC | 2 GB | ~1,000 | 2 hours |
| Firmadyne | 100+ GB | 20,000+ | Days |

---

## Troubleshooting

### Issue: Download Fails
```bash
# Use --depth 1 for faster cloning
git clone --depth 1 https://github.com/OWASP/IoTGoat.git
```

### Issue: Large Dataset Size
```bash
# Extract only specific architectures
python scripts/extract_features.py \
    --input data/datasets/firmware/ \
    --output data/datasets/features/ \
    --arch-filter x86_64,arm
```

### Issue: Processing Too Slow
```bash
# Enable parallel processing
python scripts/extract_features.py \
    --input data/datasets/firmware/ \
    --output data/datasets/features/ \
    --parallel 4
```

---

## Citations

If using these datasets in research, please cite:

**IoTGoat**:
```
OWASP Foundation. IoTGoat. https://github.com/OWASP/IoTGoat
```

**Firmadyne**:
```
D. Chen et al., "Towards Automated Dynamic Analysis for Linux-based Embedded Firmware," NDSS 2016.
```

---

## Next Steps

1. ✅ Choose dataset(s) based on your needs
2. ✅ Download and prepare data
3. ✅ Extract features
4. ✅ Train model (see TRAINING_GUIDE.md)
5. ✅ Evaluate and deploy

For complete training instructions, see: **TRAINING_GUIDE.md**
