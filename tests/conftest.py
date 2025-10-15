"""
Pytest Configuration and Fixtures
"""

import pytest
import os
import sys
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sample_binary():
    """Create a sample binary file for testing"""
    # Simple ELF header + data
    elf_header = b"\x7fELF\x02\x01\x01\x00"
    data = b"\x00" * 1024 + b"AES" + b"\x00" * 1024
    return elf_header + data


@pytest.fixture
def sample_binary_file(test_data_dir, sample_binary):
    """Create a sample binary file on disk"""
    filepath = os.path.join(test_data_dir, "sample.bin")
    with open(filepath, "wb") as f:
        f.write(sample_binary)
    return filepath


@pytest.fixture
def sample_features():
    """Generate sample feature vectors"""
    return np.random.randn(100, 512).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Generate sample labels"""
    return np.random.randint(0, 10, size=100).astype(np.int64)


@pytest.fixture
def sample_disassembly_result():
    """Sample disassembly result structure"""
    return {
        "binary_path": "/path/to/binary.bin",
        "architecture": "x86_64",
        "functions": [
            {"name": "main", "address": "0x1000", "size": 256},
            {"name": "encrypt_aes", "address": "0x2000", "size": 512},
            {"name": "hash_sha256", "address": "0x3000", "size": 128}
        ],
        "strings": [
            {"value": "AES-256-CBC", "address": "0x4000"},
            {"value": "SHA256", "address": "0x5000"}
        ],
        "imports": ["libc.so", "libcrypto.so"],
        "crypto_patterns": [
            {"type": "function_name", "keyword": "aes", "function": "encrypt_aes"},
            {"type": "string", "keyword": "sha", "value": "SHA256"}
        ],
        "metadata": {
            "file_size": 8192,
            "file_hash": "abc123def456"
        }
    }


@pytest.fixture
def mock_firmware_data():
    """Mock firmware upload data"""
    return {
        "id": 1,
        "filename": "test_firmware.bin",
        "file_hash": "abc123",
        "file_size": 8192,
        "architecture": "x86_64",
        "status": "uploaded"
    }


@pytest.fixture
def mock_analysis_result():
    """Mock analysis result"""
    return {
        "functions": [
            {"type": "AES", "confidence": 0.95, "rank": 1},
            {"type": "SHA256", "confidence": 0.87, "rank": 2},
            {"type": "RSA", "confidence": 0.72, "rank": 3}
        ],
        "confidences": {
            "AES": 0.95,
            "SHA256": 0.87,
            "RSA": 0.72
        },
        "explanations": {
            "method": "Integrated Gradients",
            "feature_importance": [
                {"feature_index": 10, "importance": 0.85, "contribution": "positive"},
                {"feature_index": 25, "importance": 0.72, "contribution": "positive"}
            ],
            "summary": "Top features contributing to prediction"
        },
        "metadata": {
            "model_version": "1.0.0",
            "device": "cpu"
        }
    }


@pytest.fixture(scope="session")
def test_database_url():
    """Test database URL"""
    return "sqlite:///./test_crypto_detection.db"


# Cleanup
def pytest_sessionfinish(session, exitstatus):
    """Cleanup after all tests"""
    # Remove test database if exists
    if os.path.exists("test_crypto_detection.db"):
        os.remove("test_crypto_detection.db")
