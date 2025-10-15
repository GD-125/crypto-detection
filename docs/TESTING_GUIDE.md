# Testing Guide

Complete guide for testing the Cryptographic Function Detection System.

## Table of Contents
1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Test Coverage](#test-coverage)
5. [Writing Tests](#writing-tests)
6. [CI/CD Integration](#cicd-integration)

---

## Overview

The test suite includes:
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and workflows
- **Performance Tests**: Test system performance and speed
- **API Tests**: Test API endpoints and workflows

---

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Pytest fixtures and configuration
├── unit/                       # Unit tests
│   ├── __init__.py
│   ├── test_ai_engine.py      # AI/ML model tests
│   ├── test_binary_analyzer.py # Binary analysis tests
│   ├── test_feature_extractor.py # Feature extraction tests
│   └── test_api_routes.py     # API route tests
└── integration/                # Integration tests
    ├── __init__.py
    ├── test_full_pipeline.py  # Complete pipeline tests
    └── test_api_workflow.py   # API workflow tests
```

---

## Running Tests

### Install Test Dependencies

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install test dependencies (already in requirements.txt)
pip install pytest pytest-asyncio pytest-cov httpx
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with detailed output
pytest -vv
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_ai_engine.py

# Specific test class
pytest tests/unit/test_ai_engine.py::TestCryptoDetector

# Specific test function
pytest tests/unit/test_ai_engine.py::TestCryptoDetector::test_detect_with_numpy
```

### Run Tests with Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only API tests
pytest -m api

# Skip slow tests
pytest -m "not slow"
```

### Run Tests in Parallel

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run with 4 workers
pytest -n 4

# Run with auto worker count
pytest -n auto
```

---

## Test Coverage

### Generate Coverage Report

```bash
# Run tests with coverage
pytest --cov=services

# Generate HTML coverage report
pytest --cov=services --cov-report=html

# Open HTML report
# Windows: start htmlcov/index.html
# Linux: xdg-open htmlcov/index.html
# Mac: open htmlcov/index.html
```

### Coverage by Component

```bash
# Coverage for specific module
pytest --cov=services.ai_engine tests/unit/test_ai_engine.py

# Coverage with missing lines
pytest --cov=services --cov-report=term-missing

# Fail if coverage below threshold
pytest --cov=services --cov-fail-under=80
```

### Expected Coverage Targets

| Component | Target Coverage |
|-----------|----------------|
| AI Engine | >85% |
| Feature Extractor | >90% |
| Binary Analyzer | >75% |
| API Routes | >85% |
| Overall | >80% |

---

## Writing Tests

### Unit Test Template

```python
"""
Unit Tests for MyComponent
"""

import pytest
from services.my_module import MyComponent


class TestMyComponent:
    """Test suite for MyComponent"""

    @pytest.fixture
    def component(self):
        """Create component instance"""
        return MyComponent()

    def test_initialization(self, component):
        """Test component initialization"""
        assert component is not None

    def test_basic_functionality(self, component):
        """Test basic functionality"""
        result = component.do_something()
        assert result is not None

    def test_error_handling(self, component):
        """Test error handling"""
        with pytest.raises(ValueError):
            component.invalid_operation()
```

### Integration Test Template

```python
"""
Integration Tests for MyWorkflow
"""

import pytest


class TestMyWorkflow:
    """Test suite for complete workflow"""

    def test_end_to_end_workflow(self):
        """Test complete workflow"""
        # Setup
        data = prepare_data()

        # Execute
        result = run_workflow(data)

        # Verify
        assert result.success
        assert result.output is not None
```

### Using Fixtures

```python
# Available fixtures from conftest.py

def test_with_sample_binary(sample_binary):
    """Use sample binary fixture"""
    assert len(sample_binary) > 0

def test_with_temp_dir(test_data_dir):
    """Use temporary directory"""
    import os
    filepath = os.path.join(test_data_dir, "test.txt")
    # Directory is automatically cleaned up

def test_with_mock_data(mock_firmware_data):
    """Use mock firmware data"""
    assert mock_firmware_data["id"] == 1
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("x86", "x86"),
    ("arm", "arm"),
    ("mips", "mips"),
])
def test_architecture_detection(input, expected):
    """Test with multiple inputs"""
    result = detect_architecture(input)
    assert result == expected
```

---

## Test Examples

### Example 1: Unit Test

```python
def test_feature_extraction():
    """Test feature extraction"""
    from services.feature_extractor.extractor import FeatureExtractor

    extractor = FeatureExtractor(feature_dim=512)

    disassembly_result = {
        "functions": [{"name": "test", "size": 100}],
        "strings": [],
        "imports": []
    }

    features = extractor.extract(disassembly_result)

    assert features.shape == (512,)
```

### Example 2: Integration Test

```python
def test_complete_analysis(sample_binary_file):
    """Test complete analysis pipeline"""
    from services.binary_analyzer.analyzer import BinaryAnalyzer
    from services.feature_extractor.extractor import FeatureExtractor
    from services.ai_engine.inference import CryptoDetector

    # Analyze binary
    analyzer = BinaryAnalyzer()
    result = analyzer.disassemble(sample_binary_file)

    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract(result)

    # Detect crypto functions
    detector = CryptoDetector()
    predictions = detector.detect(features)

    assert len(predictions["functions"]) > 0
```

### Example 3: API Test

```python
def test_firmware_upload():
    """Test firmware upload endpoint"""
    from fastapi.testclient import TestClient
    from services.api.main import app

    client = TestClient(app)

    files = {
        "file": ("test.bin", b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 1000)
    }

    response = client.post("/api/firmware/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "id" in data
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run tests
      run: |
        pytest --cov=services --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Test Commands Reference

### Basic Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=services

# Run specific test
pytest tests/unit/test_ai_engine.py::test_model_initialization

# Run tests matching pattern
pytest -k "test_extract"

# Show print output
pytest -s

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l
```

### Advanced Commands

```bash
# Run only failed tests from last run
pytest --lf

# Run failed tests first, then others
pytest --ff

# Run tests in random order
pytest --random-order

# Generate JUnit XML report
pytest --junit-xml=report.xml

# Run with coverage and HTML report
pytest --cov=services --cov-report=html --cov-report=term
```

---

## Debugging Tests

### Debug Failing Test

```bash
# Run with detailed output
pytest -vv tests/unit/test_ai_engine.py::test_failing

# Show full traceback
pytest --tb=long

# Drop into debugger on failure
pytest --pdb

# Show local variables
pytest -l
```

### Using pytest-sugar

```bash
# Install for better output
pip install pytest-sugar

# Run tests (automatically uses sugar)
pytest
```

---

## Performance Testing

### Benchmark Tests

```python
def test_performance(benchmark):
    """Test performance with pytest-benchmark"""
    from services.feature_extractor.extractor import FeatureExtractor

    extractor = FeatureExtractor()
    disassembly = {...}

    result = benchmark(extractor.extract, disassembly)

    assert result is not None
```

### Load Testing

```bash
# Install locust
pip install locust

# Create locustfile.py
# Run load test
locust -f locustfile.py
```

---

## Best Practices

### 1. Test Isolation
- Each test should be independent
- Use fixtures for setup/teardown
- Don't rely on test execution order

### 2. Test Naming
- Use descriptive names: `test_upload_invalid_file_returns_400`
- Follow pattern: `test_<what>_<condition>_<expected>`

### 3. Test Coverage
- Aim for >80% coverage
- Focus on critical paths
- Test edge cases and errors

### 4. Test Speed
- Keep unit tests fast (<1s each)
- Mark slow tests with `@pytest.mark.slow`
- Use mocking for external dependencies

### 5. Test Organization
- Group related tests in classes
- One test file per module
- Separate unit and integration tests

---

## Troubleshooting

### Issue: Tests Fail to Import Modules

```bash
# Solution: Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

### Issue: Database Errors

```bash
# Solution: Use test database
# Set DATABASE_URL in tests
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
```

### Issue: Tests Pass Locally but Fail in CI

```bash
# Solution: Check dependencies
# Ensure all test dependencies in requirements.txt
# Check Python version matches
```

### Issue: Slow Tests

```bash
# Solution: Run with pytest-xdist
pytest -n auto

# Or skip slow tests
pytest -m "not slow"
```

---

## Test Checklist

Before committing code:

- [ ] All tests pass: `pytest`
- [ ] Coverage >80%: `pytest --cov=services`
- [ ] No warnings: `pytest --strict-warnings`
- [ ] Code formatted: `black .`
- [ ] Linting passes: `flake8`
- [ ] Type checking: `mypy services`

---

## Resources

- **Pytest Documentation**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **FastAPI Testing**: https://fastapi.tiangolo.com/tutorial/testing/
- **PyTorch Testing**: https://pytorch.org/docs/stable/testing.html

---

## Quick Reference

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=services --cov-report=html

# Run specific category
pytest tests/unit/

# Run in parallel
pytest -n auto

# Debug failing test
pytest -vv --pdb tests/unit/test_ai_engine.py::test_failing

# Generate report
pytest --html=report.html --self-contained-html
```

---

**Happy Testing! 🧪**
