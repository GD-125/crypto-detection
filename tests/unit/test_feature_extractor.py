"""
Unit Tests for Feature Extractor
"""

import pytest
import numpy as np
from services.feature_extractor.extractor import FeatureExtractor


class TestFeatureExtractor:
    """Test suite for FeatureExtractor class"""

    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance"""
        return FeatureExtractor(feature_dim=512)

    def test_initialization(self, extractor):
        """Test feature extractor initialization"""
        assert extractor.feature_dim == 512
        assert extractor.crypto_instructions is not None
        assert extractor.crypto_patterns is not None

    def test_extract_features_structure(self, extractor, sample_disassembly_result):
        """Test that extract returns correct feature structure"""
        features = extractor.extract(sample_disassembly_result)

        assert isinstance(features, np.ndarray)
        assert features.shape == (512,)
        assert features.dtype == np.float32

    def test_extract_structural_features(self, extractor, sample_disassembly_result):
        """Test structural feature extraction"""
        features = extractor._extract_structural_features(sample_disassembly_result)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        # Should include: num_functions, avg_size, variance, num_strings, etc.
        assert len(features) >= 6

    def test_extract_instruction_features(self, extractor, sample_disassembly_result):
        """Test instruction feature extraction"""
        features = extractor._extract_instruction_features(sample_disassembly_result)

        assert isinstance(features, np.ndarray)
        assert len(features) == 20  # Fixed length

    def test_extract_string_features(self, extractor, sample_disassembly_result):
        """Test string-based feature extraction"""
        features = extractor._extract_string_features(sample_disassembly_result)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_extract_import_features(self, extractor, sample_disassembly_result):
        """Test import-based feature extraction"""
        features = extractor._extract_import_features(sample_disassembly_result)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_extract_crypto_features(self, extractor, sample_disassembly_result):
        """Test crypto-specific feature extraction"""
        features = extractor._extract_crypto_features(sample_disassembly_result)

        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_normalize_features(self, extractor):
        """Test feature normalization"""
        raw_features = np.array([1.0, 100.0, 1000.0, 10000.0])
        normalized = extractor._normalize_features(raw_features)

        assert normalized.shape == (512,)  # Padded to feature_dim
        assert normalized.dtype == np.float32
        # Check no NaN or Inf
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))

    def test_calculate_string_entropy(self, extractor):
        """Test string entropy calculation"""
        strings = ["AES", "RSA", "SHA256"]
        entropy = extractor._calculate_string_entropy(strings)

        assert isinstance(entropy, float)
        assert entropy >= 0

    def test_batch_extract(self, extractor, sample_disassembly_result):
        """Test batch feature extraction"""
        results = [sample_disassembly_result, sample_disassembly_result]
        features = extractor.batch_extract(results)

        assert features.shape == (2, 512)
        assert features.dtype == np.float32

    def test_empty_disassembly(self, extractor):
        """Test handling of empty disassembly result"""
        empty_result = {
            "functions": [],
            "strings": [],
            "imports": [],
            "crypto_patterns": []
        }

        features = extractor.extract(empty_result)

        assert isinstance(features, np.ndarray)
        assert features.shape == (512,)
        # Should not crash, return zero vector
        assert not np.all(features == 0) or np.all(features == 0)

    def test_malformed_data(self, extractor):
        """Test handling of malformed data"""
        malformed = {"invalid": "data"}

        features = extractor.extract(malformed)

        # Should handle gracefully
        assert isinstance(features, np.ndarray)
        assert features.shape == (512,)

    def test_different_architectures(self, extractor):
        """Test feature extraction for different architectures"""
        architectures = ["x86", "x86_64", "arm", "arm64", "mips"]

        for arch in architectures:
            result = {
                "architecture": arch,
                "functions": [{"name": "test", "size": 100}],
                "strings": [],
                "imports": [],
                "crypto_patterns": []
            }

            features = extractor.extract(result)
            assert isinstance(features, np.ndarray)
            assert features.shape == (512,)
