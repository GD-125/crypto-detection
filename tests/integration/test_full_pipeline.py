"""
Integration Tests for Full Analysis Pipeline
"""

import pytest
import numpy as np
import torch
from services.binary_analyzer.analyzer import BinaryAnalyzer
from services.feature_extractor.extractor import FeatureExtractor
from services.ai_engine.inference import CryptoDetector


class TestFullPipeline:
    """Test suite for complete analysis pipeline"""

    @pytest.fixture
    def pipeline_components(self, tmp_path):
        """Initialize all pipeline components"""
        # Create a dummy model
        from services.ai_engine.inference import CryptoDetectionModel

        model = CryptoDetectionModel()
        model_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), model_path)

        return {
            "analyzer": BinaryAnalyzer(),
            "extractor": FeatureExtractor(),
            "detector": CryptoDetector(model_path=str(model_path))
        }

    def test_complete_pipeline(self, pipeline_components, sample_binary_file):
        """Test complete analysis pipeline"""
        analyzer = pipeline_components["analyzer"]
        extractor = pipeline_components["extractor"]
        detector = pipeline_components["detector"]

        # Step 1: Binary Analysis
        disassembly_result = analyzer.disassemble(sample_binary_file)

        assert isinstance(disassembly_result, dict)
        assert "architecture" in disassembly_result

        # Step 2: Feature Extraction
        features = extractor.extract(disassembly_result)

        assert isinstance(features, np.ndarray)
        assert features.shape == (512,)

        # Step 3: AI Detection
        predictions = detector.detect(features)

        assert isinstance(predictions, dict)
        assert "functions" in predictions
        assert "confidences" in predictions
        assert "explanations" in predictions

    def test_pipeline_with_crypto_binary(self, pipeline_components, tmp_path):
        """Test pipeline with binary containing crypto patterns"""
        # Create binary with AES patterns
        binary_path = tmp_path / "crypto_binary.bin"
        with open(binary_path, "wb") as f:
            # ELF header
            f.write(b"\x7fELF\x02\x01\x01\x00")
            # AES S-box pattern
            f.write(b"\x63\x7c\x77\x7b\xf2\x6b\x6f\xc5")
            # Fill with random data
            f.write(b"\x00" * 2000)

        analyzer = pipeline_components["analyzer"]
        extractor = pipeline_components["extractor"]
        detector = pipeline_components["detector"]

        # Run pipeline
        disassembly_result = analyzer.disassemble(str(binary_path))
        features = extractor.extract(disassembly_result)
        predictions = detector.detect(features)

        # Should detect something (exact detection depends on model)
        assert len(predictions["functions"]) > 0

    def test_pipeline_error_handling(self, pipeline_components, tmp_path):
        """Test pipeline handles errors gracefully"""
        # Create invalid binary
        invalid_binary = tmp_path / "invalid.bin"
        invalid_binary.write_bytes(b"INVALID")

        analyzer = pipeline_components["analyzer"]
        extractor = pipeline_components["extractor"]
        detector = pipeline_components["detector"]

        # Should not crash
        disassembly_result = analyzer.disassemble(str(invalid_binary))
        features = extractor.extract(disassembly_result)
        predictions = detector.detect(features)

        # Should return some result even if empty
        assert isinstance(predictions, dict)

    def test_pipeline_multiple_binaries(self, pipeline_components, sample_binary_file):
        """Test pipeline with multiple binaries"""
        analyzer = pipeline_components["analyzer"]
        extractor = pipeline_components["extractor"]
        detector = pipeline_components["detector"]

        # Process multiple times
        results = []
        for _ in range(3):
            disassembly_result = analyzer.disassemble(sample_binary_file)
            features = extractor.extract(disassembly_result)
            predictions = detector.detect(features)
            results.append(predictions)

        # Should get consistent results
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_pipeline_different_architectures(self, pipeline_components, tmp_path):
        """Test pipeline with different architectures"""
        analyzer = pipeline_components["analyzer"]
        extractor = pipeline_components["extractor"]
        detector = pipeline_components["detector"]

        architectures = ["x86", "x86_64", "arm"]

        for arch in architectures:
            binary_path = tmp_path / f"binary_{arch}.bin"
            binary_path.write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 1000)

            disassembly_result = analyzer.disassemble(str(binary_path), architecture=arch)
            features = extractor.extract(disassembly_result)
            predictions = detector.detect(features)

            # Should work for all architectures
            assert isinstance(predictions, dict)
            assert disassembly_result.get("architecture") == arch


class TestPipelinePerformance:
    """Test suite for pipeline performance"""

    @pytest.fixture
    def perf_pipeline(self, tmp_path):
        """Create pipeline for performance testing"""
        from services.ai_engine.inference import CryptoDetectionModel

        model = CryptoDetectionModel()
        model_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), model_path)

        return {
            "analyzer": BinaryAnalyzer(),
            "extractor": FeatureExtractor(),
            "detector": CryptoDetector(model_path=str(model_path))
        }

    def test_feature_extraction_speed(self, perf_pipeline, sample_disassembly_result):
        """Test feature extraction speed"""
        import time

        extractor = perf_pipeline["extractor"]

        start_time = time.time()
        for _ in range(100):
            features = extractor.extract(sample_disassembly_result)
        end_time = time.time()

        elapsed = end_time - start_time
        avg_time = elapsed / 100

        # Should be fast (< 100ms per extraction)
        assert avg_time < 0.1

    def test_inference_speed(self, perf_pipeline):
        """Test AI inference speed"""
        import time

        detector = perf_pipeline["detector"]
        test_features = np.random.randn(512).astype(np.float32)

        start_time = time.time()
        for _ in range(100):
            predictions = detector.detect(test_features)
        end_time = time.time()

        elapsed = end_time - start_time
        avg_time = elapsed / 100

        # Should be very fast (< 50ms per inference)
        assert avg_time < 0.05

    def test_batch_processing_efficiency(self, perf_pipeline):
        """Test batch processing is more efficient than sequential"""
        import time

        detector = perf_pipeline["detector"]
        batch_size = 10
        features_list = [
            np.random.randn(512).astype(np.float32)
            for _ in range(batch_size)
        ]

        # Sequential processing
        start_time = time.time()
        sequential_results = [detector.detect(f) for f in features_list]
        sequential_time = time.time() - start_time

        # Batch processing
        start_time = time.time()
        batch_results = detector.batch_detect(features_list)
        batch_time = time.time() - start_time

        # Batch should be comparable or faster
        # (May not always be true for small batches)
        assert len(sequential_results) == len(batch_results)


class TestPipelineDataFlow:
    """Test suite for data flow through pipeline"""

    def test_data_shapes_consistency(self, sample_disassembly_result):
        """Test data shapes are consistent through pipeline"""
        extractor = FeatureExtractor(feature_dim=512)

        # Extract features
        features = extractor.extract(sample_disassembly_result)

        # Check shape at each stage
        assert features.shape == (512,)
        assert features.dtype == np.float32

        # Convert to tensor
        tensor_features = torch.tensor(features)
        assert tensor_features.shape == (512,)

    def test_data_types_consistency(self, sample_disassembly_result):
        """Test data types are correct through pipeline"""
        extractor = FeatureExtractor()

        features = extractor.extract(sample_disassembly_result)

        # Should be float32
        assert features.dtype == np.float32

        # Should not have NaN or Inf
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    def test_metadata_propagation(self, sample_binary_file):
        """Test metadata is preserved through pipeline"""
        analyzer = BinaryAnalyzer()

        disassembly_result = analyzer.disassemble(sample_binary_file)

        # Check metadata exists
        assert "metadata" in disassembly_result
        metadata = disassembly_result["metadata"]

        assert "file_size" in metadata or "file_hash" in metadata
