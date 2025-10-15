"""
Unit Tests for AI Engine
"""

import pytest
import torch
import numpy as np
from services.ai_engine.inference import CryptoDetector, CryptoDetectionModel
from services.ai_engine.trainer import ModelTrainer, CryptoDataset


class TestCryptoDetectionModel:
    """Test suite for CryptoDetectionModel"""

    @pytest.fixture
    def model(self):
        """Create model instance"""
        return CryptoDetectionModel(input_dim=512, hidden_dim=256, num_classes=10)

    def test_model_initialization(self, model):
        """Test model initialization"""
        assert model.input_dim == 512
        assert model.hidden_dim == 256
        assert model.num_classes == 10

    def test_forward_pass(self, model):
        """Test forward pass"""
        batch_size = 16
        x = torch.randn(batch_size, 512)

        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_single_sample(self, model):
        """Test single sample prediction"""
        x = torch.randn(1, 512)
        output = model(x)

        assert output.shape == (1, 10)

    def test_model_parameters(self, model):
        """Test model has trainable parameters"""
        params = list(model.parameters())

        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestCryptoDetector:
    """Test suite for CryptoDetector"""

    @pytest.fixture
    def detector(self, tmp_path):
        """Create detector instance (without loading model)"""
        # Create a dummy model file
        model = CryptoDetectionModel()
        model_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), model_path)

        return CryptoDetector(model_path=str(model_path))

    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector.model is not None
        assert detector.device is not None
        assert len(detector.labels) == 10

    def test_detect_with_numpy(self, detector):
        """Test detection with numpy array"""
        features = np.random.randn(512).astype(np.float32)

        result = detector.detect(features)

        assert isinstance(result, dict)
        assert "functions" in result
        assert "confidences" in result
        assert "explanations" in result
        assert "metadata" in result

    def test_detect_with_dict(self, detector):
        """Test detection with dictionary input"""
        features_dict = {f"feature_{i}": float(i) for i in range(512)}

        result = detector.detect(features_dict)

        assert isinstance(result, dict)
        assert "functions" in result

    def test_detect_results_structure(self, detector):
        """Test detection results structure"""
        features = np.random.randn(512).astype(np.float32)

        result = detector.detect(features)

        # Check functions structure
        assert isinstance(result["functions"], list)
        assert len(result["functions"]) > 0

        for func in result["functions"]:
            assert "type" in func
            assert "confidence" in func
            assert "rank" in func
            assert 0 <= func["confidence"] <= 1

        # Check confidences
        assert isinstance(result["confidences"], dict)

        # Check explanations
        assert isinstance(result["explanations"], dict)
        assert "method" in result["explanations"]

    def test_batch_detect(self, detector):
        """Test batch detection"""
        features_list = [
            np.random.randn(512).astype(np.float32)
            for _ in range(5)
        ]

        results = detector.batch_detect(features_list)

        assert len(results) == 5
        assert all(isinstance(r, dict) for r in results)

    def test_generate_explanations(self, detector):
        """Test XAI explanation generation"""
        features = torch.randn(1, 512)
        predictions = torch.tensor([0])

        explanations = detector._generate_explanations(features, predictions)

        assert isinstance(explanations, dict)
        assert "method" in explanations
        assert "feature_importance" in explanations

    def test_error_handling(self, detector):
        """Test error handling for invalid input"""
        invalid_features = np.array([])  # Empty array

        result = detector.detect(invalid_features)

        # Should handle gracefully
        assert isinstance(result, dict)
        # May have empty results or error in metadata


class TestModelTrainer:
    """Test suite for ModelTrainer"""

    @pytest.fixture
    def trainer(self):
        """Create trainer instance"""
        return ModelTrainer(
            input_dim=512,
            hidden_dim=128,  # Smaller for faster tests
            num_classes=10,
            learning_rate=0.001
        )

    def test_trainer_initialization(self, trainer):
        """Test trainer initialization"""
        assert trainer.model is not None
        assert trainer.criterion is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_training_small_dataset(self, trainer, sample_features, sample_labels):
        """Test training on small dataset"""
        # Use very few epochs for testing
        history = trainer.train(
            features=sample_features,
            labels=sample_labels,
            epochs=2,
            batch_size=16,
            validation_split=0.2
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert "train_acc" in history
        assert "val_acc" in history
        assert len(history["train_loss"]) == 2

    def test_train_epoch(self, trainer, sample_features, sample_labels):
        """Test single training epoch"""
        from torch.utils.data import DataLoader

        dataset = CryptoDataset(sample_features, sample_labels)
        loader = DataLoader(dataset, batch_size=16)

        loss, acc = trainer._train_epoch(loader)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1

    def test_validate_epoch(self, trainer, sample_features, sample_labels):
        """Test single validation epoch"""
        from torch.utils.data import DataLoader

        dataset = CryptoDataset(sample_features, sample_labels)
        loader = DataLoader(dataset, batch_size=16)

        loss, acc = trainer._validate_epoch(loader)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1

    def test_save_model(self, trainer, tmp_path):
        """Test model saving"""
        model_path = tmp_path / "test_model.pth"
        trainer.save_model(str(model_path))

        assert model_path.exists()

    def test_save_history(self, trainer, tmp_path):
        """Test saving training history"""
        trainer.history = {
            "train_loss": [1.0, 0.8],
            "val_loss": [1.1, 0.9],
            "train_acc": [0.5, 0.6],
            "val_acc": [0.4, 0.5]
        }

        history_path = tmp_path / "history.json"
        trainer.save_history(str(history_path))

        assert history_path.exists()


class TestCryptoDataset:
    """Test suite for CryptoDataset"""

    def test_dataset_creation(self, sample_features, sample_labels):
        """Test dataset creation"""
        dataset = CryptoDataset(sample_features, sample_labels)

        assert len(dataset) == len(sample_features)

    def test_dataset_getitem(self, sample_features, sample_labels):
        """Test dataset item retrieval"""
        dataset = CryptoDataset(sample_features, sample_labels)

        features, label = dataset[0]

        assert isinstance(features, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert features.shape == (512,)

    def test_dataset_iteration(self, sample_features, sample_labels):
        """Test iterating over dataset"""
        dataset = CryptoDataset(sample_features, sample_labels)

        for features, label in dataset:
            assert isinstance(features, torch.Tensor)
            assert isinstance(label, torch.Tensor)
            break  # Just test first item
