"""
AI Inference Engine for Cryptographic Function Detection
Uses PyTorch for deep learning inference and Captum for explainability
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDetectionModel(nn.Module):
    """
    Neural Network Model for Cryptographic Function Detection
    Architecture: Multi-layer Transformer-based classifier
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_classes: int = 10):
        super(CryptoDetectionModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Feature embedding layers
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """Forward pass"""
        # Embed features
        x = self.embedding(x)

        # Add sequence dimension for transformer
        x = x.unsqueeze(1)

        # Transformer encoding
        x = self.transformer(x)

        # Remove sequence dimension
        x = x.squeeze(1)

        # Classification
        logits = self.classifier(x)

        return logits


class CryptoDetector:
    """
    Main inference class with XAI capabilities
    """

    def __init__(self, model_path: str = "data/models/crypto_detector.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = CryptoDetectionModel()

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Model loaded from {model_path}")
        except FileNotFoundError:
            logger.warning(f"Model file not found at {model_path}. Using untrained model.")

        self.model.to(self.device)
        self.model.eval()

        # Initialize XAI tools
        self.integrated_gradients = IntegratedGradients(self.model)

        # Crypto function labels
        self.labels = [
            "AES",
            "RSA",
            "SHA256",
            "SHA512",
            "DES",
            "3DES",
            "ECC",
            "HMAC",
            "MD5",
            "Other"
        ]

    def detect(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Detect cryptographic functions from extracted features

        Args:
            features: Numpy array of extracted features

        Returns:
            Dictionary containing predictions, confidences, and explanations
        """
        try:
            # Convert to tensor
            if isinstance(features, dict):
                features = self._dict_to_tensor(features)
            elif isinstance(features, np.ndarray):
                features = torch.tensor(features, dtype=torch.float32)

            features = features.to(self.device)

            # Ensure correct shape
            if len(features.shape) == 1:
                features = features.unsqueeze(0)

            # Inference
            with torch.no_grad():
                logits = self.model(features)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

            # Get top-k predictions
            top_k = 3
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

            detected_functions = []
            confidence_scores = {}

            for i in range(top_k):
                func_idx = top_indices[0, i].item()
                func_name = self.labels[func_idx]
                confidence = top_probs[0, i].item()

                detected_functions.append({
                    "type": func_name,
                    "confidence": float(confidence),
                    "rank": i + 1
                })
                confidence_scores[func_name] = float(confidence)

            # Generate explanations using XAI
            explanations = self._generate_explanations(features, predictions)

            return {
                "functions": detected_functions,
                "confidences": confidence_scores,
                "explanations": explanations,
                "metadata": {
                    "model_version": "1.0.0",
                    "device": str(self.device),
                    "feature_dim": features.shape[1]
                }
            }

        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return {
                "functions": [],
                "confidences": {},
                "explanations": {},
                "metadata": {"error": str(e)}
            }

    def _generate_explanations(self, features: torch.Tensor, predictions: torch.Tensor) -> Dict[str, Any]:
        """
        Generate XAI explanations using Captum

        Args:
            features: Input features tensor
            predictions: Predicted class indices

        Returns:
            Dictionary of explanations
        """
        try:
            # Integrated Gradients attribution
            attributions = self.integrated_gradients.attribute(
                features,
                target=predictions,
                n_steps=50
            )

            # Convert to numpy
            attributions_np = attributions.cpu().detach().numpy()

            # Get top contributing features
            top_n = 10
            top_indices = np.argsort(np.abs(attributions_np[0]))[-top_n:][::-1]
            top_values = attributions_np[0][top_indices]

            feature_importance = [
                {
                    "feature_index": int(idx),
                    "importance": float(val),
                    "contribution": "positive" if val > 0 else "negative"
                }
                for idx, val in zip(top_indices, top_values)
            ]

            return {
                "method": "Integrated Gradients",
                "feature_importance": feature_importance,
                "summary": f"Top {top_n} features contributing to the prediction",
                "confidence_explanation": "Based on learned patterns from training data"
            }

        except Exception as e:
            logger.error(f"Explanation generation error: {str(e)}")
            return {
                "method": "Integrated Gradients",
                "error": str(e),
                "feature_importance": []
            }

    def _dict_to_tensor(self, feature_dict: Dict) -> torch.Tensor:
        """Convert feature dictionary to tensor"""
        # Extract numeric values and convert to tensor
        values = []
        for key in sorted(feature_dict.keys()):
            val = feature_dict[key]
            if isinstance(val, (int, float)):
                values.append(val)
            elif isinstance(val, list):
                values.extend(val)

        return torch.tensor(values, dtype=torch.float32)

    def batch_detect(self, features_list: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Batch detection for multiple firmware samples

        Args:
            features_list: List of feature arrays

        Returns:
            List of detection results
        """
        results = []
        for features in features_list:
            result = self.detect(features)
            results.append(result)

        return results
