"""
Training Module for Cryptographic Function Detection Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple, Dict, List
import logging
from tqdm import tqdm
import json

from .inference import CryptoDetectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDataset(Dataset):
    """Dataset class for cryptographic function features"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ModelTrainer:
    """
    Trainer class for the crypto detection model
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 10,
        learning_rate: float = 0.001
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on device: {self.device}")

        # Initialize model
        self.model = CryptoDetectionModel(input_dim, hidden_dim, num_classes)
        self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """
        Train the model

        Args:
            features: Training features
            labels: Training labels
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio

        Returns:
            Training history dictionary
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=validation_split, random_state=42
        )

        # Create datasets
        train_dataset = CryptoDataset(X_train, y_train)
        val_dataset = CryptoDataset(X_val, y_val)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self._train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self._validate_epoch(val_loader)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("data/models/crypto_detector_best.pth")
                logger.info("Best model saved!")

        # Save final model
        self.save_model("data/models/crypto_detector.pth")
        logger.info("Training complete!")

        return self.history

    def _train_epoch(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in tqdm(data_loader, desc="Training"):
            features, labels = features.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _validate_epoch(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in tqdm(data_loader, desc="Validation"):
                features, labels = features.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def save_history(self, path: str = "data/models/training_history.json"):
        """Save training history"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)
        logger.info(f"Training history saved to {path}")
