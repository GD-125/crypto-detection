"""
Model Training Script

Trains the cryptographic primitives detection model.
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
import json
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.ai_engine.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
    features_path,
    labels_path,
    output_path="data/models/crypto_detector.pth",
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    hidden_dim=256,
    validation_split=0.2,
    num_classes=10
):
    """
    Train the cryptographic function detection model

    Args:
        features_path: Path to features .npy file
        labels_path: Path to labels .npy file
        output_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hidden_dim: Hidden layer dimension
        validation_split: Validation split ratio
        num_classes: Number of crypto function classes
    """
    logger.info("=" * 60)
    logger.info("CRYPTOGRAPHIC FUNCTION DETECTION - MODEL TRAINING")
    logger.info("=" * 60)

    # Load data
    logger.info(f"\nLoading training data...")
    logger.info(f"  Features: {features_path}")
    logger.info(f"  Labels: {labels_path}")

    features = np.load(features_path)
    labels = np.load(labels_path)

    logger.info(f"\nDataset information:")
    logger.info(f"  Samples: {len(features)}")
    logger.info(f"  Feature dimension: {features.shape[1]}")
    logger.info(f"  Number of classes: {num_classes}")

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"\nClass distribution:")
    for cls, count in zip(unique, counts):
        logger.info(f"  Class {cls}: {count} samples ({count/len(labels)*100:.2f}%)")

    # Initialize trainer
    logger.info(f"\nInitializing model...")
    logger.info(f"  Input dimension: {features.shape[1]}")
    logger.info(f"  Hidden dimension: {hidden_dim}")
    logger.info(f"  Number of classes: {num_classes}")
    logger.info(f"  Learning rate: {learning_rate}")

    trainer = ModelTrainer(
        input_dim=features.shape[1],
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        learning_rate=learning_rate
    )

    # Training configuration
    logger.info(f"\nTraining configuration:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Validation split: {validation_split}")
    logger.info(f"  Device: {trainer.device}")

    # Train model
    logger.info(f"\nStarting training...")
    logger.info("=" * 60)

    history = trainer.train(
        features=features,
        labels=labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split
    )

    # Save model
    logger.info(f"\nSaving model to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trainer.save_model(output_path)

    # Save training history
    history_path = output_path.replace('.pth', '_history.json')
    trainer.save_history(history_path)

    # Save training configuration
    config = {
        'input_dim': features.shape[1],
        'hidden_dim': hidden_dim,
        'num_classes': num_classes,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'validation_split': validation_split,
        'final_train_loss': history['train_loss'][-1],
        'final_train_acc': history['train_acc'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': max(history['val_acc'])
    }

    config_path = output_path.replace('.pth', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\nFinal Results:")
    logger.info(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"  Train Accuracy: {history['train_acc'][-1]:.4f}")
    logger.info(f"  Val Loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"  Val Accuracy: {history['val_acc'][-1]:.4f}")
    logger.info(f"  Best Val Accuracy: {max(history['val_acc']):.4f}")

    logger.info(f"\nSaved files:")
    logger.info(f"  ✓ Model: {output_path}")
    logger.info(f"  ✓ History: {history_path}")
    logger.info(f"  ✓ Config: {config_path}")
    logger.info(f"  ✓ Best Model: {output_path.replace('.pth', '_best.pth')}")

    logger.info(f"\nNext steps:")
    logger.info(f"  1. Evaluate model: python scripts/evaluate_model.py")
    logger.info(f"  2. Start API server: uvicorn services.api.main:app")
    logger.info(f"  3. Test inference: python scripts/test_inference.py")

    return trainer, history


def main():
    parser = argparse.ArgumentParser(
        description="Train cryptographic function detection model"
    )
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path to features .npy file"
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to labels .npy file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/models/crypto_detector.pth",
        help="Output path for trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden layer dimension"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of crypto function classes"
    )

    args = parser.parse_args()

    train_model(
        features_path=args.features,
        labels_path=args.labels,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        validation_split=args.validation_split,
        num_classes=args.num_classes
    )


if __name__ == "__main__":
    main()
