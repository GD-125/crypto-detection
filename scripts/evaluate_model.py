"""
Model Evaluation Script

Evaluates the trained model on test data.
"""

import os
import sys
import argparse
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.ai_engine.inference import CryptoDetector, CryptoDetectionModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Crypto function labels
CRYPTO_LABELS = [
    "AES", "RSA", "SHA256", "SHA512", "DES",
    "3DES", "ECC", "HMAC", "MD5", "Other"
]


def evaluate_model(
    model_path,
    test_features_path,
    test_labels_path,
    output_dir="reports"
):
    """
    Evaluate model on test data

    Args:
        model_path: Path to trained model
        test_features_path: Path to test features
        test_labels_path: Path to test labels
        output_dir: Output directory for reports
    """
    logger.info("=" * 70)
    logger.info("CRYPTOGRAPHIC FUNCTION DETECTION - MODEL EVALUATION")
    logger.info("=" * 70)

    # Load test data
    logger.info(f"\nLoading test data...")
    logger.info(f"  Features: {test_features_path}")
    logger.info(f"  Labels: {test_labels_path}")

    test_features = np.load(test_features_path)
    test_labels = np.load(test_labels_path)

    logger.info(f"\nTest dataset:")
    logger.info(f"  Samples: {len(test_features)}")
    logger.info(f"  Feature dimension: {test_features.shape[1]}")

    # Load model
    logger.info(f"\nLoading model from: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CryptoDetectionModel(
        input_dim=test_features.shape[1],
        hidden_dim=256,
        num_classes=10
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    logger.info(f"  Device: {device}")

    # Make predictions
    logger.info(f"\nRunning inference...")

    test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(test_features_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    predictions_np = predictions.cpu().numpy()
    probabilities_np = probabilities.cpu().numpy()

    # Calculate metrics
    logger.info(f"\nCalculating metrics...")

    accuracy = accuracy_score(test_labels, predictions_np)
    precision = precision_score(test_labels, predictions_np, average='weighted', zero_division=0)
    recall = recall_score(test_labels, predictions_np, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, predictions_np, average='weighted', zero_division=0)

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")

    # Per-class metrics
    logger.info(f"\n" + "-" * 70)
    logger.info("Per-Class Performance:")
    logger.info("-" * 70)

    report = classification_report(
        test_labels,
        predictions_np,
        target_names=CRYPTO_LABELS,
        zero_division=0
    )
    logger.info(f"\n{report}")

    # Confusion matrix
    conf_matrix = confusion_matrix(test_labels, predictions_np)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"(Rows: True labels, Columns: Predicted labels)")
    logger.info("\n" + str(conf_matrix))

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "overall_metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        },
        "per_class_metrics": classification_report(
            test_labels,
            predictions_np,
            target_names=CRYPTO_LABELS,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": conf_matrix.tolist(),
        "test_samples": len(test_features),
        "model_path": model_path
    }

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save confusion matrix as CSV
    conf_matrix_path = os.path.join(output_dir, "confusion_matrix.csv")
    np.savetxt(
        conf_matrix_path,
        conf_matrix,
        delimiter=',',
        fmt='%d',
        header=','.join(CRYPTO_LABELS),
        comments=''
    )

    # Calculate and display confidence statistics
    avg_confidence = np.mean(np.max(probabilities_np, axis=1))
    logger.info(f"\nConfidence Statistics:")
    logger.info(f"  Average confidence: {avg_confidence:.4f}")
    logger.info(f"  Min confidence: {np.min(np.max(probabilities_np, axis=1)):.4f}")
    logger.info(f"  Max confidence: {np.max(np.max(probabilities_np, axis=1)):.4f}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"\nSaved files:")
    logger.info(f"  ✓ Results: {results_path}")
    logger.info(f"  ✓ Confusion Matrix: {conf_matrix_path}")

    # Recommendations
    logger.info(f"\nRecommendations:")
    if accuracy < 0.7:
        logger.info("  ⚠ Low accuracy - consider:")
        logger.info("    - Training for more epochs")
        logger.info("    - Collecting more training data")
        logger.info("    - Adjusting hyperparameters")
    elif accuracy < 0.85:
        logger.info("  ℹ Moderate accuracy - consider:")
        logger.info("    - Fine-tuning the model")
        logger.info("    - Adding more diverse training samples")
    else:
        logger.info("  ✓ Good accuracy! Model is ready for deployment")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cryptographic function detection model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/crypto_detector.pth",
        help="Path to trained model"
    )
    parser.add_argument(
        "--test-features",
        type=str,
        required=True,
        help="Path to test features .npy file"
    )
    parser.add_argument(
        "--test-labels",
        type=str,
        required=True,
        help="Path to test labels .npy file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory for evaluation reports"
    )

    args = parser.parse_args()

    evaluate_model(
        args.model,
        args.test_features,
        args.test_labels,
        args.output
    )


if __name__ == "__main__":
    main()
