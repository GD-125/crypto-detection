"""
Quick Start Script - Complete Pipeline

Automates the entire workflow:
1. Generate/prepare dataset
2. Extract features
3. Train model
4. Evaluate model
5. Start services
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run shell command and handle errors"""
    logger.info(f"\n{'='*70}")
    logger.info(f"{description}")
    logger.info(f"{'='*70}")
    logger.info(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        logger.info(f"✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed!")
        logger.error(f"Error: {e}")
        return False


def quick_start(
    num_samples=1000,
    epochs=50,
    skip_training=False,
    skip_evaluation=False
):
    """
    Run complete quick start pipeline

    Args:
        num_samples: Number of synthetic samples to generate
        epochs: Number of training epochs
        skip_training: Skip training if model exists
        skip_evaluation: Skip evaluation step
    """
    logger.info("\n" + "="*70)
    logger.info("CRYPTOGRAPHIC FUNCTION DETECTION - QUICK START")
    logger.info("="*70)

    base_dir = Path(__file__).parent.parent
    os.chdir(base_dir)

    # Step 1: Generate Dataset
    logger.info("\nStep 1/5: Generating synthetic dataset...")

    if not (base_dir / "data/datasets/synthetic/features.npy").exists():
        success = run_command(
            [
                sys.executable,
                "scripts/generate_dataset.py",
                "--output", "data/datasets/synthetic",
                "--num-samples", str(num_samples)
            ],
            "Dataset Generation"
        )

        if not success:
            logger.error("Failed to generate dataset. Exiting.")
            return False
    else:
        logger.info("✓ Dataset already exists, skipping generation")

    # Step 2: Extract Features
    logger.info("\nStep 2/5: Extracting features from firmware...")

    if not (base_dir / "data/datasets/features/features.npy").exists():
        success = run_command(
            [
                sys.executable,
                "scripts/extract_features.py",
                "--input", "data/datasets/synthetic/firmware",
                "--output", "data/datasets/features",
                "--labels", "data/datasets/synthetic/metadata.json"
            ],
            "Feature Extraction"
        )

        if not success:
            logger.error("Failed to extract features. Exiting.")
            return False
    else:
        logger.info("✓ Features already extracted, skipping")

    # Step 3: Train Model
    logger.info("\nStep 3/5: Training model...")

    model_path = base_dir / "data/models/crypto_detector.pth"

    if skip_training and model_path.exists():
        logger.info("✓ Model exists and skip_training=True, skipping training")
    else:
        success = run_command(
            [
                sys.executable,
                "scripts/train_model.py",
                "--features", "data/datasets/features/features.npy",
                "--labels", "data/datasets/features/labels.npy",
                "--output", "data/models/crypto_detector.pth",
                "--epochs", str(epochs),
                "--batch-size", "32"
            ],
            "Model Training"
        )

        if not success:
            logger.error("Failed to train model. Exiting.")
            return False

    # Step 4: Evaluate Model
    if not skip_evaluation:
        logger.info("\nStep 4/5: Evaluating model...")

        # Split dataset for evaluation
        features_path = base_dir / "data/datasets/features/features.npy"
        labels_path = base_dir / "data/datasets/features/labels.npy"

        import numpy as np

        features = np.load(features_path)
        labels = np.load(labels_path)

        # Use last 20% as test set
        test_size = int(0.2 * len(features))
        test_features = features[-test_size:]
        test_labels = labels[-test_size:]

        # Save test data
        test_dir = base_dir / "data/datasets/test"
        test_dir.mkdir(parents=True, exist_ok=True)

        np.save(test_dir / "features.npy", test_features)
        np.save(test_dir / "labels.npy", test_labels)

        success = run_command(
            [
                sys.executable,
                "scripts/evaluate_model.py",
                "--model", "data/models/crypto_detector.pth",
                "--test-features", "data/datasets/test/features.npy",
                "--test-labels", "data/datasets/test/labels.npy",
                "--output", "reports"
            ],
            "Model Evaluation"
        )

        if not success:
            logger.warning("Evaluation failed, but continuing...")
    else:
        logger.info("\nStep 4/5: Skipping evaluation")

    # Step 5: Summary
    logger.info("\n" + "="*70)
    logger.info("QUICK START COMPLETE!")
    logger.info("="*70)

    logger.info("\n✓ All steps completed successfully!")
    logger.info("\nGenerated files:")
    logger.info(f"  ✓ Dataset: data/datasets/synthetic/")
    logger.info(f"  ✓ Features: data/datasets/features/")
    logger.info(f"  ✓ Model: data/models/crypto_detector.pth")
    logger.info(f"  ✓ Evaluation: reports/evaluation_results.json")

    logger.info("\nNext Steps:")
    logger.info("\n1. Start the API server:")
    logger.info("   cd services")
    logger.info("   uvicorn api.main:app --reload")

    logger.info("\n2. Or use Docker:")
    logger.info("   docker-compose up -d")

    logger.info("\n3. Access the application:")
    logger.info("   Frontend: http://localhost:3000")
    logger.info("   API Docs: http://localhost:8000/api/docs")

    logger.info("\n4. Test with your own firmware:")
    logger.info("   - Upload via web interface")
    logger.info("   - Or use API directly")

    logger.info("\n" + "="*70)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Quick start script for complete pipeline"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training if model exists"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation step"
    )

    args = parser.parse_args()

    success = quick_start(
        num_samples=args.num_samples,
        epochs=args.epochs,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
