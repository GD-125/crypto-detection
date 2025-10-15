"""
Test Inference Script

Quick test of model inference on a single firmware file.
"""

import os
import sys
import argparse
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.binary_analyzer.analyzer import BinaryAnalyzer
from services.feature_extractor.extractor import FeatureExtractor
from services.ai_engine.inference import CryptoDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_inference(firmware_path, model_path="data/models/crypto_detector.pth"):
    """
    Test inference on a single firmware file

    Args:
        firmware_path: Path to firmware binary
        model_path: Path to trained model
    """
    logger.info("=" * 70)
    logger.info("CRYPTOGRAPHIC FUNCTION DETECTION - TEST INFERENCE")
    logger.info("=" * 70)

    # Check files exist
    if not os.path.exists(firmware_path):
        logger.error(f"Firmware file not found: {firmware_path}")
        return

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.info("Train model first: python scripts/train_model.py")
        return

    logger.info(f"\nFirmware: {firmware_path}")
    logger.info(f"Model: {model_path}")

    # Initialize components
    logger.info("\nInitializing analysis pipeline...")
    binary_analyzer = BinaryAnalyzer()
    feature_extractor = FeatureExtractor()
    crypto_detector = CryptoDetector(model_path=model_path)

    # Step 1: Binary Analysis
    logger.info("\nStep 1/3: Analyzing binary with Ghidra...")
    disassembly_result = binary_analyzer.disassemble(firmware_path)

    logger.info(f"  Architecture: {disassembly_result.get('architecture', 'unknown')}")
    logger.info(f"  Functions: {len(disassembly_result.get('functions', []))}")
    logger.info(f"  Strings: {len(disassembly_result.get('strings', []))}")

    # Step 2: Feature Extraction
    logger.info("\nStep 2/3: Extracting features...")
    features = feature_extractor.extract(disassembly_result)

    logger.info(f"  Feature dimension: {features.shape}")

    # Step 3: AI Inference
    logger.info("\nStep 3/3: Running AI inference...")
    predictions = crypto_detector.detect(features)

    # Display Results
    logger.info("\n" + "=" * 70)
    logger.info("DETECTION RESULTS")
    logger.info("=" * 70)

    logger.info("\nDetected Cryptographic Functions:")
    logger.info("-" * 70)

    for i, func in enumerate(predictions["functions"], 1):
        logger.info(f"\n{i}. {func['type']}")
        logger.info(f"   Confidence: {func['confidence']*100:.2f}%")
        logger.info(f"   Rank: #{func['rank']}")

        # Confidence bar
        bar_length = 40
        filled = int(bar_length * func['confidence'])
        bar = '█' * filled + '░' * (bar_length - filled)
        logger.info(f"   [{bar}]")

    # XAI Explanations
    if predictions.get("explanations"):
        logger.info("\n" + "-" * 70)
        logger.info("Explainable AI Analysis:")
        logger.info("-" * 70)

        explanations = predictions["explanations"]
        logger.info(f"\nMethod: {explanations.get('method', 'N/A')}")
        logger.info(f"Summary: {explanations.get('summary', 'N/A')}")

        if explanations.get("feature_importance"):
            logger.info("\nTop Contributing Features:")
            for feat in explanations["feature_importance"][:5]:
                logger.info(
                    f"  Feature {feat['feature_index']}: "
                    f"{feat['importance']:.4f} ({feat['contribution']})"
                )

    # Metadata
    logger.info("\n" + "-" * 70)
    logger.info("Metadata:")
    logger.info("-" * 70)
    metadata = predictions.get("metadata", {})
    for key, value in metadata.items():
        logger.info(f"  {key}: {value}")

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 70)

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Test inference on a firmware file"
    )
    parser.add_argument(
        "--firmware",
        type=str,
        required=True,
        help="Path to firmware binary"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/crypto_detector.pth",
        help="Path to trained model"
    )

    args = parser.parse_args()

    test_inference(args.firmware, args.model)


if __name__ == "__main__":
    main()
