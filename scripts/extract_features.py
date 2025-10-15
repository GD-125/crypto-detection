"""
Feature Extraction Script

Extracts features from firmware binaries using the feature extraction pipeline.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.binary_analyzer.analyzer import BinaryAnalyzer
from services.feature_extractor.extractor import FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_features_from_directory(
    input_dir,
    output_dir,
    labels_file=None,
    feature_dim=512
):
    """
    Extract features from all firmware files in a directory

    Args:
        input_dir: Directory containing firmware files
        output_dir: Output directory for features
        labels_file: Optional JSON file with labels
        feature_dim: Feature vector dimension
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize analyzers
    binary_analyzer = BinaryAnalyzer()
    feature_extractor = FeatureExtractor(feature_dim=feature_dim)

    # Find all binary files
    binary_extensions = ['.bin', '.elf', '.hex', '.fw', '']
    binary_files = []

    for ext in binary_extensions:
        binary_files.extend(input_path.glob(f'*{ext}'))

    binary_files = [f for f in binary_files if f.is_file()]

    logger.info(f"Found {len(binary_files)} binary files")

    # Load labels if provided
    labels_map = {}
    if labels_file and os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
            if 'samples' in labels_data:
                labels_map = {
                    s['filename']: s['label']
                    for s in labels_data['samples']
                }
            logger.info(f"Loaded labels for {len(labels_map)} files")

    # Extract features
    features_list = []
    labels_list = []
    metadata_list = []

    for binary_file in tqdm(binary_files, desc="Extracting features"):
        try:
            # Step 1: Binary analysis
            logger.debug(f"Analyzing {binary_file.name}")
            disassembly_result = binary_analyzer.disassemble(str(binary_file))

            # Step 2: Feature extraction
            features = feature_extractor.extract(disassembly_result)

            features_list.append(features)

            # Get label if available
            label = labels_map.get(binary_file.name, -1)
            labels_list.append(label)

            # Store metadata
            metadata_list.append({
                'filename': binary_file.name,
                'path': str(binary_file),
                'size': binary_file.stat().st_size,
                'architecture': disassembly_result.get('architecture', 'unknown'),
                'label': label
            })

        except Exception as e:
            logger.error(f"Error processing {binary_file.name}: {str(e)}")
            continue

    if not features_list:
        logger.error("No features extracted!")
        return

    # Convert to numpy arrays
    features_array = np.vstack(features_list)
    labels_array = np.array(labels_list, dtype=np.int64)

    logger.info(f"Extracted features shape: {features_array.shape}")
    logger.info(f"Labels shape: {labels_array.shape}")

    # Save features and labels
    np.save(output_path / "features.npy", features_array)
    np.save(output_path / "labels.npy", labels_array)

    # Save metadata
    with open(output_path / "metadata.json", 'w') as f:
        json.dump({
            'total_samples': len(features_list),
            'feature_dim': feature_dim,
            'samples': metadata_list
        }, f, indent=2)

    logger.info(f"\n✓ Feature extraction complete!")
    logger.info(f"  Processed: {len(features_list)} files")
    logger.info(f"  Features: {features_array.shape}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"\nFiles created:")
    logger.info(f"  - features.npy: Feature vectors")
    logger.info(f"  - labels.npy: Labels array")
    logger.info(f"  - metadata.json: Sample information")


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from firmware binaries"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing firmware files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for features"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="JSON file with labels (optional)"
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=512,
        help="Feature vector dimension"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    extract_features_from_directory(
        args.input,
        args.output,
        args.labels,
        args.feature_dim
    )


if __name__ == "__main__":
    main()
