"""
Sample Dataset Generator for Cryptographic Function Detection

Generates synthetic firmware samples with known cryptographic functions
for training and testing purposes.
"""

import os
import numpy as np
import json
import argparse
from pathlib import Path
import struct
import random

# Crypto function patterns
CRYPTO_FUNCTIONS = {
    "AES": {
        "label": 0,
        "patterns": [
            b"\x63\x7c\x77\x7b\xf2\x6b\x6f\xc5",  # AES S-box
            b"\x52\x09\x6a\xd5\x30\x36\xa5\x38",
            b"\xbf\x40\xa3\x9e\x81\xf3\xd7\xfb"
        ]
    },
    "RSA": {
        "label": 1,
        "patterns": [
            b"\x00\x02" + b"\xFF" * 8,  # PKCS#1 padding
            b"RSA",
            b"\x30\x82"  # DER encoding
        ]
    },
    "SHA256": {
        "label": 2,
        "patterns": [
            struct.pack(">I", 0x6a09e667),  # SHA-256 constants
            struct.pack(">I", 0xbb67ae85),
            struct.pack(">I", 0x3c6ef372)
        ]
    },
    "SHA512": {
        "label": 3,
        "patterns": [
            struct.pack(">Q", 0x6a09e667f3bcc908),
            struct.pack(">Q", 0xbb67ae8584caa73b)
        ]
    },
    "DES": {
        "label": 4,
        "patterns": [
            b"\x0e\x04\x0d\x01\x02\x0f\x0b\x08",  # DES permutation
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        ]
    },
    "3DES": {
        "label": 5,
        "patterns": [
            b"\x0e\x04\x0d\x01\x02\x0f\x0b\x08",
            b"3DES",
            b"\x01\x01\x02\x02\x02\x02\x02\x02"
        ]
    },
    "ECC": {
        "label": 6,
        "patterns": [
            b"secp256k1",
            b"secp256r1",
            b"\x04" + b"\x00" * 64  # Uncompressed point
        ]
    },
    "HMAC": {
        "label": 7,
        "patterns": [
            b"HMAC",
            b"\x36" * 64,  # ipad
            b"\x5c" * 64   # opad
        ]
    },
    "MD5": {
        "label": 8,
        "patterns": [
            struct.pack("<I", 0x67452301),  # MD5 constants
            struct.pack("<I", 0xefcdab89),
            struct.pack("<I", 0x98badcfe)
        ]
    },
    "Other": {
        "label": 9,
        "patterns": [
            b"\x00" * 16,
            b"\xFF" * 16
        ]
    }
}


def generate_binary_sample(crypto_type, size=4096):
    """
    Generate a synthetic binary sample with crypto patterns

    Args:
        crypto_type: Type of crypto function
        size: Size of binary in bytes

    Returns:
        Binary data
    """
    data = bytearray(random.getrandbits(8) for _ in range(size))

    if crypto_type in CRYPTO_FUNCTIONS:
        patterns = CRYPTO_FUNCTIONS[crypto_type]["patterns"]

        # Insert patterns at random positions
        for pattern in patterns:
            if len(data) > len(pattern):
                pos = random.randint(0, len(data) - len(pattern))
                data[pos:pos + len(pattern)] = pattern

    # Add ELF header to make it look like a real binary
    elf_header = b"\x7fELF\x02\x01\x01\x00"
    data[0:8] = elf_header

    return bytes(data)


def generate_dataset(output_dir, num_samples=1000, size_range=(2048, 8192)):
    """
    Generate complete dataset with features and labels

    Args:
        output_dir: Output directory
        num_samples: Number of samples to generate
        size_range: Tuple of (min_size, max_size) for binaries
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    firmware_dir = output_path / "firmware"
    firmware_dir.mkdir(exist_ok=True)

    samples = []
    labels = []

    print(f"Generating {num_samples} samples...")

    crypto_types = list(CRYPTO_FUNCTIONS.keys())

    for i in range(num_samples):
        # Random crypto type
        crypto_type = random.choice(crypto_types)
        label = CRYPTO_FUNCTIONS[crypto_type]["label"]

        # Random size
        size = random.randint(size_range[0], size_range[1])

        # Generate binary
        binary_data = generate_binary_sample(crypto_type, size)

        # Save binary
        filename = f"sample_{i:05d}_{crypto_type}.bin"
        filepath = firmware_dir / filename

        with open(filepath, "wb") as f:
            f.write(binary_data)

        samples.append({
            "id": i,
            "filename": filename,
            "path": str(filepath),
            "crypto_type": crypto_type,
            "label": label,
            "size": size
        })

        labels.append(label)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")

    # Save metadata
    metadata = {
        "total_samples": num_samples,
        "crypto_types": {k: v["label"] for k, v in CRYPTO_FUNCTIONS.items()},
        "samples": samples
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save labels
    labels_array = np.array(labels, dtype=np.int64)
    np.save(output_path / "labels.npy", labels_array)

    # Generate train/test split
    split_idx = int(0.8 * num_samples)
    train_indices = list(range(split_idx))
    test_indices = list(range(split_idx, num_samples))

    split_info = {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_size": len(train_indices),
        "test_size": len(test_indices)
    }

    with open(output_path / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n✓ Dataset generation complete!")
    print(f"  Total samples: {num_samples}")
    print(f"  Train samples: {len(train_indices)}")
    print(f"  Test samples: {len(test_indices)}")
    print(f"  Output directory: {output_path}")
    print(f"\nFiles created:")
    print(f"  - firmware/: Binary files")
    print(f"  - metadata.json: Sample information")
    print(f"  - labels.npy: Labels array")
    print(f"  - split.json: Train/test split")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic firmware dataset for crypto detection"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/datasets/samples",
        help="Output directory"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=2048,
        help="Minimum binary size in bytes"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=8192,
        help="Maximum binary size in bytes"
    )

    args = parser.parse_args()

    generate_dataset(
        args.output,
        args.num_samples,
        (args.min_size, args.max_size)
    )


if __name__ == "__main__":
    main()
