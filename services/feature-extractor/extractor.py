"""
Feature Extraction Module
Standardizes input from diverse ISAs (Instruction Set Architectures)
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging
import re
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts and standardizes features from binary disassembly
    Supports multiple ISAs (x86, ARM, MIPS, etc.)
    """

    def __init__(self, feature_dim: int = 512):
        """
        Initialize Feature Extractor

        Args:
            feature_dim: Target dimension for feature vector
        """
        self.feature_dim = feature_dim

        # Crypto-related instruction patterns
        self.crypto_instructions = {
            "x86": ["aes", "sha", "xor", "rol", "ror", "shl", "shr"],
            "arm": ["eor", "ror", "lsl", "lsr", "sha", "aes"],
            "mips": ["xor", "sll", "srl", "sra"],
            "generic": ["xor", "shift", "rotate", "and", "or"]
        }

        # Crypto-related function patterns
        self.crypto_patterns = [
            r"aes_.*",
            r"rsa_.*",
            r"sha\d+.*",
            r"md5.*",
            r".*encrypt.*",
            r".*decrypt.*",
            r".*cipher.*",
            r".*hash.*",
            r".*hmac.*"
        ]

    def extract(self, disassembly_result: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from disassembly results

        Args:
            disassembly_result: Output from BinaryAnalyzer

        Returns:
            Numpy array of extracted features
        """
        try:
            logger.info("Extracting features from disassembly...")

            # Extract different feature categories
            structural_features = self._extract_structural_features(disassembly_result)
            instruction_features = self._extract_instruction_features(disassembly_result)
            string_features = self._extract_string_features(disassembly_result)
            import_features = self._extract_import_features(disassembly_result)
            crypto_features = self._extract_crypto_features(disassembly_result)

            # Combine all features
            all_features = np.concatenate([
                structural_features,
                instruction_features,
                string_features,
                import_features,
                crypto_features
            ])

            # Normalize and pad/truncate to target dimension
            features = self._normalize_features(all_features)

            logger.info(f"Extracted {len(features)} features")

            return features

        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            # Return zero vector on error
            return np.zeros(self.feature_dim)

    def _extract_structural_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract structural features from binary"""
        features = []

        # Number of functions
        num_functions = len(data.get("functions", []))
        features.append(num_functions)

        # Average function size
        function_sizes = [f.get("size", 0) for f in data.get("functions", [])]
        avg_size = np.mean(function_sizes) if function_sizes else 0
        features.append(avg_size)

        # Function size variance
        size_variance = np.var(function_sizes) if function_sizes else 0
        features.append(size_variance)

        # Number of strings
        num_strings = len(data.get("strings", []))
        features.append(num_strings)

        # Number of imports
        num_imports = len(data.get("imports", []))
        features.append(num_imports)

        # File size
        file_size = data.get("metadata", {}).get("file_size", 0)
        features.append(file_size)

        return np.array(features, dtype=np.float32)

    def _extract_instruction_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract instruction-level features"""
        features = []

        # Get architecture
        arch = data.get("architecture", "generic").lower()

        # Count crypto-related instructions
        crypto_inst_patterns = self.crypto_instructions.get(
            arch,
            self.crypto_instructions["generic"]
        )

        # For now, use function names as proxy
        # In full implementation, would parse actual instructions
        functions = data.get("functions", [])
        function_names = [f.get("name", "").lower() for f in functions]

        # Count occurrences of crypto instruction patterns
        for pattern in crypto_inst_patterns:
            count = sum(1 for name in function_names if pattern in name)
            features.append(count)

        # Pad to fixed length
        while len(features) < 20:
            features.append(0)

        return np.array(features[:20], dtype=np.float32)

    def _extract_string_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract string-based features"""
        features = []

        strings = data.get("strings", [])
        string_values = [s.get("value", "").lower() for s in strings]

        # Crypto-related string keywords
        crypto_keywords = [
            "aes", "rsa", "des", "sha", "md5", "key",
            "encrypt", "decrypt", "cipher", "hash"
        ]

        # Count crypto keywords in strings
        for keyword in crypto_keywords:
            count = sum(1 for s in string_values if keyword in s)
            features.append(count)

        # Average string length
        if string_values:
            avg_length = np.mean([len(s) for s in string_values])
            features.append(avg_length)
        else:
            features.append(0)

        # Entropy of strings (simplified)
        entropy = self._calculate_string_entropy(string_values)
        features.append(entropy)

        return np.array(features, dtype=np.float32)

    def _extract_import_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract import-based features"""
        features = []

        imports = data.get("imports", [])
        import_names = [str(imp).lower() for imp in imports]

        # Crypto-related libraries
        crypto_libs = [
            "openssl", "crypto", "ssl", "gcrypt", "mbedtls",
            "boringssl", "libsodium", "wolfssl"
        ]

        # Check for crypto library imports
        for lib in crypto_libs:
            has_lib = any(lib in imp for imp in import_names)
            features.append(1.0 if has_lib else 0.0)

        return np.array(features, dtype=np.float32)

    def _extract_crypto_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract cryptography-specific features"""
        features = []

        # Detected crypto patterns from binary analyzer
        crypto_patterns = data.get("crypto_patterns", [])

        # Count patterns by type
        pattern_types = [p.get("keyword", "") for p in crypto_patterns]
        type_counter = Counter(pattern_types)

        # Common crypto algorithms
        algorithms = ["aes", "rsa", "des", "sha", "md5", "hmac"]

        for algo in algorithms:
            count = type_counter.get(algo, 0)
            features.append(count)

        # Total pattern count
        features.append(len(crypto_patterns))

        # Pattern density (patterns per function)
        num_functions = len(data.get("functions", []))
        if num_functions > 0:
            density = len(crypto_patterns) / num_functions
        else:
            density = 0
        features.append(density)

        return np.array(features, dtype=np.float32)

    def _calculate_string_entropy(self, strings: List[str]) -> float:
        """Calculate entropy of string set"""
        if not strings:
            return 0.0

        # Concatenate all strings
        combined = "".join(strings)

        if not combined:
            return 0.0

        # Calculate character frequency
        freq = Counter(combined)
        total = len(combined)

        # Calculate Shannon entropy
        entropy = 0.0
        for count in freq.values():
            p = count / total
            entropy -= p * np.log2(p)

        return entropy

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize and resize feature vector to target dimension

        Args:
            features: Raw feature vector

        Returns:
            Normalized feature vector of size feature_dim
        """
        # Handle NaN and Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Log transform for large values
        features = np.log1p(np.abs(features)) * np.sign(features)

        # Standardize (zero mean, unit variance)
        if np.std(features) > 0:
            features = (features - np.mean(features)) / np.std(features)

        # Resize to target dimension
        if len(features) < self.feature_dim:
            # Pad with zeros
            padding = np.zeros(self.feature_dim - len(features))
            features = np.concatenate([features, padding])
        elif len(features) > self.feature_dim:
            # Truncate
            features = features[:self.feature_dim]

        return features.astype(np.float32)

    def batch_extract(self, disassembly_results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract features from multiple binaries

        Args:
            disassembly_results: List of disassembly results

        Returns:
            2D numpy array of features (batch_size x feature_dim)
        """
        features_list = []

        for result in disassembly_results:
            features = self.extract(result)
            features_list.append(features)

        return np.vstack(features_list)
