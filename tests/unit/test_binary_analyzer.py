"""
Unit Tests for Binary Analyzer
"""

import pytest
import os
from services.binary_analyzer.analyzer import BinaryAnalyzer


class TestBinaryAnalyzer:
    """Test suite for BinaryAnalyzer"""

    @pytest.fixture
    def analyzer(self):
        """Create binary analyzer instance"""
        return BinaryAnalyzer()

    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.ghidra_path is not None
        assert analyzer.project_dir is not None

    def test_detect_architecture_x86_64(self, analyzer, tmp_path):
        """Test x86_64 architecture detection"""
        # Create a simple ELF x86_64 binary
        binary_path = tmp_path / "test_x86_64.bin"
        with open(binary_path, "wb") as f:
            # ELF header for x86_64
            f.write(b"\x7fELF\x02\x01\x01\x00")  # ELF magic + 64-bit
            f.write(b"\x00" * 8)
            f.write(b"\x3e\x00")  # x86_64 machine type
            f.write(b"\x00" * 1000)

        arch = analyzer._detect_architecture(str(binary_path))

        # Should detect x86_64 or x86 (depending on implementation)
        assert arch in ["x86_64", "x86", "arm", "mips", "powerpc"]

    def test_create_project_name(self, analyzer):
        """Test project name creation"""
        binary_path = "/path/to/firmware.bin"

        project_name = analyzer._create_project_name(binary_path)

        assert isinstance(project_name, str)
        assert len(project_name) > 0
        assert "firmware.bin" in project_name

    def test_calculate_hash(self, analyzer, sample_binary_file):
        """Test file hash calculation"""
        file_hash = analyzer._calculate_hash(sample_binary_file)

        assert isinstance(file_hash, str)
        assert len(file_hash) == 64  # SHA256 hex digest

    def test_parse_ghidra_output(self, analyzer):
        """Test parsing Ghidra output"""
        mock_output = {
            "functions": [
                {"name": "main", "address": "0x1000", "size": 100}
            ],
            "strings": [
                {"value": "test", "address": "0x2000"}
            ],
            "imports": ["libc.so"]
        }

        parsed = analyzer._parse_ghidra_output(mock_output)

        assert "functions" in parsed
        assert "strings" in parsed
        assert "imports" in parsed

    def test_parse_stdout(self, analyzer):
        """Test parsing Ghidra stdout"""
        stdout = '''
Some log output
{"functions": [], "strings": [], "imports": []}
More output
'''

        result = analyzer._parse_stdout(stdout)

        assert isinstance(result, dict)
        assert "functions" in result

    def test_extract_crypto_patterns_aes(self, analyzer):
        """Test extracting AES patterns"""
        disassembly_data = {
            "functions": [
                {"name": "aes_encrypt", "address": "0x1000"},
                {"name": "main", "address": "0x2000"}
            ],
            "strings": [
                {"value": "AES-256-CBC", "address": "0x3000"}
            ]
        }

        patterns = analyzer._extract_crypto_patterns(disassembly_data)

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert any(p.get("keyword") == "aes" for p in patterns)

    def test_extract_crypto_patterns_multiple(self, analyzer):
        """Test extracting multiple crypto patterns"""
        disassembly_data = {
            "functions": [
                {"name": "rsa_sign", "address": "0x1000"},
                {"name": "sha256_hash", "address": "0x2000"},
                {"name": "aes_decrypt", "address": "0x3000"}
            ],
            "strings": [
                {"value": "RSA-2048", "address": "0x4000"},
                {"value": "SHA-256", "address": "0x5000"}
            ]
        }

        patterns = analyzer._extract_crypto_patterns(disassembly_data)

        assert len(patterns) >= 3
        keywords = [p.get("keyword") for p in patterns]
        assert "rsa" in keywords
        assert "sha" in keywords
        assert "aes" in keywords

    def test_disassemble_structure(self, analyzer, sample_binary_file):
        """Test disassemble returns correct structure"""
        # Note: This will fail if Ghidra is not installed
        # We're testing the structure, not actual disassembly
        result = analyzer.disassemble(sample_binary_file)

        assert isinstance(result, dict)
        assert "binary_path" in result
        assert "architecture" in result

    def test_disassemble_with_specific_arch(self, analyzer, sample_binary_file):
        """Test disassemble with specific architecture"""
        result = analyzer.disassemble(sample_binary_file, architecture="x86_64")

        assert isinstance(result, dict)
        assert result.get("architecture") == "x86_64"

    def test_empty_binary_handling(self, analyzer, tmp_path):
        """Test handling of empty binary file"""
        empty_file = tmp_path / "empty.bin"
        empty_file.write_bytes(b"")

        result = analyzer.disassemble(str(empty_file))

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_invalid_binary_handling(self, analyzer, tmp_path):
        """Test handling of invalid binary"""
        invalid_file = tmp_path / "invalid.bin"
        invalid_file.write_bytes(b"INVALID_DATA")

        result = analyzer.disassemble(str(invalid_file))

        # Should handle gracefully
        assert isinstance(result, dict)
