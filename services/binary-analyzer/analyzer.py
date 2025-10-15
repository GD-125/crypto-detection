"""
Binary Analysis Module using Ghidra
Performs multi-architecture binary disassembly and analysis
"""

import os
import subprocess
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinaryAnalyzer:
    """
    Binary analyzer using Ghidra for multi-architecture support
    """

    def __init__(self, ghidra_path: Optional[str] = None):
        """
        Initialize Binary Analyzer

        Args:
            ghidra_path: Path to Ghidra installation
        """
        self.ghidra_path = ghidra_path or os.getenv("GHIDRA_HOME", "/opt/ghidra")
        self.project_dir = "data/ghidra_projects"
        os.makedirs(self.project_dir, exist_ok=True)

        logger.info(f"Ghidra path: {self.ghidra_path}")

    def disassemble(self, binary_path: str, architecture: str = "auto") -> Dict[str, Any]:
        """
        Disassemble binary using Ghidra

        Args:
            binary_path: Path to binary file
            architecture: Target architecture (auto-detect if "auto")

        Returns:
            Dictionary containing disassembly results
        """
        try:
            logger.info(f"Starting disassembly of {binary_path}")

            # Detect architecture if auto
            if architecture == "auto":
                architecture = self._detect_architecture(binary_path)
                logger.info(f"Detected architecture: {architecture}")

            # Create Ghidra project
            project_name = self._create_project_name(binary_path)

            # Run Ghidra headless analyzer
            result = self._run_ghidra_headless(binary_path, project_name, architecture)

            # Parse results
            disassembly_data = self._parse_ghidra_output(result)

            # Extract crypto-relevant patterns
            crypto_patterns = self._extract_crypto_patterns(disassembly_data)

            return {
                "binary_path": binary_path,
                "architecture": architecture,
                "disassembly": disassembly_data,
                "crypto_patterns": crypto_patterns,
                "functions": disassembly_data.get("functions", []),
                "strings": disassembly_data.get("strings", []),
                "imports": disassembly_data.get("imports", []),
                "metadata": {
                    "file_size": os.path.getsize(binary_path),
                    "file_hash": self._calculate_hash(binary_path)
                }
            }

        except Exception as e:
            logger.error(f"Disassembly error: {str(e)}")
            return {
                "error": str(e),
                "binary_path": binary_path,
                "architecture": architecture
            }

    def _detect_architecture(self, binary_path: str) -> str:
        """
        Detect binary architecture using file command

        Args:
            binary_path: Path to binary

        Returns:
            Detected architecture string
        """
        try:
            result = subprocess.run(
                ["file", binary_path],
                capture_output=True,
                text=True,
                timeout=10
            )

            output = result.stdout.lower()

            # Common architecture patterns
            if "x86-64" in output or "x86_64" in output:
                return "x86_64"
            elif "x86" in output or "80386" in output:
                return "x86"
            elif "arm64" in output or "aarch64" in output:
                return "arm64"
            elif "arm" in output:
                return "arm"
            elif "mips" in output:
                return "mips"
            elif "powerpc" in output or "ppc" in output:
                return "powerpc"
            else:
                logger.warning("Could not detect architecture, defaulting to x86")
                return "x86"

        except Exception as e:
            logger.error(f"Architecture detection error: {str(e)}")
            return "x86"

    def _create_project_name(self, binary_path: str) -> str:
        """Create unique project name from binary path"""
        filename = os.path.basename(binary_path)
        hash_suffix = hashlib.md5(binary_path.encode()).hexdigest()[:8]
        return f"{filename}_{hash_suffix}"

    def _run_ghidra_headless(
        self,
        binary_path: str,
        project_name: str,
        architecture: str
    ) -> Dict[str, Any]:
        """
        Run Ghidra in headless mode

        Args:
            binary_path: Path to binary
            project_name: Ghidra project name
            architecture: Target architecture

        Returns:
            Analysis results
        """
        try:
            # Ghidra headless analyzer command
            analyze_script = self._get_analysis_script()

            cmd = [
                os.path.join(self.ghidra_path, "support", "analyzeHeadless"),
                self.project_dir,
                project_name,
                "-import", binary_path,
                "-scriptPath", os.path.dirname(analyze_script),
                "-postScript", os.path.basename(analyze_script),
                "-deleteProject"  # Clean up after analysis
            ]

            # Run Ghidra
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                logger.warning(f"Ghidra returned non-zero: {result.stderr}")

            # Parse output
            output_file = os.path.join(
                self.project_dir,
                f"{project_name}_analysis.json"
            )

            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    return json.load(f)
            else:
                # Fallback: parse stdout
                return self._parse_stdout(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("Ghidra analysis timeout")
            return {"error": "Analysis timeout"}
        except Exception as e:
            logger.error(f"Ghidra execution error: {str(e)}")
            return {"error": str(e)}

    def _get_analysis_script(self) -> str:
        """Get or create Ghidra analysis script"""
        script_path = os.path.join(self.project_dir, "crypto_analysis.py")

        # Create basic analysis script if it doesn't exist
        if not os.path.exists(script_path):
            script_content = '''
# Ghidra Analysis Script for Crypto Detection
# @category Cryptography

from ghidra.program.model.listing import *
import json

functions = []
strings = []
imports = []

# Get all functions
function_manager = currentProgram.getFunctionManager()
for func in function_manager.getFunctions(True):
    functions.append({
        "name": func.getName(),
        "address": str(func.getEntryPoint()),
        "size": func.getBody().getNumAddresses()
    })

# Get all strings
string_table = currentProgram.getListing().getDefinedData(True)
for data in string_table:
    if data.hasStringValue():
        strings.append({
            "value": str(data.getValue()),
            "address": str(data.getAddress())
        })

# Get imports
external_manager = currentProgram.getExternalManager()
for ext_name in external_manager.getExternalLibraryNames():
    imports.append(ext_name)

# Output results
output = {
    "functions": functions[:1000],  # Limit output
    "strings": strings[:1000],
    "imports": imports
}

print(json.dumps(output))
'''
            with open(script_path, 'w') as f:
                f.write(script_content)

        return script_path

    def _parse_ghidra_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Ghidra analysis output"""
        if "error" in result:
            return result

        return {
            "functions": result.get("functions", []),
            "strings": result.get("strings", []),
            "imports": result.get("imports", [])
        }

    def _parse_stdout(self, stdout: str) -> Dict[str, Any]:
        """Parse Ghidra stdout for JSON output"""
        try:
            # Look for JSON in stdout
            lines = stdout.split('\n')
            for line in lines:
                if line.strip().startswith('{'):
                    return json.loads(line)
            return {"functions": [], "strings": [], "imports": []}
        except Exception as e:
            logger.error(f"Stdout parsing error: {str(e)}")
            return {"functions": [], "strings": [], "imports": []}

    def _extract_crypto_patterns(self, disassembly_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract cryptography-related patterns from disassembly

        Args:
            disassembly_data: Disassembly results

        Returns:
            List of detected crypto patterns
        """
        patterns = []

        # Known crypto function names
        crypto_keywords = [
            "aes", "rsa", "des", "sha", "md5", "hmac",
            "encrypt", "decrypt", "cipher", "crypto", "hash",
            "key", "sign", "verify", "digest"
        ]

        # Check function names
        for func in disassembly_data.get("functions", []):
            func_name = func.get("name", "").lower()
            for keyword in crypto_keywords:
                if keyword in func_name:
                    patterns.append({
                        "type": "function_name",
                        "keyword": keyword,
                        "function": func.get("name"),
                        "address": func.get("address")
                    })

        # Check strings
        for string in disassembly_data.get("strings", []):
            string_val = string.get("value", "").lower()
            for keyword in crypto_keywords:
                if keyword in string_val:
                    patterns.append({
                        "type": "string",
                        "keyword": keyword,
                        "value": string.get("value"),
                        "address": string.get("address")
                    })

        return patterns

    def _calculate_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
