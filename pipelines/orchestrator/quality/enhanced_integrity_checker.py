"""
Enhanced Dataset Integrity Checker

Advanced integrity checking system that enhances existing dataset validation
with deep content analysis, format verification, and corruption detection.
"""

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import magic
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class IntegrityCheckResult:
    """Result of enhanced integrity check."""

    file_path: str
    is_valid: bool
    file_size: int
    checksum: str
    format_valid: bool
    content_valid: bool
    encoding_valid: bool
    structure_valid: bool
    issues: list[str]
    check_timestamp: str


class EnhancedIntegrityChecker:
    """Enhanced integrity checking for dataset files."""

    def __init__(self):
        self.logger = get_logger(__name__)

        # Supported formats and their validators
        self.format_validators = {
            ".json": self._validate_json_format,
            ".jsonl": self._validate_jsonl_format,
            ".csv": self._validate_csv_format,
            ".txt": self._validate_text_format,
            ".zip": self._validate_zip_format,
        }

        # Expected file signatures (magic numbers)
        self.file_signatures = {
            ".json": ["application/json", "text/plain"],
            ".jsonl": ["application/json", "text/plain"],
            ".csv": ["text/csv", "text/plain"],
            ".txt": ["text/plain"],
            ".zip": ["application/zip"],
        }

        logger.info("EnhancedIntegrityChecker initialized")

    def check_file_integrity(self, file_path: str) -> IntegrityCheckResult:
        """Perform comprehensive integrity check on a file."""
        logger.debug(f"Checking integrity of: {file_path}")

        issues = []
        file_size = 0
        checksum = ""
        format_valid = False
        content_valid = False
        encoding_valid = False
        structure_valid = False

        try:
            # Basic file existence and size check
            if not os.path.exists(file_path):
                issues.append("File does not exist")
                return IntegrityCheckResult(
                    file_path=file_path,
                    is_valid=False,
                    file_size=0,
                    checksum="",
                    format_valid=False,
                    content_valid=False,
                    encoding_valid=False,
                    structure_valid=False,
                    issues=issues,
                    check_timestamp=datetime.now().isoformat(),
                )

            file_size = os.path.getsize(file_path)

            # Calculate checksum
            checksum = self._calculate_checksum(file_path)

            # Check file format
            format_valid = self._check_file_format(file_path, issues)

            # Check encoding
            encoding_valid = self._check_encoding(file_path, issues)

            # Check content validity
            content_valid = self._check_content_validity(file_path, issues)

            # Check structure validity
            structure_valid = self._check_structure_validity(file_path, issues)

            is_valid = (
                format_valid
                and content_valid
                and encoding_valid
                and structure_valid
                and file_size > 0
            )

            return IntegrityCheckResult(
                file_path=file_path,
                is_valid=is_valid,
                file_size=file_size,
                checksum=checksum,
                format_valid=format_valid,
                content_valid=content_valid,
                encoding_valid=encoding_valid,
                structure_valid=structure_valid,
                issues=issues,
                check_timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            issues.append(f"Integrity check failed: {e!s}")
            logger.error(f"Integrity check failed for {file_path}: {e}")

            return IntegrityCheckResult(
                file_path=file_path,
                is_valid=False,
                file_size=file_size,
                checksum=checksum,
                format_valid=False,
                content_valid=False,
                encoding_valid=False,
                structure_valid=False,
                issues=issues,
                check_timestamp=datetime.now().isoformat(),
            )

    def check_dataset_integrity(self, dataset_path: str) -> list[IntegrityCheckResult]:
        """Check integrity of entire dataset."""
        logger.info(f"Checking dataset integrity: {dataset_path}")

        results = []

        for root, _dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                result = self.check_file_integrity(file_path)
                results.append(result)

        logger.info(f"Integrity check completed for {len(results)} files")
        return results

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of file."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Checksum calculation failed for {file_path}: {e}")
            return ""

    def _check_file_format(self, file_path: str, issues: list[str]) -> bool:
        """Check if file format matches extension."""
        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext not in self.file_signatures:
                issues.append(f"Unsupported file extension: {file_ext}")
                return False

            # Use python-magic to detect actual file type
            try:
                file_type = magic.from_file(file_path, mime=True)
                expected_types = self.file_signatures[file_ext]

                if file_type not in expected_types:
                    issues.append(
                        f"File type mismatch: expected {expected_types}, got {file_type}"
                    )
                    return False
            except Exception:
                # Fallback to basic extension check if magic fails
                logger.warning(
                    f"Magic detection failed for {file_path}, using extension check"
                )

            return True

        except Exception as e:
            issues.append(f"Format check failed: {e!s}")
            return False

    def _check_encoding(self, file_path: str, issues: list[str]) -> bool:
        """Check file encoding validity."""
        try:
            # Try to read file with UTF-8 encoding
            with open(file_path, encoding="utf-8") as f:
                f.read(1024)  # Read first 1KB to check encoding
            return True

        except UnicodeDecodeError:
            # Try other common encodings
            encodings = ["latin-1", "cp1252", "iso-8859-1"]
            for encoding in encodings:
                try:
                    with open(file_path, encoding=encoding) as f:
                        f.read(1024)
                    issues.append(f"File uses {encoding} encoding instead of UTF-8")
                    return True
                except UnicodeDecodeError:
                    continue

            issues.append("File has invalid or unsupported encoding")
            return False

        except Exception as e:
            issues.append(f"Encoding check failed: {e!s}")
            return False

    def _check_content_validity(self, file_path: str, issues: list[str]) -> bool:
        """Check content validity based on file type."""
        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext in self.format_validators:
                return self.format_validators[file_ext](file_path, issues)
            # For unsupported formats, just check if file is readable
            with open(file_path, encoding="utf-8") as f:
                f.read()
            return True

        except Exception as e:
            issues.append(f"Content validity check failed: {e!s}")
            return False

    def _check_structure_validity(self, file_path: str, issues: list[str]) -> bool:
        """Check structural validity of dataset files."""
        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".json":
                return self._check_json_structure(file_path, issues)
            if file_ext == ".jsonl":
                return self._check_jsonl_structure(file_path, issues)
            if file_ext == ".csv":
                return self._check_csv_structure(file_path, issues)
            return True  # No specific structure requirements

        except Exception as e:
            issues.append(f"Structure validity check failed: {e!s}")
            return False

    def _validate_json_format(self, file_path: str, issues: list[str]) -> bool:
        """Validate JSON file format."""
        try:
            with open(file_path, encoding="utf-8") as f:
                json.load(f)
            return True
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON format: {e!s}")
            return False

    def _validate_jsonl_format(self, file_path: str, issues: list[str]) -> bool:
        """Validate JSONL file format."""
        try:
            line_count = 0
            with open(file_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            json.loads(line)
                            line_count += 1
                        except json.JSONDecodeError as e:
                            issues.append(f"Invalid JSON on line {line_num}: {e!s}")
                            return False

            if line_count == 0:
                issues.append("JSONL file contains no valid JSON lines")
                return False

            return True
        except Exception as e:
            issues.append(f"JSONL validation failed: {e!s}")
            return False

    def _validate_csv_format(self, file_path: str, issues: list[str]) -> bool:
        """Validate CSV file format."""
        try:
            import csv

            with open(file_path, encoding="utf-8") as f:
                # Try to detect dialect
                sample = f.read(1024)
                f.seek(0)

                try:
                    dialect = csv.Sniffer().sniff(sample)
                except csv.Error:
                    # Use default dialect if detection fails
                    dialect = csv.excel

                reader = csv.reader(f, dialect)
                row_count = 0
                for _row in reader:
                    row_count += 1
                    if row_count > 1000:  # Limit check to first 1000 rows
                        break

                if row_count == 0:
                    issues.append("CSV file is empty")
                    return False

            return True
        except Exception as e:
            issues.append(f"CSV validation failed: {e!s}")
            return False

    def _validate_text_format(self, file_path: str, issues: list[str]) -> bool:
        """Validate text file format."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                if len(content.strip()) == 0:
                    issues.append("Text file is empty")
                    return False
            return True
        except Exception as e:
            issues.append(f"Text validation failed: {e!s}")
            return False

    def _validate_zip_format(self, file_path: str, issues: list[str]) -> bool:
        """Validate ZIP file format."""
        try:
            with zipfile.ZipFile(file_path, "r") as zip_file:
                # Test the ZIP file
                bad_file = zip_file.testzip()
                if bad_file:
                    issues.append(f"Corrupted file in ZIP: {bad_file}")
                    return False
            return True
        except zipfile.BadZipFile:
            issues.append("Invalid or corrupted ZIP file")
            return False
        except Exception as e:
            issues.append(f"ZIP validation failed: {e!s}")
            return False

    def _check_json_structure(self, file_path: str, issues: list[str]) -> bool:
        """Check JSON structure for dataset compliance."""
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Check for common dataset structures
            if isinstance(data, dict):
                if "conversations" in data or "messages" in data or "data" in data:
                    return True
                issues.append(
                    "JSON structure doesn't match expected dataset format"
                )
                return False
            if isinstance(data, list):
                if len(data) > 0:
                    return True
                issues.append("JSON array is empty")
                return False
            issues.append("JSON root must be object or array")
            return False

        except Exception as e:
            issues.append(f"JSON structure check failed: {e!s}")
            return False

    def _check_jsonl_structure(self, file_path: str, issues: list[str]) -> bool:
        """Check JSONL structure for dataset compliance."""
        try:
            valid_lines = 0
            with open(file_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                valid_lines += 1
                            else:
                                issues.append(f"Line {line_num}: Expected JSON object")
                        except json.JSONDecodeError:
                            pass  # Already handled in format validation

            if valid_lines == 0:
                issues.append("No valid JSON objects found in JSONL file")
                return False

            return True
        except Exception as e:
            issues.append(f"JSONL structure check failed: {e!s}")
            return False

    def _check_csv_structure(self, file_path: str, issues: list[str]) -> bool:
        """Check CSV structure for dataset compliance."""
        try:
            import csv

            with open(file_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader, None)

                if not headers:
                    issues.append("CSV file has no headers")
                    return False

                # Check for common dataset columns
                header_lower = [h.lower() for h in headers]
                expected_columns = [
                    "content",
                    "text",
                    "message",
                    "conversation",
                    "input",
                    "output",
                ]

                if not any(col in header_lower for col in expected_columns):
                    issues.append("CSV doesn't contain expected dataset columns")
                    return False

            return True
        except Exception as e:
            issues.append(f"CSV structure check failed: {e!s}")
            return False

    def generate_integrity_report(
        self,
        results: list[IntegrityCheckResult],
        output_path: str = "integrity_check_report.json",
    ) -> str:
        """Generate comprehensive integrity check report."""
        valid_files = sum(1 for r in results if r.is_valid)
        total_size = sum(r.file_size for r in results)

        report = {
            "report_type": "Enhanced Dataset Integrity Check",
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_files": len(results),
                "valid_files": valid_files,
                "invalid_files": len(results) - valid_files,
                "validity_rate": valid_files / len(results) if results else 0,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            },
            "detailed_results": [
                {
                    "file_path": r.file_path,
                    "is_valid": r.is_valid,
                    "file_size": r.file_size,
                    "checksum": r.checksum,
                    "format_valid": r.format_valid,
                    "content_valid": r.content_valid,
                    "encoding_valid": r.encoding_valid,
                    "structure_valid": r.structure_valid,
                    "issues": r.issues,
                    "check_timestamp": r.check_timestamp,
                }
                for r in results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Integrity check report generated: {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    checker = EnhancedIntegrityChecker()

    # Test single file
    result = checker.check_file_integrity("./test_file.json")

    # Test entire dataset
    results = checker.check_dataset_integrity("./test_dataset")
    report_path = checker.generate_integrity_report(results)
