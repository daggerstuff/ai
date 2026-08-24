"""
NGC CLI Utility Module

Provides utilities for working with NVIDIA GPU Cloud (NGC) CLI to download
NeMo resources, datasets, and other NGC catalog resources.

This module handles:
- NGC CLI detection and installation
- Resource download from NGC catalog
- Configuration management
- Error handling and retry logic
"""

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_TABLE_MIN_PARTS = 3


@dataclass
class NGCConfig:
    """NGC CLI configuration"""

    api_key: str | None = None
    org: str | None = None
    team: str | None = None


class NGCCLIError(Exception):
    """Base exception for NGC CLI operations"""


class NGCCLINotFoundError(NGCCLIError):
    """NGC CLI not found or not installed"""


class NGCCLIAuthError(NGCCLIError):
    """NGC CLI authentication error"""


class NGCCLIDownloadError(NGCCLIError):
    """NGC CLI download error"""


class NGCCLI:
    """
    NGC CLI wrapper for downloading resources from NVIDIA GPU Cloud.

    Supports multiple installation methods:
    1. System-installed ngc in PATH
    2. Local installation at ~/ngc-cli/ngc
    3. Python package via uv (ngc-python-cli)
    """

    def __init__(self, use_uv: bool = True):
        """
        Initialize NGC CLI wrapper.

        Args:
            use_uv: If True, prefer uv-based installation if ngc not in PATH
        """
        self.use_uv = use_uv
        self.ngc_cmd: str | None = None
        self.uv_cmd: str | None = None
        self._detect_ngc_cli()

    def _detect_ngc_cli(self) -> None:
        """Detect and set up NGC CLI command"""
        # Method 1: Check if ngc is in PATH
        if shutil.which("ngc"):
            self.ngc_cmd = "ngc"
            logger.info("Found NGC CLI in PATH")
            return

        # Method 2: Check common installation location
        home_ngc = Path.home() / "ngc-cli" / "ngc"
        if home_ngc.exists():
            self.ngc_cmd = str(home_ngc)
            # Add to PATH for subprocess calls
            env_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{home_ngc.parent}:{env_path}"
            logger.info(f"Found NGC CLI at {home_ngc}")
            return

        # Method 3: Use uv to run ngc (if enabled)
        if self.use_uv:
            self._setup_uv_ngc()

    def _setup_uv_ngc(self) -> None:
        """Set up NGC CLI via uv"""
        # Find uv
        if shutil.which("uv"):
            self.uv_cmd = "uv"
        elif (Path.home() / ".local" / "bin" / "uv").exists():
            self.uv_cmd = str(Path.home() / ".local" / "bin" / "uv")
        elif (Path.home() / ".cargo" / "bin" / "uv").exists():
            self.uv_cmd = str(Path.home() / ".cargo" / "bin" / "uv")
        else:
            logger.warning("uv not found, cannot use uv-based NGC CLI")
            return

        # Check if ngc is installed via uv
        try:
            result = subprocess.run(
                [self.uv_cmd, "pip", "list"],
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )
            if "ngc" in result.stdout.lower():
                self.ngc_cmd = f"{self.uv_cmd} run ngc"
                logger.info("Found NGC CLI via uv")
                return
        except Exception as e:
            logger.debug(f"Error checking uv packages: {e}")

        # Note: NGC CLI is not a Python package on PyPI
        # It must be downloaded from https://catalog.ngc.nvidia.com
        # We can only check if it's available in PATH or local installation
        # The uv method here is for running Python-based NGC SDK if available
        logger.debug("NGC CLI must be installed separately from NVIDIA website")

    def is_available(self) -> bool:
        """Check if NGC CLI is available"""
        return self.ngc_cmd is not None

    def ensure_available(self) -> None:
        """Ensure NGC CLI is available, raise error if not"""
        if not self.is_available():
            raise NGCCLINotFoundError(
                "NGC CLI not found. Please install it:\n"
                "  1. Download from https://catalog.ngc.nvidia.com\n"
                "  2. Or install to ~/ngc-cli/ directory\n"
                "  3. Or add to system PATH\n"
                "\n"
                "Note: NGC CLI is not available as a PyPI package.\n"
                "You must download it directly from NVIDIA."
            )

    def check_config(self) -> dict[str, Any]:
        """
        Check NGC CLI configuration.

        Returns:
            Configuration dictionary with API key status, org, team, etc.

        Raises:
            NGCCLINotFoundError: If NGC CLI is not available
            NGCCLIAuthError: If authentication is not configured
        """
        self.ensure_available()

        if self.ngc_cmd is None:
            raise NGCCLINotFoundError("NGC CLI command not set")
        try:
            result = subprocess.run(
                [*self.ngc_cmd.split(), "config", "current"],
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )

            config = {}
            # Parse the table format output
            lines = result.stdout.strip().split("\n")
            current_key = None

            for line in lines:
                if "|" in line and "| key " not in line.lower() and "---" not in line:
                    parts = [part.strip() for part in line.split("|") if part.strip()]
                    if len(parts) >= CONFIG_TABLE_MIN_PARTS:  # key | value | source
                        key, value, _source = parts[0], parts[1], parts[2]
                        if key:  # New key
                            current_key = key
                            config[key] = value
                        elif current_key and value:  # Continuation of previous key
                            config[current_key] += value
                    elif len(parts) == 1 and current_key:  # Just a value continuation
                        config[current_key] += parts[0]

            # Check if API key is configured (it will be masked with asterisks)
            # If we have any config and apikey exists (even masked), consider it configured
            if config and ("apikey" in config or "API key" in config):
                return config

            raise NGCCLIAuthError(
                "NGC CLI not configured. Run: ngc config set\nGet your API key from: https://catalog.ngc.nvidia.com"
            )

            return config
        except subprocess.CalledProcessError as e:
            raise NGCCLIAuthError(f"Failed to check NGC config: {e.stderr}") from e

    def set_config(self, api_key: str, _org: str | None = None, _team: str | None = None) -> None:
        """
        Configure NGC CLI with API key.

        Args:
            api_key: NGC API key from https://catalog.ngc.nvidia.com
            _org: Optional organization name (reserved for future use)
            _team: Optional team name (reserved for future use)
        """
        self.ensure_available()

        if self.ngc_cmd is None:
            raise NGCCLINotFoundError("NGC CLI command not set")
        # Set API key
        try:
            subprocess.run(
                [*self.ngc_cmd.split(), "config", "set"],
                input=f"{api_key}\n",
                text=True,
                check=True,
                capture_output=True,
                shell=False,
            )
            logger.info("NGC CLI configured successfully")
        except subprocess.CalledProcessError as e:
            raise NGCCLIAuthError(f"Failed to configure NGC CLI: {e.stderr}") from e

    def download_resource(
        self,
        resource_path: str,
        version: str | None = None,
        output_dir: Path | None = None,
        _extract: bool = True,
    ) -> Path:
        """
        Download a resource from NGC catalog.

        Args:
            resource_path: Resource path in format
                "org/team/resource"
                or "nvidia/nemo-microservices/nemo-microservices-quickstart"
            version: Optional version tag (e.g., "25.10")
            output_dir: Optional output directory (defaults to current directory)
            extract: Whether to extract downloaded archive

        Returns:
            Path to downloaded/extracted resource

        Raises:
            NGCCLINotFoundError: If NGC CLI is not available
            NGCCLIAuthError: If authentication failed
            NGCCLIDownloadError: If download failed
        """
        self.ensure_available()

        # Check config first
        try:
            self.check_config()
        except NGCCLIAuthError:
            logger.warning("NGC CLI not configured. Attempting download anyway...")

        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if self.ngc_cmd is None:
            raise NGCCLINotFoundError("NGC CLI command not set")
        # Build download command
        cmd = [*self.ngc_cmd.split(), "registry", "resource", "download-version"]

        resource_spec = f"{resource_path}:{version}" if version else resource_path

        cmd.append(resource_spec)

        # Change to output directory for download
        original_cwd = Path.cwd()
        try:
            return self._execute_download_in_directory(output_dir, resource_spec, cmd)
        finally:
            os.chdir(original_cwd)

    def _execute_download_in_directory(self, output_dir: Path, resource_spec: str, cmd: list[str]) -> Path:
        """
        Execute download command in the specified directory and locate the downloaded resource.

        Args:
            output_dir: Directory to download into
            resource_spec: Resource specification string for logging
            cmd: Command to execute

        Returns:
            Path to the downloaded resource (most recently modified item, or output_dir if empty)

        Raises:
            NGCCLIDownloadError: If download fails
        """
        os.chdir(output_dir)
        logger.info(f"Downloading {resource_spec} to {output_dir}...")

        result = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False)

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            raise NGCCLIDownloadError(f"Failed to download {resource_spec}:\n{error_msg}")

        logger.info(f"Successfully downloaded {resource_spec}")

        if downloaded_items := list(output_dir.iterdir()):
            # Return the most recently modified item
            return max(downloaded_items, key=lambda p: p.stat().st_mtime)

        return output_dir

    def list_resources(self, org: str | None = None, team: str | None = None) -> list[dict[str, Any]]:
        """
        List available resources in NGC catalog.

        Args:
            org: Optional organization filter
            team: Optional team filter

        Returns:
            List of resource dictionaries
        """
        self.ensure_available()

        if self.ngc_cmd is None:
            raise NGCCLINotFoundError("NGC CLI command not set")
        cmd = [*self.ngc_cmd.split(), "registry", "resource", "list"]

        if org:
            cmd.extend(["--org", org])
        if team:
            cmd.extend(["--team", team])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=False)
            raw_output = result.stdout.strip()
            if not raw_output:
                return []
            return self._parse_resources_output(raw_output)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to list resources: {e.stderr}")
            return []

    @staticmethod
    def _parse_resources_output(raw_output: str) -> list[dict[str, Any]]:
        normalized_output = raw_output.lower()
        if "no resources found" in normalized_output or "no results found" in normalized_output:
            return []

        resources = NGCCLI._parse_json_resources(raw_output)
        if resources:
            return resources

        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if not lines:
            return []

        resources = NGCCLI._parse_pipe_delimited_resources(lines)
        if resources:
            return resources

        return NGCCLI._parse_whitespace_aligned_resources(lines)

    @staticmethod
    def _parse_json_resources(raw_output: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            return []

        if isinstance(payload, list):
            return [dict(resource) for resource in payload if isinstance(resource, dict)]
        if isinstance(payload, dict):
            resources = payload.get("resources")
            if isinstance(resources, list):
                return [dict(resource) for resource in resources if isinstance(resource, dict)]
        return []

    @staticmethod
    def _parse_pipe_delimited_resources(lines: list[str]) -> list[dict[str, Any]]:
        table_rows = [line for line in lines if "|" in line and "---" not in line and "+" not in line[:2]]
        if not table_rows:
            return []

        headers = [
            part.strip() for part in table_rows[0].split("|") if part.strip() and not part.strip().startswith("+")
        ]
        if not headers:
            return []

        resources: list[dict[str, Any]] = []
        start_idx = 1
        # Skip table headers that were parsed
        if len(lines) > 1 and "Name" in lines[1] and "Version" in lines[1]:
            start_idx = 2

        for row in lines[start_idx:]:
            if "---" in row or row.startswith("+") or "|" not in row:
                continue
            values = [part.strip() for part in row.split("|") if part.strip() and not part.strip().startswith("+")]
            if len(values) < len(headers):
                continue
            resources.append(dict(zip(headers, values[: len(headers)], strict=True)))
        return resources

    @staticmethod
    def _parse_whitespace_aligned_resources(lines: list[str]) -> list[dict[str, Any]]:
        # Filter out divider rows before processing
        lines = [line for line in lines if not set(line.strip()) <= {"-", "=", " "}]
        if not lines:
            return []
        header_line: str | None = None
        data_start = 0
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped in {"-", "--", "---", "====", "====="}:
                continue
            if " " in line:
                header_line = line
                data_start = idx + 1
                break

        if not header_line:
            return []

        headers = [part.strip() for part in header_line.split("  ") if part.strip()]
        if not headers:
            return []

        resources: list[dict[str, Any]] = []
        for row in lines[data_start:]:
            values = [part.strip() for part in row.split("  ") if part.strip()]
            if len(values) < len(headers):
                continue
            resources.append(dict(zip(headers, values[: len(headers)], strict=True)))
        return resources


def get_ngc_cli(use_uv: bool = True) -> NGCCLI:
    """
    Get an NGC CLI instance.

    Args:
        use_uv: If True, prefer uv-based installation

    Returns:
        NGCCLI instance
    """
    return NGCCLI(use_uv=use_uv)


def ensure_ngc_cli_configured(api_key: str | None = None) -> NGCCLI:
    """
    Ensure NGC CLI is available and configured.

    Args:
        api_key: Optional API key to configure (if not already configured)

    Returns:
        Configured NGCCLI instance

    Raises:
        NGCCLINotFoundError: If NGC CLI cannot be found or installed
        NGCCLIAuthError: If configuration fails
    """
    cli = get_ngc_cli()

    if not cli.is_available():
        raise NGCCLINotFoundError(
            "NGC CLI not available. Install with:\n"
            "  uv pip install nvidia-pyindex nvidia-nim ngc-python-cli\n"
            "Or download from: https://catalog.ngc.nvidia.com"
        )

    # Check if already configured
    try:
        cli.check_config()
        return cli
    except NGCCLIAuthError as err:
        if api_key:
            cli.set_config(api_key)
            return cli
        raise NGCCLIAuthError(
            "NGC CLI not configured. Provide API key or run: ngc config set\n"
            "Get API key from: https://catalog.ngc.nvidia.com"
        ) from err
