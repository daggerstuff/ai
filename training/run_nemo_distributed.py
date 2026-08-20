"""Multi-node torchrun launcher for NeMo distributed training.

Builds the torchrun command, configures NCCL/InfiniBand environment,
and provides verification helpers for IB, NVLink, and NCCL collective
performance.

Topology rules (Appendix A):
  - TP within node (NVLink domain)
  - PP/DP/CP across nodes (InfiniBand/RoCE)
  - Cross-node TP is possible but NOT recommended (IB throughput << NVLink)

References:
  - Appendix A (NCCL / InfiniBand), docs/training-pipeline-blueprint-2026-08-10.md
  - Linear PIX-4348
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NPROC_PER_NODE = 8
DEFAULT_NNODES = 2
DEFAULT_NODE_RANK = 0
DEFAULT_RDZV_ID = "llama70b"
DEFAULT_RDZV_BACKEND = "c10d"
DEFAULT_RDZV_PORT = 29500
DEFAULT_MASTER_ADDR = "127.0.0.1"

DEFAULT_NCCL_ENV: dict[str, str] = {
    "NCCL_DEBUG": "INFO",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_HCA": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1",
    "NCCL_P2P_LEVEL": "NVL",
    "NCCL_NET_GDR_LEVEL": "5",
    "NCCL_BUFFSIZE": "8388608",
}

DEFAULT_NCCL_TEST_ARGS = ["-b", "8", "-e", "8G", "-f", "2", "-g", "8"]

MIN_IB_LINK_RATE_GBPS = 200  # 200 / 400 / 800 Gb/s expected


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TorchrunConfig:
    """Configuration for a torchrun launch."""

    nproc_per_node: int = DEFAULT_NPROC_PER_NODE
    nnodes: int = DEFAULT_NNODES
    node_rank: int = DEFAULT_NODE_RANK
    rdzv_id: str = DEFAULT_RDZV_ID
    rdzv_backend: str = DEFAULT_RDZV_BACKEND
    master_addr: str = DEFAULT_MASTER_ADDR
    rdzv_port: int = DEFAULT_RDZV_PORT
    training_script: str = ""
    training_script_args: list[str] = field(default_factory=list)

    @property
    def rdzv_endpoint(self) -> str:
        """Return rdzv_endpoint as master_addr:port."""
        return f"{self.master_addr}:{self.rdzv_port}"


@dataclass
class TopologyConfig:
    """Parallelism topology configuration.

    Topology rules (Appendix A):
      - TP within a single node (NVLink domain)
      - PP/DP/CP across nodes
      - Cross-node TP allowed but discouraged
    """

    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    nproc_per_node: int = DEFAULT_NPROC_PER_NODE

    @property
    def world_size(self) -> int:
        """Total GPUs across all nodes."""
        return self.tp_size * self.pp_size * self.dp_size * self.cp_size

    @property
    def nnodes(self) -> int:
        """Number of nodes required."""
        if self.nproc_per_node == 0:
            return 1
        return max(1, self.world_size // self.nproc_per_node)


@dataclass
class IBLinkInfo:
    """InfiniBand link information from ibstat."""

    port: str = ""
    state: str = ""  # From "State:" line (Active, Down, etc.)
    physical_state: str = ""  # From "Physical state:" line (LinkUp, Disabled, etc.)
    rate: str = ""
    rate_gb: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.state.lower() == "active"

    @property
    def is_link_up(self) -> bool:
        return "up" in self.physical_state.lower() or "up" in self.state.lower()


@dataclass
class NCCLTestResult:
    """Result of an NCCL collective performance test."""

    test_name: str = ""
    passed: bool = False
    output: str = ""
    error: str = ""
    busbw: float = 0.0  # GB/s
    algbw: float = 0.0  # GB/s
    time_us: float = 0.0


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------


def build_torchrun_cmd(
    config: TorchrunConfig,
    training_script: str | None = None,
    script_args: list[str] | None = None,
) -> list[str]:
    """Build the torchrun command list.

    Args:
        config: TorchrunConfig with nproc, nnodes, rdzv settings.
        training_script: Path to the training script (overrides config).
        script_args: Arguments to pass to the training script.

    Returns:
        List of command parts for subprocess.
    """
    script = training_script or config.training_script
    if not script:
        raise ValueError("training_script is required (set config.training_script or pass explicitly)")

    args = script_args if script_args is not None else config.training_script_args

    cmd = [
        "torchrun",
        f"--nproc_per_node={config.nproc_per_node}",
        f"--nnodes={config.nnodes}",
        f"--node_rank={config.node_rank}",
        f"--rdzv_id={config.rdzv_id}",
        f"--rdzv_backend={config.rdzv_backend}",
        f"--rdzv_endpoint={config.rdzv_endpoint}",
    ]

    if args:
        cmd.extend(args)

    cmd.append(script)
    return cmd


def build_nccl_env(
    overrides: dict[str, str] | None = None,
    ib_hcas: list[str] | None = None,
) -> dict[str, str]:
    """Build the NCCL environment variable dict.

    Args:
        overrides: Dict of env var overrides (merged on top of defaults).
        ib_hcas: List of InfiniBand HCA names (e.g. ["mlx5_0:1", "mlx5_1:1"]).
            Overrides NCCL_IB_HCA if provided.

    Returns:
        Dict of NCCL environment variables.
    """
    env = dict(DEFAULT_NCCL_ENV)
    if overrides:
        env.update(overrides)
    if ib_hcas:
        env["NCCL_IB_HCA"] = ",".join(ib_hcas)
    return env


def build_env(
    base_env: dict[str, str] | None = None,
    nccl_env: dict[str, str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Merge OS env, NCCL env, and extras into a single dict for subprocess.

    Args:
        base_env: Starting environment (defaults to os.environ copy).
        nccl_env: NCCL-specific vars (from build_nccl_env).
        extra_env: Additional vars to merge on top.

    Returns:
        Complete environment dict for subprocess.run.
    """
    env = dict(base_env if base_env is not None else os.environ)
    if nccl_env:
        env.update(nccl_env)
    if extra_env:
        env.update(extra_env)
    return env


# ---------------------------------------------------------------------------
# Topology validation
# ---------------------------------------------------------------------------


def validate_topology(topo: TopologyConfig) -> list[str]:
    """Validate topology configuration against Appendix A rules.

    Rules:
      - TP must fit within a single node (tp_size <= nproc_per_node)
      - Cross-node TP is flagged as a warning (not error)
      - World size must match tp * pp * dp * cp
      - nnodes must be >= 1

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    if topo.tp_size < 1:
        errors.append(f"tp_size must be >= 1, got {topo.tp_size}")
    if topo.pp_size < 1:
        errors.append(f"pp_size must be >= 1, got {topo.pp_size}")
    if topo.dp_size < 1:
        errors.append(f"dp_size must be >= 1, got {topo.dp_size}")
    if topo.cp_size < 1:
        errors.append(f"cp_size must be >= 1, got {topo.cp_size}")
    if topo.ep_size < 1:
        errors.append(f"ep_size must be >= 1, got {topo.ep_size}")
    if topo.nproc_per_node < 1:
        errors.append(f"nproc_per_node must be >= 1, got {topo.nproc_per_node}")

    if errors:
        return errors

    # TP within node check
    if topo.tp_size > topo.nproc_per_node:
        errors.append(
            f"TP size ({topo.tp_size}) exceeds nproc_per_node ({topo.nproc_per_node}). "
            "Cross-node TP requires InfiniBand and is NOT recommended "
            "(NVLink throughput >> IB). Preferred: TP per node → PP across nodes → CP."
        )

    # World size consistency
    expected_world = topo.tp_size * topo.pp_size * topo.dp_size * topo.cp_size
    if expected_world != topo.world_size:
        errors.append(
            f"World size mismatch: tp({topo.tp_size}) * pp({topo.pp_size}) * "
            f"dp({topo.dp_size}) * cp({topo.cp_size}) = {expected_world} "
            f"but world_size = {topo.world_size}"
        )

    # Node count
    if topo.nnodes < 1:
        errors.append(f"Computed nnodes = {topo.nnodes}, must be >= 1")

    return errors


def topology_warnings(topo: TopologyConfig) -> list[str]:
    """Return non-fatal topology warnings.

    These are best-practice advisories from Appendix A.
    """
    warnings: list[str] = []

    if topo.tp_size > topo.nproc_per_node:
        warnings.append(
            "Cross-node TP detected. InfiniBand/RoCE required. "
            "Consider TP/node → PP across nodes → CP for sequence dim."
        )

    if topo.pp_size > 1 and topo.tp_size == 1:
        warnings.append(
            "Pipeline parallelism without TP: each pipeline stage runs on a single GPU. "
            "Consider TP within node for better utilization."
        )

    if topo.cp_size > 1:
        warnings.append(
            "Context parallelism requires SDPA or Transformer Engine attention. "
            "SDPBackend.MATH is NOT compatible with DTensor."
        )

    if topo.ep_size > 1:
        warnings.append(
            "Expert parallelism is not supported with MegatronFSDP. Use fsdp2."
        )

    return warnings


# ---------------------------------------------------------------------------
# InfiniBand verification
# ---------------------------------------------------------------------------


def parse_ibstat(output: str) -> list[IBLinkInfo]:
    """Parse ibstat output to extract port states and link rates.

    Args:
        output: stdout from `ibstat` command.

    Returns:
        List of IBLinkInfo for each port found.
    """
    links: list[IBLinkInfo] = []
    current = IBLinkInfo()

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Port ") and ":" in line:
            if current.port:
                links.append(current)
            current = IBLinkInfo(port=line.rstrip(":"))
        elif line.startswith("Physical state:"):
            current.physical_state = line.split(":", 1)[1].strip()
        elif line.startswith("State:"):
            current.state = line.split(":", 1)[1].strip()
        elif line.startswith("Rate:") :
            rate_str = line.split(":", 1)[1].strip()
            current.rate = rate_str
            # Extract numeric Gb/s from strings like "200 Gb/s (4x EDR)"
            try:
                current.rate_gb = float(rate_str.split()[0])
            except (ValueError, IndexError):
                pass

    if current.port:
        links.append(current)

    return links


def verify_ib(ibstat_output: str | None = None, ibv_devinfo_output: str | None = None) -> dict[str, Any]:
    """Verify InfiniBand connectivity.

    Args:
        ibstat_output: Pre-captured ibstat output (for testing). If None, runs ibstat.
        ibv_devinfo_output: Pre-captured ibv_devinfo output. If None, runs ibv_devinfo.

    Returns:
        Dict with keys: ib_ok (bool), links (list), min_rate_gb (float),
        messages (list of str).
    """
    messages: list[str] = []
    links: list[IBLinkInfo] = []

    if ibstat_output is not None:
        links = parse_ibstat(ibstat_output)
    else:
        ibstat_bin = shutil.which("ibstat")
        if ibstat_bin:
            try:
                result = subprocess.run(
                    [ibstat_bin], capture_output=True, text=True, timeout=30, check=False
                )
                if result.returncode == 0:
                    links = parse_ibstat(result.stdout)
                else:
                    messages.append(f"ibstat failed (rc={result.returncode}): {result.stderr.strip()}")
            except subprocess.TimeoutExpired:
                messages.append("ibstat timed out after 30s")
            except Exception as e:
                messages.append(f"ibstat error: {e}")
        else:
            messages.append("ibstat not found in PATH")

    # ibv_devinfo
    min_rate = 0.0
    if ibv_devinfo_output is not None:
        # Parse for link rate
        for line in ibv_devinfo_output.splitlines():
            if "hca_id" in line.lower() or "port" in line.lower():
                messages.append(f"ibv_devinfo: {line.strip()}")
            if "link_layer" in line.lower():
                messages.append(f"ibv_devinfo: {line.strip()}")
    else:
        ibv_bin = shutil.which("ibv_devinfo")
        if ibv_bin:
            try:
                result = subprocess.run(
                    [ibv_bin], capture_output=True, text=True, timeout=30, check=False
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if "hca_id" in line.lower() or "port" in line.lower():
                            messages.append(f"ibv_devinfo: {line.strip()}")
                else:
                    messages.append(f"ibv_devinfo failed (rc={result.returncode}): {result.stderr.strip()}")
            except subprocess.TimeoutExpired:
                messages.append("ibv_devinfo timed out after 30s")
            except Exception as e:
                messages.append(f"ibv_devinfo error: {e}")
        else:
            messages.append("ibv_devinfo not found in PATH")

    # Evaluate links
    ib_ok = True
    if links:
        all_active = all(l.is_active for l in links)
        min_rate = min((l.rate_gb for l in links if l.rate_gb > 0), default=0.0)
        if not all_active:
            ib_ok = False
            inactive = [l.port for l in links if not l.is_active]
            messages.append(f"Inactive IB ports: {inactive}")
        if min_rate < MIN_IB_LINK_RATE_GBPS:
            ib_ok = False
            messages.append(
                f"IB link rate {min_rate} Gb/s below minimum {MIN_IB_LINK_RATE_GBPS} Gb/s"
            )
    else:
        # If no links found and ibstat ran, that's a problem
        if not any("not found" in m or "failed" in m or "timed out" in m for m in messages):
            ib_ok = False
            messages.append("No IB links detected")

    return {
        "ib_ok": ib_ok,
        "links": links,
        "min_rate_gb": min_rate,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# NVLink verification
# ---------------------------------------------------------------------------


def verify_nvlink(nvidia_smi_output: str | None = None) -> dict[str, Any]:
    """Verify NVLink connectivity via nvidia-smi.

    Args:
        nvidia_smi_output: Pre-captured nvidia-smi nvlink output. If None, runs nvidia-smi.

    Returns:
        Dict with keys: nvlink_ok (bool), gpu_count (int), link_counts (dict),
        messages (list).
    """
    messages: list[str] = []
    link_counts: dict[int, int] = {}

    if nvidia_smi_output is not None:
        output = nvidia_smi_output
    else:
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            messages.append("nvidia-smi not found in PATH")
            return {
                "nvlink_ok": False,
                "gpu_count": 0,
                "link_counts": {},
                "messages": messages,
            }
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "nvlink",
                    "--status",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            output = result.stdout
            if result.returncode != 0:
                messages.append(f"nvidia-smi nvlink failed (rc={result.returncode}): {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            messages.append("nvidia-smi nvlink timed out after 30s")
            return {
                "nvlink_ok": False,
                "gpu_count": 0,
                "link_counts": {},
                "messages": messages,
            }
        except Exception as e:
            messages.append(f"nvidia-smi nvlink error: {e}")
            return {
                "nvlink_ok": False,
                "gpu_count": 0,
                "link_counts": {},
                "messages": messages,
            }

    # Parse NVLink status output
    # Typical format: "GPU 00000000:XX:XX.X NVIDIA H100 NVLink: 18 links"
    gpu_idx = -1
    for line in output.splitlines():
        line = line.strip()
        if "GPU" in line and "NVLink" in line and "links" in line.lower():
            tokens = line.split()
            # Find the number right before "links"
            for i, token in enumerate(tokens):
                if "links" in token.lower() and i > 0:
                    prev = tokens[i - 1]
                    if prev.isdigit():
                        gpu_idx += 1
                        link_counts[gpu_idx] = int(prev)
                    break

    nvlink_ok = len(link_counts) > 0 and all(v > 0 for v in link_counts.values())

    if not link_counts:
        messages.append("No NVLink connections detected")

    return {
        "nvlink_ok": nvlink_ok,
        "gpu_count": len(link_counts),
        "link_counts": link_counts,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# NCCL test runner
# ---------------------------------------------------------------------------


def run_nccl_test(
    test_name: str = "all_reduce_perf",
    test_args: list[str] | None = None,
    test_bin_dir: str = "/usr/local/nccl-tests/build",
    env: dict[str, str] | None = None,
    timeout: int = 300,
) -> NCCLTestResult:
    """Run an NCCL collective performance test.

    Args:
        test_name: Name of the NCCL test binary (e.g. all_reduce_perf).
        test_args: Arguments for the test (defaults to all_reduce_perf args).
        test_bin_dir: Directory containing nccl-tests binaries.
        env: Environment dict for subprocess.
        timeout: Timeout in seconds.

    Returns:
        NCCLTestResult with parsed output.
    """
    args = test_args if test_args is not None else DEFAULT_NCCL_TEST_ARGS
    test_path = os.path.join(test_bin_dir, test_name)

    if not os.path.isfile(test_path):
        return NCCLTestResult(
            test_name=test_name,
            passed=False,
            error=f"Test binary not found: {test_path}",
        )

    run_env = env or dict(os.environ)

    try:
        result = subprocess.run(
            [test_path] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=run_env,
        )
    except subprocess.TimeoutExpired:
        return NCCLTestResult(
            test_name=test_name,
            passed=False,
            error=f"Timed out after {timeout}s",
        )
    except Exception as e:
        return NCCLTestResult(
            test_name=test_name,
            passed=False,
            error=str(e),
        )

    return _parse_nccl_test_output(result.stdout, result.stderr, result.returncode, test_name)


def _parse_nccl_test_output(
    stdout: str, stderr: str, returncode: int, test_name: str
) -> NCCLTestResult:
    """Parse NCCL test output for bandwidth and timing metrics.

    NCCL test output format (tab-separated):
    #       size    count   type    redop   time  algbw  busbw  error
    0        8      ...
    """
    result = NCCLTestResult(
        test_name=test_name,
        output=stdout,
        error=stderr,
        passed=returncode == 0,
    )

    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # Find redop keyword (sum, max, prod, min), then next 3 values are time/algbw/busbw
        redops = {"sum", "max", "prod", "min"}
        for i, part in enumerate(parts):
            if part.lower() in redops and i + 3 < len(parts):
                try:
                    result.time_us = float(parts[i + 1])
                    result.algbw = float(parts[i + 2])
                    result.busbw = float(parts[i + 3])
                except (ValueError, IndexError):
                    pass
                break

    return result


# ---------------------------------------------------------------------------
# Full verification pipeline
# ---------------------------------------------------------------------------


def run_verification(
    skip_ib: bool = False,
    skip_nvlink: bool = False,
    skip_nccl: bool = True,
    nccl_test_bin_dir: str = "/usr/local/nccl-tests/build",
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the full verification pipeline (IB → NVLink → NCCL tests).

    Args:
        skip_ib: Skip InfiniBand verification.
        skip_nvlink: Skip NVLink verification.
        skip_nccl: Skip NCCL collective tests (default True — needs GPU cluster).
        nccl_test_bin_dir: Path to nccl-tests build directory.
        env: Environment for subprocess calls.

    Returns:
        Dict with ib_result, nvlink_result, nccl_result, all_ok.
    """
    results: dict[str, Any] = {}

    if not skip_ib:
        results["ib_result"] = verify_ib()
    else:
        results["ib_result"] = {"ib_ok": None, "skipped": True}

    if not skip_nvlink:
        results["nvlink_result"] = verify_nvlink()
    else:
        results["nvlink_result"] = {"nvlink_ok": None, "skipped": True}

    if not skip_nccl:
        results["nccl_result"] = run_nccl_test(
            env=env,
            test_bin_dir=nccl_test_bin_dir,
        )
    else:
        results["nccl_result"] = {"passed": None, "skipped": True}

    ib = results["ib_result"]
    ib_ok = True if ib.get("skipped") else ib.get("ib_ok", True)

    nv = results["nvlink_result"]
    nvlink_ok = True if nv.get("skipped") else nv.get("nvlink_ok", True)

    nccl = results["nccl_result"]
    if isinstance(nccl, NCCLTestResult):
        nccl_ok = nccl.passed
    elif isinstance(nccl, dict):
        nccl_ok = True if nccl.get("skipped") else nccl.get("passed", True)
    else:
        nccl_ok = True

    results["all_ok"] = bool(ib_ok and nvlink_ok and nccl_ok)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point for the multi-node torchrun launcher.

    Usage:
        # Build and print the torchrun command
        python run_nemo_distributed.py --print-cmd --training-script train.py

        # Run verification only
        python run_nemo_distributed.py --verify

        # Launch torchrun
        python run_nemo_distributed.py --training-script train.py \\
            --master-addr 10.0.0.1 --node-rank 0 --nnodes 2

    Returns:
        0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Multi-node torchrun launcher for NeMo distributed training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--training-script",
        type=str,
        default="",
        help="Path to the training script to launch",
    )
    parser.add_argument(
        "--script-args",
        type=str,
        nargs="*",
        default=[],
        help="Arguments to pass to the training script",
    )
    parser.add_argument("--nproc-per-node", type=int, default=DEFAULT_NPROC_PER_NODE)
    parser.add_argument("--nnodes", type=int, default=DEFAULT_NNODES)
    parser.add_argument("--node-rank", type=int, default=DEFAULT_NODE_RANK)
    parser.add_argument("--rdzv-id", type=str, default=DEFAULT_RDZV_ID)
    parser.add_argument("--rdzv-backend", type=str, default=DEFAULT_RDZV_BACKEND)
    parser.add_argument("--master-addr", type=str, default=DEFAULT_MASTER_ADDR)
    parser.add_argument("--rdzv-port", type=int, default=DEFAULT_RDZV_PORT)

    # Topology
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--cp-size", type=int, default=1, help="Context parallel size")
    parser.add_argument("--ep-size", type=int, default=1, help="Expert parallel size")

    # NCCL overrides
    parser.add_argument("--ib-hca", type=str, default="", help="Override NCCL_IB_HCA (comma-separated)")
    parser.add_argument("--nccl-debug", type=str, default="INFO")
    parser.add_argument("--nccl-ib-disable", type=str, default="0")

    # Actions
    parser.add_argument("--print-cmd", action="store_true", help="Print the torchrun command and exit")
    parser.add_argument("--verify", action="store_true", help="Run IB/NVLink/NCCL verification and exit")
    parser.add_argument("--skip-ib", action="store_true", help="Skip IB verification")
    parser.add_argument("--skip-nvlink", action="store_true", help="Skip NVLink verification")
    parser.add_argument("--skip-nccl", action="store_true", help="Skip NCCL tests (default: skip)")
    parser.add_argument("--nccl-test-bin", type=str, default="/usr/local/nccl-tests/build")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    # Build config
    config = TorchrunConfig(
        nproc_per_node=args.nproc_per_node,
        nnodes=args.nnodes,
        node_rank=args.node_rank,
        rdzv_id=args.rdzv_id,
        rdzv_backend=args.rdzv_backend,
        master_addr=args.master_addr,
        rdzv_port=args.rdzv_port,
        training_script=args.training_script,
        training_script_args=args.script_args,
    )

    topo = TopologyConfig(
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        dp_size=args.dp_size,
        cp_size=args.cp_size,
        ep_size=args.ep_size,
        nproc_per_node=args.nproc_per_node,
    )

    # Topology validation
    topo_errors = validate_topology(topo)
    if topo_errors:
        for err in topo_errors:
            logger.error(err)
        if not args.json:
            print("ERROR: Topology validation failed:", file=sys.stderr)
            for err in topo_errors:
                print(f"  - {err}", file=sys.stderr)
        return 1

    topo_warns = topology_warnings(topo)
    for w in topo_warns:
        logger.warning(w)

    # Verify mode
    if args.verify:
        results = run_verification(
            skip_ib=args.skip_ib,
            skip_nvlink=args.skip_nvlink,
            skip_nccl=args.skip_nccl,
            nccl_test_bin_dir=args.nccl_test_bin,
        )
        if args.json:
            # Serialize dataclasses
            print(json.dumps(results, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o), indent=2))
        else:
            _print_verification(results)
        return 0 if results["all_ok"] else 1

    # Build NCCL env
    overrides: dict[str, str] = {
        "NCCL_DEBUG": args.nccl_debug,
        "NCCL_IB_DISABLE": args.nccl_ib_disable,
    }
    ib_hcas = [h.strip() for h in args.ib_hca.split(",") if h.strip()] if args.ib_hca else None
    nccl_env = build_nccl_env(overrides=overrides, ib_hcas=ib_hcas)

    # Build command
    try:
        cmd = build_torchrun_cmd(config)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    full_env = build_env(nccl_env=nccl_env)

    if args.print_cmd or args.dry_run:
        if args.json:
            print(json.dumps({
                "cmd": cmd,
                "env": nccl_env,
                "topology": {
                    "tp": topo.tp_size,
                    "pp": topo.pp_size,
                    "dp": topo.dp_size,
                    "cp": topo.cp_size,
                    "world_size": topo.world_size,
                    "nnodes": topo.nnodes,
                },
            }, indent=2))
        else:
            print("Command:", " ".join(cmd))
            print("NCCL env:")
            for k, v in sorted(nccl_env.items()):
                print(f"  {k}={v}")
            print(f"World size: {topo.world_size} ({topo.nnodes} nodes × {topo.nproc_per_node} GPUs)")
        return 0

    # Launch
    if not args.training_script:
        print("Error: --training-script is required to launch", file=sys.stderr)
        return 1

    logger.info("Launching torchrun: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, env=full_env, check=False)
        return result.returncode
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error("torchrun failed: %s", e)
        return 1


def _print_verification(results: dict[str, Any]) -> None:
    """Print verification results in human-readable format."""
    ib = results.get("ib_result", {})
    if not ib.get("skipped"):
        status = "✅ PASS" if ib.get("ib_ok") else "❌ FAIL"
        print(f"InfiniBand: {status}")
        if ib.get("min_rate_gb"):
            print(f"  Min link rate: {ib['min_rate_gb']} Gb/s")
        for msg in ib.get("messages", []):
            print(f"  {msg}")
    else:
        print("InfiniBand: SKIPPED")

    nv = results.get("nvlink_result", {})
    if not nv.get("skipped"):
        status = "✅ PASS" if nv.get("nvlink_ok") else "❌ FAIL"
        print(f"NVLink: {status}")
        if nv.get("gpu_count"):
            print(f"  GPUs with NVLink: {nv['gpu_count']}")
    else:
        print("NVLink: SKIPPED")

    nccl = results.get("nccl_result", {})
    if not isinstance(nccl, dict) or not nccl.get("skipped"):
        if isinstance(nccl, NCCLTestResult):
            status = "✅ PASS" if nccl.passed else "❌ FAIL"
            print(f"NCCL {nccl.test_name}: {status}")
            if nccl.busbw:
                print(f"  Bus bandwidth: {nccl.busbw} GB/s")
            if nccl.algbw:
                print(f"  Alg bandwidth: {nccl.algbw} GB/s")
        elif isinstance(nccl, dict) and not nccl.get("skipped"):
            status = "✅ PASS" if nccl.get("passed") else "❌ FAIL"
            print(f"NCCL test: {status}")
    else:
        print("NCCL tests: SKIPPED")

    print(f"\nOverall: {'✅ ALL PASS' if results.get('all_ok') else '❌ FAILURES'}")


if __name__ == "__main__":
    sys.exit(main())
