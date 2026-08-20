"""Tests for run_nemo_distributed.py — multi-node torchrun launcher.

All subprocess calls are mocked; no GPU, IB hardware, or torchrun binary required.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.run_nemo_distributed import (
    DEFAULT_NCCL_ENV,
    DEFAULT_NCCL_TEST_ARGS,
    IBLinkInfo,
    NCCLTestResult,
    TorchrunConfig,
    TopologyConfig,
    _parse_nccl_test_output,
    build_env,
    build_nccl_env,
    build_torchrun_cmd,
    main,
    parse_ibstat,
    run_nccl_test,
    run_verification,
    topology_warnings,
    validate_topology,
    verify_ib,
    verify_nvlink,
)


# ---------------------------------------------------------------------------
# TorchrunConfig
# ---------------------------------------------------------------------------


class TestTorchrunConfig:
    def test_defaults(self):
        c = TorchrunConfig()
        assert c.nproc_per_node == 8
        assert c.nnodes == 2
        assert c.node_rank == 0
        assert c.rdzv_id == "llama70b"
        assert c.rdzv_backend == "c10d"
        assert c.rdzv_port == 29500

    def test_rdzv_endpoint(self):
        c = TorchrunConfig(master_addr="10.0.0.1", rdzv_port=29500)
        assert c.rdzv_endpoint == "10.0.0.1:29500"

    def test_rdzv_endpoint_custom_port(self):
        c = TorchrunConfig(master_addr="192.168.1.100", rdzv_port=12345)
        assert c.rdzv_endpoint == "192.168.1.100:12345"


# ---------------------------------------------------------------------------
# build_torchrun_cmd
# ---------------------------------------------------------------------------


class TestBuildTorchrunCmd:
    def test_basic_command(self):
        c = TorchrunConfig(training_script="train.py")
        cmd = build_torchrun_cmd(c)
        assert cmd[0] == "torchrun"
        assert "--nproc_per_node=8" in cmd
        assert "--nnodes=2" in cmd
        assert "--node_rank=0" in cmd
        assert f"--rdzv_id={c.rdzv_id}" in cmd
        assert f"--rdzv_backend={c.rdzv_backend}" in cmd
        assert "--rdzv_endpoint=127.0.0.1:29500" in cmd
        assert cmd[-1] == "train.py"

    def test_custom_config(self):
        c = TorchrunConfig(
            nproc_per_node=4,
            nnodes=4,
            node_rank=2,
            rdzv_id="qwen32b",
            master_addr="10.0.0.1",
            rdzv_port=29501,
            training_script="launch.py",
        )
        cmd = build_torchrun_cmd(c)
        assert "--nproc_per_node=4" in cmd
        assert "--nnodes=4" in cmd
        assert "--node_rank=2" in cmd
        assert "--rdzv_id=qwen32b" in cmd
        assert "--rdzv_endpoint=10.0.0.1:29501" in cmd
        assert cmd[-1] == "launch.py"

    def test_script_args(self):
        c = TorchrunConfig(training_script="train.py")
        cmd = build_torchrun_cmd(c, script_args=["--batch-size", "32", "--epochs", "3"])
        assert "--batch-size" in cmd
        assert "32" in cmd
        assert "--epochs" in cmd
        assert "3" in cmd
        assert cmd[-1] == "train.py"

    def test_no_training_script_raises(self):
        c = TorchrunConfig()
        with pytest.raises(ValueError, match="training_script"):
            build_torchrun_cmd(c)

    def test_explicit_script_overrides_config(self):
        c = TorchrunConfig(training_script="config_script.py")
        cmd = build_torchrun_cmd(c, training_script="override.py")
        assert cmd[-1] == "override.py"


# ---------------------------------------------------------------------------
# build_nccl_env
# ---------------------------------------------------------------------------


class TestBuildNcclEnv:
    def test_defaults(self):
        env = build_nccl_env()
        assert env["NCCL_DEBUG"] == "INFO"
        assert env["NCCL_IB_DISABLE"] == "0"
        assert env["NCCL_IB_GID_INDEX"] == "3"
        assert env["NCCL_P2P_LEVEL"] == "NVL"
        assert env["NCCL_NET_GDR_LEVEL"] == "5"
        assert env["NCCL_BUFFSIZE"] == "8388608"

    def test_overrides(self):
        env = build_nccl_env(overrides={"NCCL_DEBUG": "WARN"})
        assert env["NCCL_DEBUG"] == "WARN"
        # Other defaults preserved
        assert env["NCCL_IB_DISABLE"] == "0"

    def test_ib_hca_override(self):
        env = build_nccl_env(ib_hcas=["mlx5_0:1", "mlx5_1:1"])
        assert env["NCCL_IB_HCA"] == "mlx5_0:1,mlx5_1:1"

    def test_ib_hca_overrides_default(self):
        env = build_nccl_env(ib_hcas=["mlx5_4:1"])
        assert env["NCCL_IB_HCA"] == "mlx5_4:1"

    def test_empty_ib_hcas(self):
        env = build_nccl_env(ib_hcas=[])
        # Empty list should not override
        assert env["NCCL_IB_HCA"] == DEFAULT_NCCL_ENV["NCCL_IB_HCA"]


# ---------------------------------------------------------------------------
# build_env
# ---------------------------------------------------------------------------


class TestBuildEnv:
    def test_merges_nccl_into_os_env(self):
        nccl = build_nccl_env()
        env = build_env(nccl_env=nccl)
        assert "NCCL_DEBUG" in env
        assert env["NCCL_DEBUG"] == "INFO"
        assert "PATH" in env or "path" in env  # from os.environ

    def test_extra_env_overrides(self):
        env = build_env(
            base_env={"A": "1", "B": "2"},
            nccl_env={"B": "3"},
            extra_env={"C": "4"},
        )
        assert env["A"] == "1"
        assert env["B"] == "3"  # nccl overrides base
        assert env["C"] == "4"

    def test_no_nccl_no_extra(self):
        env = build_env(base_env={"X": "1"})
        assert env == {"X": "1"}


# ---------------------------------------------------------------------------
# TopologyConfig
# ---------------------------------------------------------------------------


class TestTopologyConfig:
    def test_world_size(self):
        t = TopologyConfig(tp_size=2, pp_size=2, dp_size=2, cp_size=1)
        assert t.world_size == 8

    def test_world_size_large(self):
        t = TopologyConfig(tp_size=8, pp_size=4, dp_size=2, cp_size=2)
        assert t.world_size == 128

    def test_nnodes(self):
        t = TopologyConfig(tp_size=8, pp_size=2, dp_size=2, cp_size=1, nproc_per_node=8)
        assert t.nnodes == 4  # 32 / 8

    def test_nnodes_minimum_1(self):
        t = TopologyConfig(tp_size=1, pp_size=1, dp_size=1, cp_size=1, nproc_per_node=8)
        assert t.nnodes == 1


# ---------------------------------------------------------------------------
# validate_topology
# ---------------------------------------------------------------------------


class TestValidateTopology:
    def test_valid_single_node(self):
        t = TopologyConfig(tp_size=8, pp_size=1, dp_size=1, cp_size=1, nproc_per_node=8)
        assert validate_topology(t) == []

    def test_valid_multi_node_pp(self):
        t = TopologyConfig(tp_size=8, pp_size=2, dp_size=1, cp_size=1, nproc_per_node=8)
        assert validate_topology(t) == []

    def test_tp_exceeds_nproc(self):
        t = TopologyConfig(tp_size=16, pp_size=1, dp_size=1, cp_size=1, nproc_per_node=8)
        errors = validate_topology(t)
        assert len(errors) == 1
        assert "Cross-node TP" in errors[0] or "exceeds nproc_per_node" in errors[0]

    def test_negative_tp(self):
        t = TopologyConfig(tp_size=0, nproc_per_node=8)
        errors = validate_topology(t)
        assert any("tp_size" in e for e in errors)

    def test_negative_pp(self):
        t = TopologyConfig(pp_size=0, nproc_per_node=8)
        errors = validate_topology(t)
        assert any("pp_size" in e for e in errors)

    def test_negative_dp(self):
        t = TopologyConfig(dp_size=0, nproc_per_node=8)
        errors = validate_topology(t)
        assert any("dp_size" in e for e in errors)

    def test_negative_cp(self):
        t = TopologyConfig(cp_size=0, nproc_per_node=8)
        errors = validate_topology(t)
        assert any("cp_size" in e for e in errors)

    def test_negative_ep(self):
        t = TopologyConfig(ep_size=0, nproc_per_node=8)
        errors = validate_topology(t)
        assert any("ep_size" in e for e in errors)


# ---------------------------------------------------------------------------
# topology_warnings
# ---------------------------------------------------------------------------


class TestTopologyWarnings:
    def test_cross_node_tp_warning(self):
        t = TopologyConfig(tp_size=16, pp_size=1, dp_size=1, cp_size=1, nproc_per_node=8)
        warns = topology_warnings(t)
        assert any("Cross-node TP" in w for w in warns)

    def test_pp_without_tp_warning(self):
        t = TopologyConfig(tp_size=1, pp_size=2, dp_size=1, cp_size=1, nproc_per_node=8)
        warns = topology_warnings(t)
        assert any("Pipeline parallelism without TP" in w for w in warns)

    def test_cp_warning(self):
        t = TopologyConfig(tp_size=1, pp_size=1, dp_size=1, cp_size=2, nproc_per_node=8)
        warns = topology_warnings(t)
        assert any("Context parallelism" in w for w in warns)

    def test_ep_warning(self):
        t = TopologyConfig(tp_size=1, pp_size=1, dp_size=1, cp_size=1, ep_size=2, nproc_per_node=8)
        warns = topology_warnings(t)
        assert any("Expert parallelism" in w for w in warns)

    def test_no_warnings_for_simple(self):
        t = TopologyConfig(tp_size=8, pp_size=1, dp_size=1, cp_size=1, ep_size=1, nproc_per_node=8)
        assert topology_warnings(t) == []


# ---------------------------------------------------------------------------
# parse_ibstat
# ---------------------------------------------------------------------------


class TestParseIbstat:
    def test_parse_active_link(self):
        output = """CA 'mlx5_0'
    Port 1:
        State: Active
        Physical state: LinkUp
        Rate: 200 Gb/s (4x EDR)
"""
        links = parse_ibstat(output)
        assert len(links) == 1
        assert links[0].port == "Port 1"
        assert links[0].is_active
        assert links[0].rate_gb == 200.0

    def test_parse_multiple_ports(self):
        output = """CA 'mlx5_0'
    Port 1:
        State: Active
        Physical state: LinkUp
        Rate: 400 Gb/s (4x HDR)
    Port 2:
        State: Down
        Physical state: Disabled
        Rate: 100 Gb/s (4x FDR)
"""
        links = parse_ibstat(output)
        assert len(links) == 2
        assert links[0].is_active
        assert links[1].is_active is False

    def test_empty_output(self):
        links = parse_ibstat("")
        assert links == []

    def test_rate_extraction(self):
        output = """CA 'mlx5_0'
    Port 1:
        State: Active
        Rate: 800 Gb/s (4x NDR)
"""
        links = parse_ibstat(output)
        assert links[0].rate_gb == 800.0


# ---------------------------------------------------------------------------
# verify_ib
# ---------------------------------------------------------------------------


class TestVerifyIB:
    def test_active_ib(self):
        ibstat = """CA 'mlx5_0'
    Port 1:
        State: Active
        Physical state: LinkUp
        Rate: 400 Gb/s (4x HDR)
"""
        result = verify_ib(ibstat_output=ibstat, ibv_devinfo_output="")
        assert result["ib_ok"] is True
        assert result["min_rate_gb"] == 400.0
        assert len(result["links"]) == 1

    def test_inactive_ib(self):
        ibstat = """CA 'mlx5_0'
    Port 1:
        State: Down
        Physical state: Disabled
        Rate: 200 Gb/s (4x EDR)
"""
        result = verify_ib(ibstat_output=ibstat, ibv_devinfo_output="")
        assert result["ib_ok"] is False

    def test_low_rate_ib(self):
        ibstat = """CA 'mlx5_0'
    Port 1:
        State: Active
        Rate: 100 Gb/s (4x FDR)
"""
        result = verify_ib(ibstat_output=ibstat, ibv_devinfo_output="")
        assert result["ib_ok"] is False
        assert result["min_rate_gb"] == 100.0

    def test_no_links(self):
        result = verify_ib(ibstat_output="", ibv_devinfo_output="")
        assert result["ib_ok"] is False

    def test_multiple_links_all_active(self):
        ibstat = """CA 'mlx5_0'
    Port 1:
        State: Active
        Rate: 400 Gb/s (4x HDR)
    Port 2:
        State: Active
        Rate: 400 Gb/s (4x HDR)
"""
        result = verify_ib(ibstat_output=ibstat, ibv_devinfo_output="")
        assert result["ib_ok"] is True
        assert result["min_rate_gb"] == 400.0


# ---------------------------------------------------------------------------
# verify_nvlink
# ---------------------------------------------------------------------------


class TestVerifyNVLink:
    def test_nvlink_present(self):
        output = """GPU 00000000:81:00.0 NVIDIA H100 NVLink: 18 links
GPU 00000000:82:00.0 NVIDIA H100 NVLink: 18 links
"""
        result = verify_nvlink(nvidia_smi_output=output)
        assert result["nvlink_ok"] is True
        assert result["gpu_count"] == 2

    def test_no_nvlink(self):
        result = verify_nvlink(nvidia_smi_output="")
        assert result["nvlink_ok"] is False
        assert result["gpu_count"] == 0

    def test_partial_nvlink(self):
        output = """GPU 00000000:81:00.0 NVIDIA H100 NVLink: 18 links
GPU 00000000:82:00.0 NVIDIA H100 NVLink: 0 links
"""
        result = verify_nvlink(nvidia_smi_output=output)
        assert result["nvlink_ok"] is False


# ---------------------------------------------------------------------------
# _parse_nccl_test_output
# ---------------------------------------------------------------------------


class TestParseNCCLTestOutput:
    def test_successful_output(self):
        stdout = """#       size    count   type    redop   time    algbw   busbw   error
#        (B)    (elements)           (us)    (GB/s)  (GB/s)
0           8        2     float    sum     12.5     0.001   0.001   0
0           8G       2G    float    sum     100.0    800.0   1600.0  0
"""
        result = _parse_nccl_test_output(stdout, "", 0, "all_reduce_perf")
        assert result.passed is True
        assert result.busbw == 1600.0
        assert result.algbw == 800.0
        assert result.time_us == 100.0

    def test_failed_output(self):
        result = _parse_nccl_test_output("", "NCCL error", 1, "all_reduce_perf")
        assert result.passed is False

    def test_empty_output(self):
        result = _parse_nccl_test_output("", "", 0, "all_reduce_perf")
        assert result.passed is True
        assert result.busbw == 0.0


# ---------------------------------------------------------------------------
# run_nccl_test
# ---------------------------------------------------------------------------


class TestRunNCCLTest:
    def test_binary_not_found(self):
        result = run_nccl_test(
            test_bin_dir="/nonexistent/path",
            env={},
        )
        assert result.passed is False
        assert "not found" in result.error

    @patch("training.run_nemo_distributed.os.path.isfile", return_value=True)
    @patch("training.run_nemo_distributed.subprocess.run")
    def test_successful_run(self, mock_run, mock_isfile):
        mock_run.return_value = MagicMock(
            stdout="""#       size    count   type    redop   time    algbw   busbw
0           8G       2G    float    sum     100.0    800.0   1600.0
""",
            stderr="",
            returncode=0,
        )
        result = run_nccl_test(
            test_bin_dir="/fake/path",
            env={"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
        )
        assert result.passed is True
        assert result.busbw == 1600.0
        mock_run.assert_called_once()

    @patch("training.run_nemo_distributed.os.path.isfile", return_value=True)
    @patch("training.run_nemo_distributed.subprocess.run")
    def test_timeout(self, mock_run, mock_isfile):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)
        result = run_nccl_test(
            test_bin_dir="/fake/path",
            timeout=5,
            env={},
        )
        assert result.passed is False
        assert "timed out" in result.error.lower()

    @patch("training.run_nemo_distributed.os.path.isfile", return_value=True)
    @patch("training.run_nemo_distributed.subprocess.run")
    def test_custom_args(self, mock_run, mock_isfile):
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        run_nccl_test(
            test_name="all_gather_perf",
            test_args=["-b", "16", "-e", "16G"],
            test_bin_dir="/fake/path",
            env={},
        )
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "/fake/path/all_gather_perf"
        assert "-b" in call_args
        assert "16" in call_args


# ---------------------------------------------------------------------------
# run_verification
# ---------------------------------------------------------------------------


class TestRunVerification:
    def test_all_skipped(self):
        result = run_verification(skip_ib=True, skip_nvlink=True, skip_nccl=True)
        assert result["all_ok"] is True
        assert result["ib_result"]["skipped"] is True
        assert result["nvlink_result"]["skipped"] is True
        assert result["nccl_result"]["skipped"] is True

    def test_ib_only_with_mock(self):
        ibstat = """CA 'mlx5_0'
    Port 1:
        State: Active
        Rate: 400 Gb/s (4x HDR)
"""
        with patch("training.run_nemo_distributed.verify_ib", return_value=verify_ib(ibstat_output=ibstat, ibv_devinfo_output="")):
            result = run_verification(skip_nvlink=True, skip_nccl=True)
        assert result["ib_result"]["ib_ok"] is True
        assert result["all_ok"] is True

    def test_nvlink_fail(self):
        with patch("training.run_nemo_distributed.verify_nvlink", return_value={"nvlink_ok": False, "gpu_count": 0, "link_counts": {}, "messages": ["no nvlink"]}):
            result = run_verification(skip_ib=True, skip_nccl=True)
        assert result["all_ok"] is False


# ---------------------------------------------------------------------------
# CLI (main)
# ---------------------------------------------------------------------------


class TestCLI:
    def test_print_cmd(self, capsys):
        with patch.object(sys, "argv", [
            "run_nemo_distributed.py",
            "--print-cmd",
            "--training-script", "train.py",
            "--nproc-per-node", "8",
            "--nnodes", "2",
            "--master-addr", "10.0.0.1",
        ]):
            rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "torchrun" in captured.out
        assert "--nproc_per_node=8" in captured.out
        assert "10.0.0.1:29500" in captured.out

    def test_print_cmd_json(self, capsys):
        with patch.object(sys, "argv", [
            "run_nemo_distributed.py",
            "--print-cmd",
            "--json",
            "--training-script", "train.py",
        ]):
            rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "cmd" in data
        assert "env" in data
        assert "topology" in data
        assert data["cmd"][0] == "torchrun"

    def test_verify_all_skipped(self, capsys):
        with patch.object(sys, "argv", [
            "run_nemo_distributed.py",
            "--verify",
            "--skip-ib",
            "--skip-nvlink",
            "--skip-nccl",
        ]):
            rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "SKIPPED" in captured.out

    def test_invalid_topology(self, capsys):
        with patch.object(sys, "argv", [
            "run_nemo_distributed.py",
            "--tp-size", "16",
            "--nproc-per-node", "8",
            "--print-cmd",
            "--training-script", "train.py",
        ]):
            rc = main()
        assert rc == 1

    def test_dry_run(self, capsys):
        with patch.object(sys, "argv", [
            "run_nemo_distributed.py",
            "--dry-run",
            "--training-script", "train.py",
        ]):
            rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "torchrun" in captured.out

    def test_no_training_script(self, capsys):
        with patch.object(sys, "argv", [
            "run_nemo_distributed.py",
            "--print-cmd",
        ]):
            rc = main()
        assert rc == 1

    def test_ib_hca_override(self, capsys):
        with patch.object(sys, "argv", [
            "run_nemo_distributed.py",
            "--print-cmd",
            "--training-script", "train.py",
            "--ib-hca", "mlx5_0:1,mlx5_1:1",
        ]):
            rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "mlx5_0:1,mlx5_1:1" in captured.out

    def test_nccl_debug_override(self, capsys):
        with patch.object(sys, "argv", [
            "run_nemo_distributed.py",
            "--print-cmd",
            "--training-script", "train.py",
            "--nccl-debug", "WARN",
        ]):
            rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "NCCL_DEBUG=WARN" in captured.out
