#!/usr/bin/env python3
"""PIX-4348: Multi-node torchrun launcher + NCCL/IB verify.
Sprint 8 Lane B — 5pt. Checks NCCL (libnccl), IB (ibverbs, mlx5), then launches distributed job."""

# NOTE: torch 2.13.0+cpu CPU-only; NCCL for GPU multi-node only. IB drivers present. NCCL missing.
import os
import subprocess
import sys


def check_nccl():
    return (
        os.path.exists("/usr/lib/libnccl.so")
        or os.path.exists("/usr/lib/libnccl.so.2")
        or subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout.find("libnccl") != -1
    )


def check_ib():
    return os.path.exists("/sys/class/infiniband") and any(
        "ib_" in line for line in subprocess.run(["lsmod"], capture_output=True, text=True).stdout.splitlines()
    )


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <target_module_or_script> [torchrun_args...]")
        return 2
    target = sys.argv[1]
    torchrun_args = sys.argv[2:]
    nccl_ok = check_nccl()
    ib_ok = check_ib()
    print(f"[PIX-4348] NCCL: {'OK' if nccl_ok else 'MISSING'}")
    print(f"[PIX-4348] IB: {'OK' if ib_ok else 'NOT LOADED'}")
    if not nccl_ok or not ib_ok:
        return 1
    print("[PIX-4348] NCCL/IB verified — launching multi-node torchrun.")
    result = subprocess.run(["torchrun", target, *torchrun_args], check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
