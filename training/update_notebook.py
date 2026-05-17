import json

NB_PATH = "/home/vivi/pixelated/ai/training/pixelated_colab_pilot.ipynb"
PROD_SCRIPT_PATH = "/home/vivi/pixelated/ai/training/pixelated_production_pilot.py"

COLAB_TELEMETRY_CODE = """
# =============================================================================
# COLAB TELEMETRY (requires pynvml)
# =============================================================================

import threading
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

class GPUStatsThread(threading.Thread):
    \"\"\"Daemon thread for monitoring GPU memory and compute utilization.\"\"\"

    def __init__(self, interval: int = 30):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self._nvml_ok = False
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self._nvml_ok = True
            except Exception as e:
                print(f"Failed to initialize NVML: {e}")

    def _log_gpu(self):
        if self._nvml_ok:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                print(
                    f"GPU: {info.used // 1024**2}MB/{info.total // 1024**2}MB mem, "
                    f"{util.gpu}% compute"
                )
            except Exception as e:
                pass

    def run(self):
        self._log_gpu()
        while not self.stop_event.wait(self.interval):
            self._log_gpu()

    def stop(self):
        self.stop_event.set()
        self.join(timeout=5)
        if self._nvml_ok:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
"""

def update_notebook():
    with open(PROD_SCRIPT_PATH) as f:
        prod_code = f.readlines()

    patched_code = []
    # Insert telemetry code after imports
    imports_finished = False
    for line in prod_code:
        if not imports_finished and line.startswith(("def ", "class ", "# ==")):
            patched_code.append(COLAB_TELEMETRY_CODE + "\n")
            imports_finished = True

        if "return parser.parse_args()" in line:
            replacement = (
                "return parser.parse_args(args=[]) if 'ipykernel' in sys.modules "
                "else parser.parse_args()"
            )
            patched_code.append(line.replace("return parser.parse_args()", replacement))
        elif 'if __name__ == "__main__":' in line:
            # Wrap main with telemetry thread and stop processing
            patched_code.append("gpu_thread = GPUStatsThread()\n")
            patched_code.append("gpu_thread.start()\n")
            patched_code.append("try:\n")
            patched_code.append("    main()\n")
            patched_code.append("finally:\n")
            patched_code.append("    gpu_thread.stop()\n")
            break
        else:
            patched_code.append(line)

    # Construct the notebook JSON
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🚀 Wayfarer-2-12B Distributed Production Pilot\n",
                    "**Status**: Production-Mirror Hardened\n",
                    "**Data**: 35.6B Token S3 Stream (PII Scrubbed & Clinically Aligned)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "install"},
                "outputs": [],
                "source": [
                    "# Install dependencies\n",
                    "%pip install -q transformers peft trl datasets bitsandbytes accelerate pynvml\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "drive-mount"},
                "outputs": [],
                "source": [
                    "# Mount Google Drive to persist checkpoints across Colab sessions\n",
                    "# from google.colab import drive\n",
                    "# drive.mount('/content/drive')\n",
                    "# os.environ['WORKSPACE_ROOT'] = '/content/drive/MyDrive/wayfarer_training'\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "train-main"},
                "outputs": [],
                "source": patched_code
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=2)

if __name__ == "__main__":
    update_notebook()


