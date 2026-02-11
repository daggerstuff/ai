#!/usr/bin/env python3
"""
GGUF Export Script - Multiple Quantizations
"""

import os
import subprocess
import sys
from pathlib import Path


def check_llama_cpp():
    """Check if llama.cpp is available"""
    try:
        result = subprocess.run(["python", "-c", "import llama_cpp"],
                              check=False, capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except:
        pass

    subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python[server]"], check=False)
    return True

def export_to_gguf(model_path, output_dir):
    """Export model to GGUF format with multiple quantizations"""


    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Quantization configurations
    quant_configs = [
        ("Q8_0", "q8_0", "8-bit quantization (recommended)"),
        ("Q6_K", "q6_k", "6-bit quantization (good balance)"),
        ("Q4_K_M", "q4_k_m", "4-bit quantization (smaller size)"),
        ("Q2_K", "q2_k", "2-bit quantization (minimal size)")
    ]

    # Base GGUF conversion (FP16)
    base_gguf = output_path / "Wayfarer2-Pixelated-fp16.gguf"

    try:
        # Convert to GGUF using llama.cpp convert script
        convert_cmd = [
            "python", "-m", "llama_cpp.convert",
            "--model", str(model_path),
            "--output", str(base_gguf),
            "--vocab-type", "hf"
        ]

        result = subprocess.run(convert_cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            return False


    except Exception:
        return False

    # Create quantized versions
    for quant_name, quant_type, _description in quant_configs:
        output_file = output_path / f"Wayfarer2-Pixelated-{quant_name}.gguf"

        try:
            quant_cmd = [
                "python", "-m", "llama_cpp.quantize",
                str(base_gguf),
                str(output_file),
                quant_type
            ]

            result = subprocess.run(quant_cmd, check=False, capture_output=True, text=True)
            if result.returncode == 0:
                pass
            else:
                pass

        except Exception:
            pass

    return True

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./wayfarer-finetuned"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./gguf-exports"

    if not os.path.exists(model_path):
        return False

    # Check dependencies
    check_llama_cpp()

    # Export to GGUF
    success = export_to_gguf(model_path, output_dir)

    if success:
        pass
    else:
        pass

    return success

if __name__ == "__main__":
    main()
