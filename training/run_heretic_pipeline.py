#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

# Default to the HF equivalents of the local ollama models detected
DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-3-4b-it",
    "ibm-granite/granite-3.1-8b-instruct",
]


def run_pipeline(models):
    # 1. Ensure llama.cpp exists
    if not os.path.exists("llama.cpp"):
        print("Cloning llama.cpp to handle GGUF conversion...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git"], check=True
        )
        print("Installing llama.cpp requirements...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                "llama.cpp/requirements.txt",
            ],
            check=True,
        )
    else:
        print("llama.cpp already exists.")

    for model in models:
        model_basename = model.split("/")[-1]
        output_dir = f"heretic_{model_basename}"
        gguf_path = f"{output_dir}.gguf"

        if not os.path.exists(output_dir):
            print(f"\n--- Running Heretic against {model} ---")
            print("This may take 45+ minutes depending on your GPU...")
            # Notice we pass --save to explicitly save the output weights locally
            subprocess.run(["heretic", model, "--save", output_dir], check=True)
        else:
            print(
                f"\n--- Output directory {output_dir} already exists. "
                "Skipping Heretic step. ---"
            )

        if not os.path.exists(gguf_path) and os.path.exists(output_dir):
            print(f"\n--- Converting {output_dir} to GGUF ---")
            subprocess.run(
                [
                    sys.executable,
                    "llama.cpp/convert_hf_to_gguf.py",
                    output_dir,
                    "--outfile",
                    gguf_path,
                    "--outtype",
                    "f16",
                ],
                check=True,
            )
        elif os.path.exists(gguf_path):
            print(
                f"\n--- GGUF file {gguf_path} already exists. Skipping conversion. ---"
            )

        print(f"\n--- Importing model {model_basename}-heretic into Ollama ---")
        # create modelfile
        modelfile_content = f"FROM /tmp/{gguf_path}\n"
        modelfile_path = f"Modelfile.{model_basename}"

        try:
            # We must map the GGUF file into the Docker container, or copy it.
            # Easiest way is to copy it directly into the container
            print(f"Copying {gguf_path} into Ollama container...")
            subprocess.run(
                ["docker", "cp", gguf_path, f"pixelated-ollama:/tmp/{gguf_path}"],
                check=True,
            )

            # Execute ollama create
            ollama_cmd = (
                f"cat > /tmp/{modelfile_path} && "
                f"ollama create {model_basename}-heretic -f /tmp/{modelfile_path}"
            )
            subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    "pixelated-ollama",
                    "sh",
                    "-c",
                    ollama_cmd,
                ],
                input=modelfile_content.encode(),
                check=True,
            )
            print(f"Successfully imported {model_basename}-heretic into Ollama!")

            # Cleanup container temp space
            subprocess.run(
                [
                    "docker",
                    "exec",
                    "pixelated-ollama",
                    "rm",
                    f"/tmp/{gguf_path}",
                    f"/tmp/{modelfile_path}",
                ],
                check=False,
            )

        except Exception as e:
            print(f"Warning: Could not import to Ollama via docker exec. Error: {e}")
            print(
                f"Please import manually by ensuring Ollama has access to {gguf_path}."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Heretic optimization and convert to Ollama GGUF."
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=DEFAULT_MODELS,
        help=(
            "HuggingFace model IDs to decensor. "
            "Defaults to Llama 3.2 3B, Gemma 3 4B, "
            "Granite 3.1 8B."
        ),
    )
    args = parser.parse_args()

    run_pipeline(args.models)
