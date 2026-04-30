import json

NB_PATH = "/home/vivi/pixelated/ai/training/pixelated_colab_pilot.ipynb"
PROD_SCRIPT_PATH = "/home/vivi/pixelated/ai/training/pixelated_production_pilot.py"

def update_notebook():
    with open(PROD_SCRIPT_PATH) as f:
        prod_code = f.readlines()

    patched_code = []
    for line in prod_code:
        if "return parser.parse_args()" in line:
            patched_code.append(line.replace("return parser.parse_args()", "return parser.parse_args(args=[]) if 'ipykernel' in sys.modules else parser.parse_args()"))
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
                "metadata": {},
                "outputs": [],
                "source": patched_code
            }
        ],
        "metadata": {
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
        "nbformat_minor": 4
    }

    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=2)

if __name__ == "__main__":
    update_notebook()


