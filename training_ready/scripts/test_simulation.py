#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from ai.dataset_pipeline.simulation.session_simulator import SessionSimulator


def main():
    print("Testing Session Simulator...")
    sim = SessionSimulator()
    batch = sim.generate_batch(count=3)
    print(f"Generated {len(batch)} sessions.")
    print("Sample Transcript Turn 1:")
    print(batch[0]["transcript"][1])  # Skip system msg


if __name__ == "__main__":
    main()
