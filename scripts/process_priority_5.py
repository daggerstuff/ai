#!/usr/bin/env python3
"""
Console script wrapper for Priority 5 processor.
"""

import sys
import importlib.util
import os

def main():
    """Main entry point for priority 5 processing."""
    # Import the actual processor module
    processor_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "dataset_pipeline", 
        "processors", 
        "phase-01", 
        "run_task_5_5_priority_5.py"
    )
    
    spec = importlib.util.spec_from_file_location("processor", processor_path)
    processor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(processor_module)
    
    # Run the processor
    processor_module.process_priority_5()

if __name__ == "__main__":
    main()
