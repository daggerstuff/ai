# Edge Case Pipeline Notebooks

This directory contains three Jupyter notebooks for edge case detection, generation, and validation, plus shared utilities and configuration files.

## Structure
- `edge_case_detection.ipynb`: Detects statistical outliers and anomalies.
- `edge_case_generation.ipynb`: Generates synthetic, boundary, adversarial, and counterfactual edge cases.
- `edge_case_validation.ipynb`: Validates model performance and bias detection on edge cases.
- `shared_utilities.py`: Reusable functions for all notebooks.
- `config.yaml`: Central configuration for paths and parameters.

## Usage
1. Update `config.yaml` with your paths and parameters.
2. Run each notebook in order:
   - Detection → Generation → Validation
3. Use `shared_utilities.py` for common functions.
4. Integrate with your bias detection pipeline via the API endpoint in config.

## Data Validation
Each notebook includes cells to check for missing values and data types before processing.

## Security
Do not hardcode secrets. Use environment variables for sensitive configs.

## Testing
Test cells/scripts are included to validate outputs and integration.

## Troubleshooting
- Ensure all paths in `config.yaml` are correct.
- Check for missing values before running analysis.
- Review documentation in each notebook for guidance.
