"""
Shared utility functions for edge case detection, generation, and validation notebooks.
Imports and wraps reusable logic from edge_case_generator.py, extreme_edge_cases.py, and ultra_nightmare_categories.py.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../edge_case_pipeline_standalone')))

from edge_case_generator import EdgeCaseGenerator
from extreme_edge_cases import ExtremeEdgeCases
from ultra_nightmare_categories import UltraNightmareCategories
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Example: Unified loader for all edge case categories
def load_all_edge_case_categories():
    base = EdgeCaseGenerator().get_categories()
    extreme = ExtremeEdgeCases().get_categories()
    ultra = UltraNightmareCategories().get_categories()
    return base + extreme + ultra

# Add more shared utilities as needed for data loading, preprocessing, and reporting.

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] < lower) | (data[column] > upper)]

def detect_outliers_zscore(data, column, threshold=3):
    mean = data[column].mean()
    std = data[column].std()
    z_scores = (data[column] - mean) / std
    return data[np.abs(z_scores) > threshold]

def isolation_forest_anomaly(data, column, contamination=0.05):
    iso = IsolationForest(contamination=contamination)
    data['anomaly'] = iso.fit_predict(data[[column]])
    return data[data['anomaly'] == -1]

def validate_data(df):
    missing = df.isnull().sum()
    types = df.dtypes
    return missing, types
