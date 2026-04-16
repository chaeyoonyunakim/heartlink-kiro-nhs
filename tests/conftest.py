"""
Shared pytest fixtures for the HeartLink test suite.

Provides a sample DataFrame that mirrors the UCI Heart Disease schema,
suitable for unit and property-based tests across all pipeline stages.
"""

import numpy as np
import pandas as pd
import pytest


# The 16 columns expected in the raw dataset
UCI_COLUMNS = [
    "id", "age", "sex", "dataset", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Return a small, realistic DataFrame matching the UCI schema.

    Contains 10 rows with a mix of values covering:
    - Both sexes
    - Multiple chest pain types
    - Biologically impossible zero values (chol=0 in row 5)
    - Some NaN values in ``ca`` and ``thal`` (mimicking non-Cleveland sources)
    - ``num`` values spanning 0–4 for binarisation testing
    """
    data = {
        "id": list(range(1, 11)),
        "age": [63, 67, 67, 37, 41, 56, 62, 57, 63, 53],
        "sex": [
            "Male", "Male", "Male", "Male", "Female",
            "Male", "Female", "Female", "Male", "Male",
        ],
        "dataset": [
            "Cleveland", "Cleveland", "Hungary", "Cleveland", "Switzerland",
            "VA Long Beach", "Cleveland", "Hungary", "Cleveland", "Switzerland",
        ],
        "cp": [
            "typical angina", "asymptomatic", "asymptomatic", "non-anginal",
            "atypical angina", "asymptomatic", "non-anginal", "asymptomatic",
            "asymptomatic", "typical angina",
        ],
        "trestbps": [145.0, 160.0, 120.0, 130.0, 130.0, 120.0, 140.0, 120.0, 130.0, 140.0],
        "chol": [233.0, 286.0, 229.0, 250.0, 204.0, 0.0, 268.0, 354.0, 254.0, 203.0],
        "fbs": [
            "TRUE", "FALSE", "FALSE", "FALSE", "FALSE",
            "FALSE", "FALSE", "FALSE", "FALSE", "TRUE",
        ],
        "restecg": [
            "lv hypertrophy", "lv hypertrophy", "lv hypertrophy", "normal",
            "lv hypertrophy", "normal", "lv hypertrophy", "normal",
            "lv hypertrophy", "normal",
        ],
        "thalch": [150.0, 108.0, 129.0, 187.0, 172.0, 140.0, 160.0, 163.0, 147.0, 155.0],
        "exang": [
            "FALSE", "TRUE", "TRUE", "FALSE", "FALSE",
            "FALSE", "FALSE", "TRUE", "FALSE", "TRUE",
        ],
        "oldpeak": [2.3, 1.5, 2.6, 3.5, 1.4, 0.0, 3.6, 0.6, 1.4, 3.1],
        "slope": [
            "downsloping", "flat", "flat", "downsloping", "upsloping",
            "flat", np.nan, np.nan, "flat", "downsloping",
        ],
        "ca": [0.0, 3.0, 2.0, 0.0, 0.0, np.nan, np.nan, np.nan, 1.0, 0.0],
        "thal": [
            "fixed defect", "normal", "reversable defect", "normal",
            "normal", np.nan, np.nan, np.nan, "reversable defect", "normal",
        ],
        "num": [0, 2, 1, 0, 0, 3, 4, 1, 2, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def sample_target(sample_df: pd.DataFrame) -> pd.Series:
    """Return the expected binarised target for ``sample_df``.

    Mapping: num == 0 → 0, num ∈ {1,2,3,4} → 1.
    """
    return (sample_df["num"] > 0).astype(int).rename("Target_Variable")
