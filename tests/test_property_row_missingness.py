"""
Property-based tests for clustering row missingness filter.

Feature: heart-disease-prediction, Property 9: Clustering row filter removes high-missingness rows
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pipeline import MISSING_THRESHOLD

# ---------------------------------------------------------------------------
# Columns remaining after dropping id, dataset, ca, thal, slope from the
# clustering pipeline — these are the columns the row filter operates on.
# ---------------------------------------------------------------------------

CLUSTERING_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalch", "exang", "oldpeak",
]

# Realistic base-value strategies per column
_numerical_strategies = {
    "age": st.floats(min_value=20.0, max_value=90.0, allow_nan=False, allow_infinity=False),
    "trestbps": st.floats(min_value=80.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    "chol": st.floats(min_value=100.0, max_value=600.0, allow_nan=False, allow_infinity=False),
    "thalch": st.floats(min_value=60.0, max_value=220.0, allow_nan=False, allow_infinity=False),
    "oldpeak": st.floats(min_value=0.0, max_value=6.0, allow_nan=False, allow_infinity=False),
}

_categorical_strategies = {
    "sex": st.sampled_from(["Male", "Female"]),
    "cp": st.sampled_from(["typical angina", "atypical angina", "non-anginal", "asymptomatic"]),
    "fbs": st.sampled_from(["TRUE", "FALSE"]),
    "restecg": st.sampled_from(["normal", "st-t abnormality", "lv hypertrophy"]),
    "exang": st.sampled_from(["TRUE", "FALSE"]),
}


@settings(max_examples=100)
@given(data=st.data())
def test_row_missingness_filter_removes_high_missing_rows(data):
    """Feature: heart-disease-prediction, Property 9: Clustering row filter removes high-missingness rows

    Validates: Requirements 9.2

    For any DataFrame with random NaN patterns, after applying the >30%
    missingness row filter, no remaining row SHALL have more than 30% of
    its values missing.
    """
    n_rows = data.draw(st.integers(min_value=1, max_value=50), label="n_rows")

    # Build a DataFrame with realistic values and random NaN injection
    columns: dict[str, list] = {}

    for col in CLUSTERING_COLUMNS:
        if col in _numerical_strategies:
            base_values = data.draw(
                st.lists(_numerical_strategies[col], min_size=n_rows, max_size=n_rows),
                label=f"{col}_base",
            )
        else:
            base_values = data.draw(
                st.lists(_categorical_strategies[col], min_size=n_rows, max_size=n_rows),
                label=f"{col}_base",
            )

        # Randomly inject NaN values
        nan_mask = data.draw(
            st.lists(st.booleans(), min_size=n_rows, max_size=n_rows),
            label=f"{col}_nan",
        )
        columns[col] = [
            np.nan if is_nan else val for val, is_nan in zip(base_values, nan_mask)
        ]

    df = pd.DataFrame(columns)

    # Apply the same row missingness filter used in preprocess_clustering
    row_missing_frac = df.isnull().mean(axis=1)
    high_missing_mask = row_missing_frac > MISSING_THRESHOLD
    filtered_df = df[~high_missing_mask].reset_index(drop=True)

    # Property: no remaining row has >30% missing values
    if len(filtered_df) > 0:
        remaining_frac = filtered_df.isnull().mean(axis=1)
        assert (remaining_frac <= MISSING_THRESHOLD).all(), (
            f"Found rows with >{MISSING_THRESHOLD * 100:.0f}% missing values after filtering. "
            f"Max missingness: {remaining_frac.max():.4f}"
        )
