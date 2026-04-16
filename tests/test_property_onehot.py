"""
Property-based tests for one-hot encoding column count.

Feature: heart-disease-prediction, Property 4: One-hot encoding produces correct column count
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.preprocessing import OneHotEncoder


# Strategy to generate a single categorical column with at least 2 unique values
def categorical_column_strategy(n_rows: int):
    """Generate a list of categorical string values with at least 2 unique values."""
    return st.lists(
        st.sampled_from(["cat_a", "cat_b", "cat_c", "cat_d", "cat_e"]),
        min_size=n_rows,
        max_size=n_rows,
    ).filter(lambda vals: len(set(vals)) >= 2)


@settings(max_examples=100)
@given(data=st.data())
def test_onehot_encoding_column_count(data):
    """Feature: heart-disease-prediction, Property 4: One-hot encoding produces correct column count

    Validates: Requirements 4.6, 9.5

    For any set of categorical columns, applying one-hot encoding with
    drop='first' SHALL produce a number of output columns equal to the sum
    of (unique non-null categories - 1) for each input categorical column.
    """
    n_rows = data.draw(st.integers(min_value=5, max_value=50), label="n_rows")
    n_cols = data.draw(st.integers(min_value=1, max_value=5), label="n_cols")

    # Build a DataFrame with n_cols categorical columns, each with at least
    # 2 unique values so that drop='first' is meaningful.
    col_data: dict[str, list[str]] = {}
    for i in range(n_cols):
        # Draw from a pool of category labels; vary the pool per column
        pool_size = data.draw(
            st.integers(min_value=2, max_value=6), label=f"pool_size_{i}"
        )
        pool = [f"level_{j}" for j in range(pool_size)]
        values = data.draw(
            st.lists(
                st.sampled_from(pool), min_size=n_rows, max_size=n_rows
            ).filter(lambda vals: len(set(vals)) >= 2),
            label=f"col_{i}_values",
        )
        col_data[f"cat_{i}"] = values

    df = pd.DataFrame(col_data)

    # Apply OneHotEncoder with the same settings as the pipeline
    encoder = OneHotEncoder(
        drop="first", handle_unknown="ignore", sparse_output=False
    )
    transformed = encoder.fit_transform(df)

    # Expected column count: sum of (unique_categories - 1) per column
    expected_cols = sum(df[col].nunique() - 1 for col in df.columns)

    assert transformed.shape[1] == expected_cols, (
        f"Expected {expected_cols} output columns but got {transformed.shape[1]}. "
        f"Per-column unique counts: "
        f"{[df[col].nunique() for col in df.columns]}"
    )
