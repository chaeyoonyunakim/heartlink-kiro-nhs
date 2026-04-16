"""
Property-based tests for target variable binarisation.

Feature: heart-disease-prediction, Property 1: Binarisation correctness and completeness
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames

from src.pipeline import create_target_variable


# Strategy: generate DataFrames with a `num` column containing values in {0,1,2,3,4}
# and at least one row. Additional columns are included to ensure `num` is the only
# one dropped.
num_dfs = data_frames(
    columns=[
        column("id", elements=st.integers(min_value=1, max_value=1000)),
        column("age", elements=st.integers(min_value=20, max_value=90)),
        column("num", elements=st.integers(min_value=0, max_value=4)),
    ],
    index=st.just(pd.RangeIndex(0)),  # placeholder, overridden by rows
    rows=st.tuples(
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=20, max_value=90),
        st.integers(min_value=0, max_value=4),
    ),
)


@settings(max_examples=100)
@given(
    data=st.data(),
)
def test_binarisation_correctness_and_completeness(data):
    """Feature: heart-disease-prediction, Property 1: Binarisation correctness and completeness

    Validates: Requirements 2.1, 2.3, 2.4

    For any DataFrame with a `num` column containing values in {0,1,2,3,4},
    applying create_target_variable SHALL produce a Target_Variable series where
    every value is in {0,1}, where original 0 maps to 0 and original 1-4 map to 1,
    and the returned features DataFrame SHALL NOT contain the `num` column.
    """
    n_rows = data.draw(st.integers(min_value=1, max_value=50), label="n_rows")
    num_values = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=4),
            min_size=n_rows,
            max_size=n_rows,
        ),
        label="num_values",
    )

    df = pd.DataFrame(
        {
            "id": list(range(1, n_rows + 1)),
            "age": [50] * n_rows,
            "num": num_values,
        }
    )

    features_df, target = create_target_variable(df)

    # 1. Every Target_Variable value is in {0, 1}
    assert set(target.unique()).issubset({0, 1}), (
        f"Target_Variable contains values outside {{0, 1}}: {set(target.unique())}"
    )

    # 2. Original num == 0 maps to 0
    for i, num_val in enumerate(num_values):
        if num_val == 0:
            assert target.iloc[i] == 0, (
                f"Row {i}: num=0 should map to Target_Variable=0, got {target.iloc[i]}"
            )

    # 3. Original num in {1, 2, 3, 4} maps to 1
    for i, num_val in enumerate(num_values):
        if num_val in {1, 2, 3, 4}:
            assert target.iloc[i] == 1, (
                f"Row {i}: num={num_val} should map to Target_Variable=1, got {target.iloc[i]}"
            )

    # 4. The `num` column is absent from the returned features DataFrame
    assert "num" not in features_df.columns, (
        "The returned features DataFrame still contains the 'num' column"
    )

    # 5. Target series is named correctly
    assert target.name == "Target_Variable", (
        f"Target series should be named 'Target_Variable', got '{target.name}'"
    )

    # 6. Row count is preserved
    assert len(features_df) == n_rows, (
        f"Features DataFrame should have {n_rows} rows, got {len(features_df)}"
    )
    assert len(target) == n_rows, (
        f"Target series should have {n_rows} elements, got {len(target)}"
    )
