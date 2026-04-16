"""
Property-based tests for clinical guardrails.

Feature: heart-disease-prediction, Property 2: Clinical guardrails replace biologically impossible values
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pipeline import apply_clinical_guardrails


# Strategy for generating float values that include 0.0 alongside positive values.
# We use a mix of 0.0 and positive floats to ensure both zero and non-zero paths
# are exercised.
guardrail_values = st.one_of(
    st.just(0.0),
    st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
)


@settings(max_examples=100)
@given(data=st.data())
def test_clinical_guardrails_replace_biologically_impossible_values(data):
    """Feature: heart-disease-prediction, Property 2: Clinical guardrails replace biologically impossible values

    Validates: Requirements 4.2, 8.2

    For any DataFrame containing trestbps and chol columns with zero values,
    applying apply_clinical_guardrails SHALL replace every zero in those columns
    with NaN, whilst leaving all non-zero values unchanged.
    """
    n_rows = data.draw(st.integers(min_value=1, max_value=50), label="n_rows")
    trestbps_values = data.draw(
        st.lists(guardrail_values, min_size=n_rows, max_size=n_rows),
        label="trestbps_values",
    )
    chol_values = data.draw(
        st.lists(guardrail_values, min_size=n_rows, max_size=n_rows),
        label="chol_values",
    )

    # Include an extra column to verify it is unaffected
    other_values = data.draw(
        st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=n_rows,
            max_size=n_rows,
        ),
        label="other_values",
    )

    df = pd.DataFrame(
        {
            "trestbps": trestbps_values,
            "chol": chol_values,
            "age": other_values,
        }
    )

    result = apply_clinical_guardrails(df)

    # 1. All zeroes in trestbps become NaN
    for i, val in enumerate(trestbps_values):
        if val == 0.0:
            assert np.isnan(result["trestbps"].iloc[i]), (
                f"Row {i}: trestbps=0.0 should become NaN, got {result['trestbps'].iloc[i]}"
            )

    # 2. All zeroes in chol become NaN
    for i, val in enumerate(chol_values):
        if val == 0.0:
            assert np.isnan(result["chol"].iloc[i]), (
                f"Row {i}: chol=0.0 should become NaN, got {result['chol'].iloc[i]}"
            )

    # 3. All non-zero values in trestbps remain unchanged
    for i, val in enumerate(trestbps_values):
        if val != 0.0:
            assert result["trestbps"].iloc[i] == val, (
                f"Row {i}: trestbps={val} should be unchanged, got {result['trestbps'].iloc[i]}"
            )

    # 4. All non-zero values in chol remain unchanged
    for i, val in enumerate(chol_values):
        if val != 0.0:
            assert result["chol"].iloc[i] == val, (
                f"Row {i}: chol={val} should be unchanged, got {result['chol'].iloc[i]}"
            )

    # 5. Other columns are unaffected
    for i, val in enumerate(other_values):
        assert result["age"].iloc[i] == val, (
            f"Row {i}: age={val} should be unchanged, got {result['age'].iloc[i]}"
        )

    # 6. The original DataFrame is not mutated (function returns a copy)
    for i, val in enumerate(trestbps_values):
        assert df["trestbps"].iloc[i] == val, (
            "Original DataFrame was mutated by apply_clinical_guardrails"
        )
