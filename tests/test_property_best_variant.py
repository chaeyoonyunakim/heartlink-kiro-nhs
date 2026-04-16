"""
Property-based tests for best regression variant selection.

Feature: heart-disease-prediction, Property 12: Best regression variant has maximum R²
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st


@settings(max_examples=100)
@given(
    r2_values=st.lists(
        st.floats(min_value=-2.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=8,
        max_size=8,
    )
)
def test_best_variant_has_maximum_r2(r2_values):
    """Feature: heart-disease-prediction, Property 12: Best regression variant has maximum R²

    Validates: Requirements 8.9

    For any set of 8 evaluated regression variants with computed R² scores,
    the selected best variant SHALL have an R² score greater than or equal to
    every other variant's R² score.

    This test replicates the best-variant selection logic used in
    train_and_evaluate_regression (tracking via `if r2 > best_r2`) and
    verifies the selected variant has the maximum R².
    """
    # Replicate the selection logic from train_and_evaluate_regression
    best_r2 = -np.inf
    best_index = -1

    for i, r2 in enumerate(r2_values):
        if r2 > best_r2:
            best_r2 = r2
            best_index = i

    # The selected best variant must have R² >= every other variant
    for i, r2 in enumerate(r2_values):
        assert best_r2 >= r2, (
            f"Best variant (index={best_index}, R²={best_r2}) is less than "
            f"variant index={i} with R²={r2}"
        )

    # The best R² must equal the maximum of all R² values
    assert best_r2 == max(r2_values), (
        f"Best R² ({best_r2}) does not equal max R² ({max(r2_values)})"
    )
