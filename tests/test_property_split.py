"""
Property-based tests for train-test split proportions.

Feature: heart-disease-prediction, Property 6: Train-test split preserves data and respects proportions
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pipeline import split_data


@settings(max_examples=100)
@given(data=st.data())
def test_split_preserves_data_and_respects_proportions(data):
    """Feature: heart-disease-prediction, Property 6: Train-test split preserves data and respects proportions

    Validates: Requirements 5.1, 8.5

    For any dataset of size n >= 5, splitting with test_size=0.2 SHALL produce
    a training set and testing set whose sizes sum to n, with the testing set
    size equal to round(n * 0.2) within +/-1 due to rounding and stratification.
    """
    # Minimum n=10 ensures test set has at least 2 samples (ceil(10*0.2)=2),
    # which is required by sklearn's stratified split (each class needs ≥2 members).
    n = data.draw(st.integers(min_value=10, max_value=1000), label="n")
    n_features = data.draw(st.integers(min_value=1, max_value=5), label="n_features")

    # Generate random feature array
    X = data.draw(
        st.lists(
            st.lists(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=n_features,
                max_size=n_features,
            ),
            min_size=n,
            max_size=n,
        ),
        label="X",
    )
    X = np.array(X, dtype=np.float64)

    # Generate binary target ensuring both classes have enough members for stratification.
    # Each class needs at least 2 members in the test set, so each class needs
    # at least ceil(2 / 0.2) = 10 members overall to be safe. Use a conservative
    # minimum of 3 to ensure at least 1 per split side, with enough for stratification.
    min_per_class = max(3, int(np.ceil(n * 0.2)) + 1)
    n_ones = data.draw(
        st.integers(min_value=min_per_class, max_value=n - min_per_class),
        label="n_ones",
    )
    y_values = np.array([1] * n_ones + [0] * (n - n_ones))
    # Shuffle to avoid ordering bias
    rng = np.random.RandomState(data.draw(st.integers(min_value=0, max_value=2**31 - 1), label="seed"))
    rng.shuffle(y_values)
    y = pd.Series(y_values, name="Target_Variable")

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Verify sizes sum to n
    assert X_train.shape[0] + X_test.shape[0] == n, (
        f"Expected train + test = {n}, got {X_train.shape[0]} + {X_test.shape[0]}"
    )

    # Verify test size is within +/-1 of round(n * 0.2)
    expected_test_size = round(n * 0.2)
    actual_test_size = X_test.shape[0]
    assert abs(actual_test_size - expected_test_size) <= 1, (
        f"Expected test size ≈ {expected_test_size}, got {actual_test_size} (n={n})"
    )
