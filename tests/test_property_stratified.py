"""
Property-based tests for stratified split class proportions.

Feature: heart-disease-prediction, Property 7: Stratified split preserves class proportions
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pipeline import split_data


@settings(max_examples=100)
@given(data=st.data())
def test_stratified_split_preserves_class_proportions(data):
    """Feature: heart-disease-prediction, Property 7: Stratified split preserves class proportions

    Validates: Requirements 5.3

    For any binary target variable, after a stratified 80/20 split, the proportion
    of class 1 in the training set and the proportion of class 1 in the testing set
    SHALL each be within 0.05 of the proportion of class 1 in the original dataset.
    """
    # Use n >= 40 so the test set (≈8 samples) is large enough that
    # rounding a single sample doesn't exceed the 0.05 tolerance.
    n = data.draw(st.integers(min_value=40, max_value=1000), label="n")
    n_features = data.draw(st.integers(min_value=1, max_value=5), label="n_features")

    # Generate random feature array
    X = data.draw(
        st.lists(
            st.lists(
                st.floats(
                    min_value=-100.0,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n_features,
                max_size=n_features,
            ),
            min_size=n,
            max_size=n,
        ),
        label="X",
    )
    X = np.array(X, dtype=np.float64)

    # Generate binary target with varying class ratios.
    # Ensure both classes have at least 3 members for stratification,
    # and each class has enough members so rounding in the test set
    # doesn't push the proportion beyond the 0.05 tolerance.
    min_per_class = max(3, int(np.ceil(n * 0.1)))
    n_ones = data.draw(
        st.integers(min_value=min_per_class, max_value=n - min_per_class),
        label="n_ones",
    )
    y_values = np.array([1] * n_ones + [0] * (n - n_ones))

    # Shuffle to avoid ordering bias
    rng = np.random.RandomState(
        data.draw(st.integers(min_value=0, max_value=2**31 - 1), label="seed")
    )
    rng.shuffle(y_values)
    y = pd.Series(y_values, name="Target_Variable")

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Compute class 1 proportions
    original_prop = y.mean()
    train_prop = y_train.mean()
    test_prop = y_test.mean()

    # Verify train and test proportions are each within 0.05 of the original
    assert abs(train_prop - original_prop) <= 0.05, (
        f"Train class-1 proportion {train_prop:.4f} deviates from "
        f"original {original_prop:.4f} by more than 0.05"
    )
    assert abs(test_prop - original_prop) <= 0.05, (
        f"Test class-1 proportion {test_prop:.4f} deviates from "
        f"original {original_prop:.4f} by more than 0.05"
    )
