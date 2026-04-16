"""
Property-based tests for standardisation.

Feature: heart-disease-prediction, Property 5: Standardisation produces zero mean and unit variance
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.preprocessing import StandardScaler


@settings(max_examples=100)
@given(data=st.data())
def test_standardisation_zero_mean_unit_variance(data):
    """Feature: heart-disease-prediction, Property 5: Standardisation produces zero mean and unit variance

    Validates: Requirements 4.7, 9.5

    For any numerical column with at least two distinct values, after applying
    StandardScaler, the transformed column SHALL have a mean within 1e-7 of 0
    and a standard deviation within 1e-7 of 1.
    """
    n_rows = data.draw(st.integers(min_value=3, max_value=200), label="n_rows")
    n_cols = data.draw(st.integers(min_value=1, max_value=5), label="n_cols")

    # Generate a 2D array where each column has at least 2 distinct values
    columns = []
    for i in range(n_cols):
        col = data.draw(
            st.lists(
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=n_rows,
                max_size=n_rows,
            ).filter(lambda vals: len(set(vals)) >= 2),
            label=f"col_{i}",
        )
        columns.append(col)

    X = np.array(columns, dtype=np.float64).T  # shape (n_rows, n_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for col_idx in range(n_cols):
        col_mean = np.mean(X_scaled[:, col_idx])
        col_std = np.std(X_scaled[:, col_idx], ddof=0)

        assert abs(col_mean) < 1e-7, (
            f"Column {col_idx}: expected mean ≈ 0 but got {col_mean}"
        )
        assert abs(col_std - 1.0) < 1e-7, (
            f"Column {col_idx}: expected std ≈ 1 but got {col_std}"
        )
