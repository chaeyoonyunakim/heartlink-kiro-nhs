"""
Property-based tests for PCA variance retention.

Feature: heart-disease-prediction, Property 11: PCA retains at least 90% of variance
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.decomposition import PCA


@settings(max_examples=100)
@given(data=st.data())
def test_pca_retains_at_least_90_percent_variance(data):
    """Feature: heart-disease-prediction, Property 11: PCA retains at least 90% of variance

    Validates: Requirements 9.6

    For any standardised feature matrix with more than one feature, applying
    PCA(n_components=0.9) SHALL produce a transformed matrix whose cumulative
    explained variance ratio is >= 0.90.
    """
    n_cols = data.draw(st.integers(min_value=2, max_value=10), label="n_cols")
    n_rows = data.draw(
        st.integers(min_value=n_cols + 1, max_value=max(n_cols + 1, 200)),
        label="n_rows",
    )

    # Generate a random standardised matrix (each column has at least 2 distinct values)
    columns = []
    for i in range(n_cols):
        col = data.draw(
            st.lists(
                st.floats(
                    min_value=-1e3,
                    max_value=1e3,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n_rows,
                max_size=n_rows,
            ).filter(lambda vals: len(set(vals)) >= 2),
            label=f"col_{i}",
        )
        columns.append(col)

    X = np.array(columns, dtype=np.float64).T  # shape (n_rows, n_cols)

    # Standardise: zero mean, unit variance per column
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    # Guard against zero std (constant columns filtered above, but be safe)
    stds[stds == 0] = 1.0
    X_standardised = (X - means) / stds

    pca = PCA(n_components=0.9)
    pca.fit_transform(X_standardised)

    cumulative_variance = pca.explained_variance_ratio_.sum()

    assert cumulative_variance >= 0.90, (
        f"PCA cumulative explained variance ratio is {cumulative_variance:.4f}, "
        f"expected >= 0.90 (n_rows={n_rows}, n_cols={n_cols}, "
        f"n_components_retained={pca.n_components_})"
    )
