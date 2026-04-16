"""
Property-based tests for imputation completeness.

Feature: heart-disease-prediction, Property 3: Imputation completeness
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_classification_preprocessor,
)

# ---------------------------------------------------------------------------
# Realistic value strategies for each column
# ---------------------------------------------------------------------------

# Numerical feature value pools (non-NaN base values)
numerical_strategies = {
    "age": st.floats(min_value=20.0, max_value=90.0, allow_nan=False, allow_infinity=False),
    "trestbps": st.floats(min_value=80.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    "chol": st.floats(min_value=100.0, max_value=600.0, allow_nan=False, allow_infinity=False),
    "thalch": st.floats(min_value=60.0, max_value=220.0, allow_nan=False, allow_infinity=False),
    "oldpeak": st.floats(min_value=0.0, max_value=6.0, allow_nan=False, allow_infinity=False),
    "ca": st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
}

# Categorical feature value pools
categorical_strategies = {
    "sex": st.sampled_from(["Male", "Female"]),
    "cp": st.sampled_from(["typical angina", "atypical angina", "non-anginal", "asymptomatic"]),
    "fbs": st.sampled_from(["TRUE", "FALSE"]),
    "restecg": st.sampled_from(["normal", "st-t abnormality", "lv hypertrophy"]),
    "exang": st.sampled_from(["TRUE", "FALSE"]),
    "slope": st.sampled_from(["upsloping", "flat", "downsloping"]),
    "thal": st.sampled_from(["normal", "fixed defect", "reversable defect"]),
}


@settings(max_examples=100)
@given(data=st.data())
def test_imputation_completeness(data):
    """Feature: heart-disease-prediction, Property 3: Imputation completeness

    Validates: Requirements 4.4, 4.5, 9.4

    For any DataFrame with NaN values in numerical columns (age, trestbps, chol,
    thalch, oldpeak, ca) and categorical columns (sex, cp, fbs, restecg, exang,
    slope, thal), after applying the imputation step of the preprocessing pipeline,
    no NaN values SHALL remain in any of those columns.
    """
    n_rows = data.draw(st.integers(min_value=3, max_value=30), label="n_rows")

    # Build a DataFrame with realistic values, then randomly inject NaN.
    # We guarantee at least one non-NaN value per column so imputation can work.
    columns: dict[str, list] = {}

    for col in NUMERICAL_FEATURES:
        base_values = data.draw(
            st.lists(numerical_strategies[col], min_size=n_rows, max_size=n_rows),
            label=f"{col}_base",
        )
        # Decide which rows get NaN — leave at least one non-NaN
        nan_mask = data.draw(
            st.lists(st.booleans(), min_size=n_rows, max_size=n_rows),
            label=f"{col}_nan_mask",
        )
        # Ensure at least one value is NOT NaN
        if all(nan_mask):
            nan_mask[0] = False
        columns[col] = [
            np.nan if is_nan else val for val, is_nan in zip(base_values, nan_mask)
        ]

    for col in CATEGORICAL_FEATURES:
        base_values = data.draw(
            st.lists(categorical_strategies[col], min_size=n_rows, max_size=n_rows),
            label=f"{col}_base",
        )
        nan_mask = data.draw(
            st.lists(st.booleans(), min_size=n_rows, max_size=n_rows),
            label=f"{col}_nan_mask",
        )
        # Ensure at least one value is NOT NaN
        if all(nan_mask):
            nan_mask[0] = False
        columns[col] = [
            np.nan if is_nan else val for val, is_nan in zip(base_values, nan_mask)
        ]

    df = pd.DataFrame(columns)

    # Fit-transform using the classification preprocessor
    preprocessor = build_classification_preprocessor()
    transformed = preprocessor.fit_transform(df)

    # Verify: no NaN values remain in the transformed output
    assert not np.isnan(transformed).any(), (
        f"NaN values remain after imputation. "
        f"NaN positions: {np.argwhere(np.isnan(transformed)).tolist()}"
    )
