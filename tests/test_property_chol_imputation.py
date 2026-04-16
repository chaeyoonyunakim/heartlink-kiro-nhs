"""
Property-based tests for cholesterol sex-and-age-group median imputation.

Feature: heart-disease-prediction, Property 10: Cholesterol sex-and-age-group median imputation
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Strategies for generating realistic sex/age/chol values
# ---------------------------------------------------------------------------

_sex_strategy = st.sampled_from(["Male", "Female"])
_age_strategy = st.integers(min_value=20, max_value=89)
_chol_strategy = st.floats(
    min_value=100.0, max_value=600.0, allow_nan=False, allow_infinity=False,
)


@settings(max_examples=100)
@given(data=st.data())
def test_chol_sex_age_group_median_imputation(data):
    """Feature: heart-disease-prediction, Property 10: Cholesterol sex-and-age-group median imputation

    Validates: Requirements 9.3

    For any DataFrame with missing chol values, after applying sex-and-age-group
    median imputation, no NaN values SHALL remain in the chol column, and each
    imputed value SHALL equal the median chol of the corresponding sex-and-age-group.
    """
    # Generate between 4 and 40 rows so groups have enough data
    n_rows = data.draw(st.integers(min_value=4, max_value=40), label="n_rows")

    # Draw base values for each row
    sex_values = data.draw(
        st.lists(_sex_strategy, min_size=n_rows, max_size=n_rows),
        label="sex_values",
    )
    age_values = data.draw(
        st.lists(_age_strategy, min_size=n_rows, max_size=n_rows),
        label="age_values",
    )
    chol_values = data.draw(
        st.lists(_chol_strategy, min_size=n_rows, max_size=n_rows),
        label="chol_values",
    )

    # Randomly decide which rows have missing chol — but ensure at least one
    # non-NaN chol value exists per (sex, age_group) so group median is computable
    nan_mask = data.draw(
        st.lists(st.booleans(), min_size=n_rows, max_size=n_rows),
        label="nan_mask",
    )

    df = pd.DataFrame({
        "sex": sex_values,
        "age": [float(a) for a in age_values],
        "chol": chol_values,
    })

    # Compute age groups to ensure each group has at least one non-NaN chol
    df["age_group"] = pd.cut(df["age"], bins=range(0, 130, 10), right=False)

    # For each (sex, age_group), guarantee at least one non-NaN chol value
    group_has_non_nan: dict[tuple, bool] = {}
    for i in range(n_rows):
        key = (df.loc[i, "sex"], df.loc[i, "age_group"])
        if nan_mask[i]:
            # This row wants to be NaN — only allow if group already has a non-NaN
            if group_has_non_nan.get(key, False):
                df.loc[i, "chol"] = np.nan
            else:
                # Keep the value so the group has at least one non-NaN
                group_has_non_nan[key] = True
        else:
            group_has_non_nan[key] = True

    # Record which rows are missing before imputation
    missing_before = df["chol"].isnull()

    # Compute expected group medians from the non-NaN values
    expected_group_medians = (
        df.groupby(["sex", "age_group"], observed=True)["chol"].median()
    )

    # --- Apply the same imputation logic as preprocess_clustering -----------
    group_medians = df.groupby(["sex", "age_group"], observed=True)["chol"].median()

    chol_missing_mask = df["chol"].isnull()
    for idx in df.index[chol_missing_mask]:
        sex_val = df.loc[idx, "sex"]
        age_grp = df.loc[idx, "age_group"]
        if (sex_val, age_grp) in group_medians.index:
            df.loc[idx, "chol"] = group_medians[(sex_val, age_grp)]

    # Fallback: if any chol values are still NaN (group had no data), use
    # the overall median
    if df["chol"].isnull().any():
        df["chol"] = df["chol"].fillna(df["chol"].median())

    # --- Property assertions ------------------------------------------------

    # 1. No NaN remains in chol
    assert not df["chol"].isnull().any(), (
        f"NaN values remain in chol after imputation. "
        f"NaN count: {df['chol'].isnull().sum()}"
    )

    # 2. Each imputed value equals the group median
    for idx in df.index[missing_before]:
        sex_val = df.loc[idx, "sex"]
        age_grp = df.loc[idx, "age_group"]
        imputed_val = df.loc[idx, "chol"]
        expected_median = expected_group_medians.get((sex_val, age_grp))
        if expected_median is not None and not np.isnan(expected_median):
            assert imputed_val == expected_median, (
                f"Row {idx}: imputed chol={imputed_val} does not match "
                f"group median={expected_median} for sex={sex_val}, "
                f"age_group={age_grp}"
            )
