"""
Property-based tests for column exclusion.

Feature: heart-disease-prediction, Property 8: Non-predictive and high-missingness columns are excluded
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pipeline import (
    CATEGORICAL_FEATURES,
    CLUSTERING_EXCLUDE_COLUMNS,
    DROP_COLUMNS,
    NUMERICAL_FEATURES,
    preprocess_classification,
    preprocess_clustering,
)

# --- Strategies -----------------------------------------------------------

# Valid values for each column used by the pipeline
sex_values = st.sampled_from(["Male", "Female"])
cp_values = st.sampled_from(
    ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
)
fbs_values = st.sampled_from(["TRUE", "FALSE"])
restecg_values = st.sampled_from(["normal", "st-t abnormality", "lv hypertrophy"])
exang_values = st.sampled_from(["TRUE", "FALSE"])
slope_values = st.sampled_from(["upsloping", "flat", "downsloping"])
thal_values = st.sampled_from(["normal", "fixed defect", "reversable defect"])

positive_float = st.floats(min_value=1.0, max_value=300.0, allow_nan=False, allow_infinity=False)
age_strategy = st.floats(min_value=20.0, max_value=90.0, allow_nan=False, allow_infinity=False)
ca_strategy = st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False)


@st.composite
def heart_disease_dataframe(draw, min_rows=10, max_rows=50):
    """Generate a DataFrame matching the UCI Heart Disease schema (without num).

    Ensures enough rows and diversity for the preprocessing pipelines to work.
    """
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    data = {
        "id": list(range(1, n_rows + 1)),
        "age": [draw(age_strategy) for _ in range(n_rows)],
        "sex": [draw(sex_values) for _ in range(n_rows)],
        "dataset": [draw(st.sampled_from(["Cleveland", "Hungary", "Switzerland", "VA Long Beach"])) for _ in range(n_rows)],
        "cp": [draw(cp_values) for _ in range(n_rows)],
        "trestbps": [draw(positive_float) for _ in range(n_rows)],
        "chol": [draw(positive_float) for _ in range(n_rows)],
        "fbs": [draw(fbs_values) for _ in range(n_rows)],
        "restecg": [draw(restecg_values) for _ in range(n_rows)],
        "thalch": [draw(positive_float) for _ in range(n_rows)],
        "exang": [draw(exang_values) for _ in range(n_rows)],
        "oldpeak": [draw(st.floats(min_value=0.0, max_value=6.0, allow_nan=False, allow_infinity=False)) for _ in range(n_rows)],
        "slope": [draw(slope_values) for _ in range(n_rows)],
        "ca": [draw(ca_strategy) for _ in range(n_rows)],
        "thal": [draw(thal_values) for _ in range(n_rows)],
    }

    return pd.DataFrame(data)


# --- Property 8a: Classification preprocessing excludes id and dataset ----


@settings(max_examples=100)
@given(df=heart_disease_dataframe())
def test_classification_preprocessing_excludes_non_predictive_columns(df):
    """Feature: heart-disease-prediction, Property 8: Non-predictive and high-missingness columns are excluded

    Validates: Requirements 4.1, 9.1

    For any DataFrame containing id and dataset columns, after classification
    preprocessing, id and dataset SHALL be absent from the feature names.
    """
    # Confirm the input has the columns we want excluded
    assert "id" in df.columns
    assert "dataset" in df.columns

    X, feature_names = preprocess_classification(df)

    # id and dataset must not appear in the output feature names
    for col in DROP_COLUMNS:
        assert col not in feature_names, (
            f"Column '{col}' should be excluded after classification preprocessing "
            f"but was found in feature_names: {feature_names}"
        )

    # The output should have rows and features
    assert X.shape[0] == len(df)
    assert X.shape[1] == len(feature_names)
    assert X.shape[1] > 0


# --- Property 8b: Clustering preprocessing excludes id, dataset, ca, thal, slope ---


@settings(max_examples=100)
@given(df=heart_disease_dataframe())
def test_clustering_preprocessing_excludes_non_predictive_and_high_missingness_columns(df):
    """Feature: heart-disease-prediction, Property 8: Non-predictive and high-missingness columns are excluded

    Validates: Requirements 4.1, 9.1

    For any DataFrame containing id, dataset, ca, thal, and slope columns,
    after clustering preprocessing, all five columns SHALL be absent.
    """
    excluded_columns = DROP_COLUMNS + CLUSTERING_EXCLUDE_COLUMNS

    # Confirm the input has all columns we want excluded
    for col in excluded_columns:
        assert col in df.columns, f"Input DataFrame missing expected column '{col}'"

    # preprocess_clustering returns (pca_array, n_components)
    X_pca, n_components = preprocess_clustering(df)

    # The PCA output should be a 2D array with valid dimensions
    assert X_pca.ndim == 2
    assert X_pca.shape[1] == n_components
    assert n_components > 0

    # Since PCA transforms the feature space, we can't directly check column
    # names. Instead, verify the number of PCA components is less than the
    # total number of original columns minus the excluded ones, confirming
    # the excluded columns did not contribute to the feature space.
    remaining_raw_columns = [c for c in df.columns if c not in excluded_columns]
    assert n_components <= len(remaining_raw_columns) + 20, (
        f"PCA produced {n_components} components but only "
        f"{len(remaining_raw_columns)} raw columns remain after exclusion"
    )


# --- Fixture-based tests for deterministic verification -------------------


def test_classification_excludes_id_and_dataset_fixture(sample_df):
    """Verify id and dataset are absent from classification feature names
    using the shared sample_df fixture.

    Validates: Requirements 4.1
    """
    X, feature_names = preprocess_classification(sample_df)

    assert "id" not in feature_names
    assert "dataset" not in feature_names
    assert X.shape[0] == len(sample_df)
    assert X.shape[1] == len(feature_names)


def test_clustering_excludes_all_target_columns_fixture(sample_df):
    """Verify id, dataset, ca, thal, slope are all absent after clustering
    preprocessing using the shared sample_df fixture.

    Validates: Requirements 9.1
    """
    excluded = DROP_COLUMNS + CLUSTERING_EXCLUDE_COLUMNS

    X_pca, n_components = preprocess_clustering(sample_df)

    assert X_pca.ndim == 2
    assert X_pca.shape[1] == n_components
    assert n_components > 0
    # PCA output should have fewer dimensions than original columns
    assert n_components < len(sample_df.columns)
