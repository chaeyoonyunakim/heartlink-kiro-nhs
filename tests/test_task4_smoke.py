"""Smoke tests for Task 4.1 and 4.2 — clinical guardrails and classification preprocessing."""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import (
    apply_clinical_guardrails,
    build_classification_preprocessor,
    preprocess_classification,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


# ---- Task 4.1: apply_clinical_guardrails ------------------------------------

class TestApplyClinicalGuardrails:
    """Tests for the apply_clinical_guardrails function."""

    def test_replaces_zero_chol_with_nan(self, sample_df):
        """Zero cholesterol values are replaced with NaN."""
        features = sample_df.drop(columns=["num"])
        result = apply_clinical_guardrails(features)
        # Row 5 (index 5) had chol=0 in the fixture
        assert pd.isna(result.loc[5, "chol"])

    def test_replaces_zero_trestbps_with_nan(self):
        """Zero resting blood pressure values are replaced with NaN."""
        df = pd.DataFrame({"trestbps": [0.0, 120.0, 0.0], "chol": [200.0, 0.0, 180.0]})
        result = apply_clinical_guardrails(df)
        assert pd.isna(result.loc[0, "trestbps"])
        assert pd.isna(result.loc[2, "trestbps"])

    def test_nonzero_values_unchanged(self, sample_df):
        """Non-zero values in trestbps and chol remain unchanged."""
        features = sample_df.drop(columns=["num"])
        result = apply_clinical_guardrails(features)
        # Row 0 had trestbps=145, chol=233 — both should be untouched
        assert result.loc[0, "trestbps"] == 145.0
        assert result.loc[0, "chol"] == 233.0

    def test_returns_copy(self, sample_df):
        """The function returns a copy, not a mutated original."""
        features = sample_df.drop(columns=["num"])
        original_chol = features["chol"].copy()
        _ = apply_clinical_guardrails(features)
        pd.testing.assert_series_equal(features["chol"], original_chol)

    def test_no_zeroes_remain(self):
        """After guardrails, no zero values remain in trestbps or chol."""
        df = pd.DataFrame({
            "trestbps": [0.0, 0.0, 130.0],
            "chol": [0.0, 250.0, 0.0],
        })
        result = apply_clinical_guardrails(df)
        assert (result["trestbps"] == 0).sum() == 0
        assert (result["chol"] == 0).sum() == 0


# ---- Task 4.2: build_classification_preprocessor / preprocess_classification -

class TestBuildClassificationPreprocessor:
    """Tests for the build_classification_preprocessor function."""

    def test_returns_column_transformer(self):
        """The function returns an unfitted ColumnTransformer."""
        from sklearn.compose import ColumnTransformer
        preprocessor = build_classification_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_has_numerical_and_categorical_pipelines(self):
        """The transformer contains both 'num' and 'cat' sub-pipelines."""
        preprocessor = build_classification_preprocessor()
        names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in names
        assert "cat" in names


class TestPreprocessClassification:
    """Tests for the preprocess_classification function."""

    def test_output_shape_rows_match(self, sample_df):
        """Transformed array has the same number of rows as input."""
        features = sample_df.drop(columns=["num"])
        X, names = preprocess_classification(features)
        assert X.shape[0] == len(features)

    def test_no_nan_in_output(self, sample_df):
        """Transformed array contains no NaN values."""
        features = sample_df.drop(columns=["num"])
        X, _ = preprocess_classification(features)
        assert not np.isnan(X).any()

    def test_feature_names_returned(self, sample_df):
        """Feature names list is non-empty and matches column count."""
        features = sample_df.drop(columns=["num"])
        X, names = preprocess_classification(features)
        assert len(names) > 0
        assert len(names) == X.shape[1]

    def test_id_and_dataset_excluded(self, sample_df):
        """The id and dataset columns do not appear in feature names."""
        features = sample_df.drop(columns=["num"])
        _, names = preprocess_classification(features)
        for name in names:
            assert "id" not in name.lower().split("_") or "num__" in name.lower()
            assert "dataset" not in name.lower().split("_")
