"""Quick verification that Task 1 artefacts are correctly set up."""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import (
    CATEGORICAL_FEATURES,
    CLUSTERING_EXCLUDE_COLUMNS,
    DATA_PATH,
    DROP_COLUMNS,
    EXPECTED_COLUMNS,
    K_RANGE,
    MISSING_THRESHOLD,
    NUMERICAL_FEATURES,
    OUTPUT_DIR,
    PCA_VARIANCE_THRESHOLD,
    RANDOM_SEED,
    REGRESSION_FEATURES,
    REGRESSION_TARGET,
    ensure_output_dir,
)


def test_numerical_features():
    assert NUMERICAL_FEATURES == ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]


def test_categorical_features():
    assert CATEGORICAL_FEATURES == ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


def test_drop_columns():
    assert DROP_COLUMNS == ["id", "dataset"]


def test_regression_features():
    assert REGRESSION_FEATURES == ["age", "trestbps", "chol"]


def test_regression_target():
    assert REGRESSION_TARGET == "thalch"


def test_clustering_exclude_columns():
    assert CLUSTERING_EXCLUDE_COLUMNS == ["ca", "thal", "slope"]


def test_missing_threshold():
    assert MISSING_THRESHOLD == 0.30


def test_pca_variance_threshold():
    assert PCA_VARIANCE_THRESHOLD == 0.90


def test_k_range():
    assert list(K_RANGE) == [2, 3, 4, 5, 6, 7, 8]


def test_random_seed():
    assert RANDOM_SEED == 42


def test_expected_columns_count():
    assert len(EXPECTED_COLUMNS) == 16


def test_ensure_output_dir_creates_directory(tmp_path):
    target = str(tmp_path / "test_outputs")
    ensure_output_dir(target)
    assert os.path.isdir(target)


def test_ensure_output_dir_idempotent(tmp_path):
    target = str(tmp_path / "test_outputs")
    ensure_output_dir(target)
    ensure_output_dir(target)  # Should not raise
    assert os.path.isdir(target)
