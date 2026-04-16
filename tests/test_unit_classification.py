"""
Unit tests for the classification evaluation stage of the HeartLink pipeline.

Validates Requirements 7.4, 7.5, 7.6.
"""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.pipeline import evaluate_classifier


@pytest.fixture()
def trained_classifier_artifacts(tmp_path):
    """Train a simple RandomForestClassifier on synthetic data and return
    the model, test set, and output directory for evaluation tests."""
    rng = np.random.RandomState(42)

    # Synthetic binary classification dataset (100 samples, 5 features)
    n_samples = 100
    X = rng.randn(n_samples, 5)
    y = pd.Series((X[:, 0] + X[:, 1] > 0).astype(int), name="Target_Variable")

    # Simple train/test split
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test, str(tmp_path)


class TestClassificationEvaluationOutputs:
    """Verify that evaluate_classifier creates all expected output files."""

    @pytest.fixture(autouse=True)
    def _run_evaluation(self, trained_classifier_artifacts):
        model, X_test, y_test, output_dir = trained_classifier_artifacts
        self.output_dir = output_dir
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
        self.metrics = evaluate_classifier(model, X_test, y_test, output_dir, feature_names=feature_names)

    def test_confusion_matrix_created(self):
        """Requirement 7.4 — confusion matrix PNG is saved."""
        assert os.path.isfile(os.path.join(self.output_dir, "confusion_matrix.png"))

    def test_roc_curve_created(self):
        """Requirement 7.5 — ROC curve PNG is saved."""
        assert os.path.isfile(os.path.join(self.output_dir, "roc_curve.png"))

    def test_classification_report_created(self):
        """Requirement 7.6 — classification report text file is saved."""
        assert os.path.isfile(
            os.path.join(self.output_dir, "classification_report.txt")
        )

    def test_metrics_contains_expected_keys(self):
        """Metrics dict contains accuracy, precision, recall, f1, auc_roc."""
        expected_keys = {"accuracy", "precision", "recall", "f1", "auc_roc"}
        assert set(self.metrics.keys()) == expected_keys

    def test_metrics_values_are_valid_floats(self):
        """All metric values are floats in [0, 1]."""
        for key, value in self.metrics.items():
            assert isinstance(value, float), f"{key} is not a float"
            assert 0.0 <= value <= 1.0, f"{key}={value} is out of [0, 1]"

    def test_feature_importance_created(self):
        """Requirement 10.7 — feature importance PNG is saved."""
        assert os.path.isfile(os.path.join(self.output_dir, "feature_importance.png"))

    def test_classifier_performance_created(self):
        """Requirement 10.8 — classifier performance summary PNG is saved."""
        assert os.path.isfile(os.path.join(self.output_dir, "classifier_performance.png"))
