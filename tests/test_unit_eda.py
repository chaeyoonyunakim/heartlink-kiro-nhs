"""
Unit tests for the EDA stage of the HeartLink pipeline.

Validates Requirements 3.3, 3.4, 3.5, 3.6.
"""

import os

import pytest

from src.pipeline import NUMERICAL_FEATURES, run_eda


NUMERICAL_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]


class TestEDAOutputFiles:
    """Verify that run_eda creates all expected PNG files."""

    @pytest.fixture(autouse=True)
    def _run_eda(self, sample_df, sample_target, tmp_path):
        """Run EDA once into a temporary directory for all tests in this class."""
        # Prepare features df (drop 'num' as the real pipeline does)
        features_df = sample_df.drop(columns=["num"])
        self.output_dir = str(tmp_path)
        run_eda(features_df, sample_target, self.output_dir)

    def test_correlation_heatmap_created(self):
        """Requirement 3.3 — correlation heatmap PNG is saved."""
        assert os.path.isfile(os.path.join(self.output_dir, "correlation_heatmap.png"))

    def test_target_distribution_created(self):
        """Requirement 3.4 — target distribution bar chart PNG is saved."""
        assert os.path.isfile(os.path.join(self.output_dir, "target_distribution.png"))

    def test_distribution_histograms_created(self):
        """Requirement 3.5 — a distribution histogram PNG exists for each numerical feature."""
        for col in NUMERICAL_COLS:
            path = os.path.join(self.output_dir, f"distribution_{col}.png")
            assert os.path.isfile(path), f"Missing distribution_{col}.png"

    def test_boxplots_created(self):
        """Requirement 3.6 — a boxplot PNG exists for each numerical feature."""
        for col in NUMERICAL_COLS:
            path = os.path.join(self.output_dir, f"boxplot_{col}.png")
            assert os.path.isfile(path), f"Missing boxplot_{col}.png"

    def test_no_extra_unexpected_files(self):
        """All files in the output directory are expected EDA outputs."""
        expected = {"correlation_heatmap.png", "target_distribution.png"}
        for col in NUMERICAL_COLS:
            expected.add(f"distribution_{col}.png")
            expected.add(f"boxplot_{col}.png")
        actual = set(os.listdir(self.output_dir))
        assert actual == expected
