"""
Unit tests for the clustering analysis stage of the HeartLink pipeline.

Validates Requirements 9.11, 9.12, 9.13.
"""

import os

import pytest

from src.pipeline import create_target_variable, load_data, run_clustering_analysis


@pytest.fixture()
def clustering_outputs(tmp_path):
    """Load real data, create target variable, and run clustering analysis.

    Returns the output directory path after clustering has completed.
    """
    df = load_data()
    features_df, _ = create_target_variable(df)
    run_clustering_analysis(features_df, str(tmp_path))
    return str(tmp_path)


class TestClusteringAnalysisOutputs:
    """Verify that run_clustering_analysis creates expected output files."""

    @pytest.fixture(autouse=True)
    def _setup(self, clustering_outputs):
        self.output_dir = clustering_outputs

    def test_elbow_plot_created(self):
        """Requirement 9.11 — elbow plot PNG is saved."""
        assert os.path.isfile(os.path.join(self.output_dir, "clustering_elbow.png"))

    def test_scatter_plot_created(self):
        """Requirement 9.12 — cluster scatter plot PNG is saved."""
        assert os.path.isfile(os.path.join(self.output_dir, "clustering_scatter.png"))

    def test_elbow_plot_nonzero_size(self):
        """Requirement 9.11 — elbow plot file is not empty."""
        path = os.path.join(self.output_dir, "clustering_elbow.png")
        assert os.path.getsize(path) > 0

    def test_scatter_plot_nonzero_size(self):
        """Requirement 9.12 — scatter plot file is not empty."""
        path = os.path.join(self.output_dir, "clustering_scatter.png")
        assert os.path.getsize(path) > 0


def test_cluster_summary_printed(tmp_path, capsys):
    """Requirement 9.13 — cluster summary with mean feature values is printed."""
    df = load_data()
    features_df, _ = create_target_variable(df)
    run_clustering_analysis(features_df, str(tmp_path))

    captured = capsys.readouterr().out
    assert "[CLUSTERING] Mean feature values per cluster:" in captured
