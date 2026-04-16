"""End-to-end smoke test for the HeartLink pipeline.

Runs ``python src/pipeline.py`` as a subprocess and verifies that:
1. The process exits with code 0.
2. All expected output files are present in ``outputs/``.

**Validates: Requirements 10.1, 10.4**
"""

import os
import subprocess
import sys

import pytest

# Every file the pipeline is expected to produce in outputs/
EXPECTED_OUTPUT_FILES = [
    # EDA
    "correlation_heatmap.png",
    "target_distribution.png",
    "distribution_age.png",
    "distribution_trestbps.png",
    "distribution_chol.png",
    "distribution_thalch.png",
    "distribution_oldpeak.png",
    "distribution_ca.png",
    "boxplot_age.png",
    "boxplot_trestbps.png",
    "boxplot_chol.png",
    "boxplot_thalch.png",
    "boxplot_oldpeak.png",
    "boxplot_ca.png",
    # Classification
    "confusion_matrix.png",
    "roc_curve.png",
    "classification_report.txt",
    # Regression
    "regression_variant_comparison.csv",
    "regression_scatter.png",
    "regression_residuals.png",
    # Clustering
    "clustering_elbow.png",
    "clustering_scatter.png",
]


@pytest.fixture(autouse=True)
def _clean_outputs():
    """Remove existing output files before the test so we know they were freshly generated."""
    for fname in EXPECTED_OUTPUT_FILES:
        path = os.path.join("outputs", fname)
        if os.path.exists(path):
            os.remove(path)
    yield


def test_pipeline_runs_successfully_and_produces_all_outputs():
    """Run the full pipeline and assert exit-code 0 + all output files exist."""
    result = subprocess.run(
        [sys.executable, "src/pipeline.py"],
        capture_output=True,
        text=True,
        timeout=300,
    )

    # 1. Exit code must be 0
    assert result.returncode == 0, (
        f"Pipeline exited with code {result.returncode}.\n"
        f"--- stdout (last 2000 chars) ---\n{result.stdout[-2000:]}\n"
        f"--- stderr (last 2000 chars) ---\n{result.stderr[-2000:]}"
    )

    # 2. Every expected file must exist in outputs/
    missing = [
        f for f in EXPECTED_OUTPUT_FILES if not os.path.isfile(os.path.join("outputs", f))
    ]
    assert not missing, f"Missing output files after pipeline run: {missing}"
