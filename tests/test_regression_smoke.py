"""Smoke test for regression analysis functions (Tasks 7.1 and 7.2)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import (
    load_data,
    create_target_variable,
    generate_regression_variants,
    train_and_evaluate_regression,
)


def test_generate_regression_variants_produces_8_variants():
    """Verify that generate_regression_variants produces exactly 8 variants."""
    df = load_data()
    features_df, _ = create_target_variable(df)
    variants = generate_regression_variants(features_df)

    assert len(variants) == 8, f"Expected 8 variants, got {len(variants)}"

    # Check each variant has the required keys
    required_keys = {
        "label", "hospital", "imputation", "outlier_removal",
        "X", "y", "sample_count", "feature_count",
    }
    for v in variants:
        assert required_keys.issubset(v.keys()), f"Missing keys in variant: {required_keys - v.keys()}"
        assert v["sample_count"] > 0, f"Variant '{v['label']}' has 0 samples"
        assert v["feature_count"] == 3, f"Variant '{v['label']}' has {v['feature_count']} features, expected 3"


def test_train_and_evaluate_regression_outputs(tmp_path):
    """Verify regression evaluation produces CSV and plot outputs."""
    df = load_data()
    features_df, _ = create_target_variable(df)
    variants = generate_regression_variants(features_df)

    output_dir = str(tmp_path)
    comparison_df = train_and_evaluate_regression(variants, output_dir)

    # Comparison table has 8 rows
    assert len(comparison_df) == 8, f"Expected 8 rows, got {len(comparison_df)}"

    # Required columns present
    expected_cols = ["variant_label", "hospital", "imputation", "outlier_removal", "n_samples", "MAE", "RMSE", "R2"]
    for col in expected_cols:
        assert col in comparison_df.columns, f"Missing column: {col}"

    # Output files exist
    assert os.path.isfile(os.path.join(output_dir, "regression_variant_comparison.csv"))
    assert os.path.isfile(os.path.join(output_dir, "regression_scatter.png"))
    assert os.path.isfile(os.path.join(output_dir, "regression_residuals.png"))


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
