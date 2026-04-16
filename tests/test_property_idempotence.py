"""
Property-based test for pipeline idempotence.

Feature: heart-disease-prediction, Property 14: Pipeline idempotence

Validates: Requirements 5.2, 6.2, 10.5

This is a deterministic test (not a Hypothesis property test) that runs the
full pipeline stages twice with seed 42 and verifies identical results.
"""

import os
import random

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sklearn_silhouette_score

from src.pipeline import (
    DATA_PATH,
    K_RANGE,
    RANDOM_SEED,
    create_target_variable,
    evaluate_classifier,
    generate_regression_variants,
    load_data,
    preprocess_classification,
    preprocess_clustering,
    split_data,
    train_and_evaluate_regression,
    train_classifier,
)


def _run_pipeline(output_dir: str) -> dict:
    """Run the core pipeline stages and return collected results.

    Sets numpy and random seeds to 42 before execution, then runs:
    load_data → create_target_variable → preprocess_classification →
    split_data → train_classifier → evaluate_classifier →
    generate_regression_variants → train_and_evaluate_regression →
    clustering (preprocess_clustering + KMeans evaluation).

    Returns a dict with classification metrics, regression comparison
    DataFrame, and clustering results (silhouette scores, optimal k,
    cluster labels).
    """
    # Reset seeds
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Stage 1: Load data
    df = load_data(DATA_PATH)

    # Stage 2: Create target variable
    features_df, target = create_target_variable(df)

    # Stage 3: Preprocess for classification
    X, feature_names = preprocess_classification(features_df)

    # Stage 4: Split data
    X_train, X_test, y_train, y_test = split_data(X, target)

    # Stage 5: Train classifier
    model = train_classifier(X_train, y_train)

    # Stage 6: Evaluate classifier
    classification_metrics = evaluate_classifier(model, X_test, y_test, output_dir)

    # Stage 7: Generate regression variants and evaluate
    variants = generate_regression_variants(features_df)
    regression_df = train_and_evaluate_regression(variants, output_dir)

    # Stage 8: Clustering — run preprocess_clustering + KMeans manually
    X_pca, n_components = preprocess_clustering(features_df)

    silhouette_scores: dict[int, float] = {}
    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X_pca)
        sil = sklearn_silhouette_score(X_pca, labels)
        silhouette_scores[k] = sil

    optimal_k = max(silhouette_scores, key=silhouette_scores.get)  # type: ignore[arg-type]
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = final_kmeans.fit_predict(X_pca)

    return {
        "classification_metrics": classification_metrics,
        "regression_df": regression_df,
        "silhouette_scores": silhouette_scores,
        "optimal_k": optimal_k,
        "cluster_labels": cluster_labels,
        "n_pca_components": n_components,
    }


def test_pipeline_idempotence(tmp_path):
    """Feature: heart-disease-prediction, Property 14: Pipeline idempotence

    Validates: Requirements 5.2, 6.2, 10.5

    For any valid input CSV and fixed random seed of 42, executing the
    pipeline twice SHALL produce identical classification metrics (accuracy,
    precision, recall, F1, AUC-ROC), identical regression metrics (MAE,
    RMSE, R² per variant), and identical clustering results (silhouette
    score, cluster assignments).
    """
    # Create separate output directories for each run
    out_dir_1 = str(tmp_path / "run1")
    out_dir_2 = str(tmp_path / "run2")
    os.makedirs(out_dir_1, exist_ok=True)
    os.makedirs(out_dir_2, exist_ok=True)

    # Run the pipeline twice
    results_1 = _run_pipeline(out_dir_1)
    results_2 = _run_pipeline(out_dir_2)

    # --- Compare classification metrics -----------------------------------
    metrics_1 = results_1["classification_metrics"]
    metrics_2 = results_2["classification_metrics"]

    for key in ("accuracy", "precision", "recall", "f1", "auc_roc"):
        assert metrics_1[key] == metrics_2[key], (
            f"Classification metric '{key}' differs between runs: "
            f"{metrics_1[key]} vs {metrics_2[key]}"
        )

    # --- Compare regression comparison DataFrames -------------------------
    reg_df_1 = results_1["regression_df"].reset_index(drop=True)
    reg_df_2 = results_2["regression_df"].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        reg_df_1,
        reg_df_2,
        check_exact=True,
        obj="Regression comparison DataFrame",
    )

    # --- Compare clustering results ---------------------------------------
    # Silhouette scores must be identical
    sil_1 = results_1["silhouette_scores"]
    sil_2 = results_2["silhouette_scores"]
    for k in K_RANGE:
        assert sil_1[k] == sil_2[k], (
            f"Silhouette score for k={k} differs: {sil_1[k]} vs {sil_2[k]}"
        )

    # Optimal k must be the same
    assert results_1["optimal_k"] == results_2["optimal_k"], (
        f"Optimal k differs: {results_1['optimal_k']} vs {results_2['optimal_k']}"
    )

    # Cluster labels must be identical
    np.testing.assert_array_equal(
        results_1["cluster_labels"],
        results_2["cluster_labels"],
        err_msg="Cluster labels differ between runs",
    )

    # PCA components retained must match
    assert results_1["n_pca_components"] == results_2["n_pca_components"], (
        f"PCA components differ: {results_1['n_pca_components']} vs "
        f"{results_2['n_pca_components']}"
    )
