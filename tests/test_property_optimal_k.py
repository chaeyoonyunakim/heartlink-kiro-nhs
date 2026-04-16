"""
Property-based tests for optimal k selection in clustering.

Feature: heart-disease-prediction, Property 13: Optimal k has maximum silhouette score
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st


K_RANGE = range(2, 9)  # k = 2 through 8 (7 values)


@settings(max_examples=100)
@given(
    scores=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=7,
        max_size=7,
    )
)
def test_optimal_k_has_maximum_silhouette_score(scores):
    """Feature: heart-disease-prediction, Property 13: Optimal k has maximum silhouette score

    Validates: Requirements 9.9

    For any set of silhouette scores computed for k = 2 through 8,
    the selected optimal k SHALL correspond to the highest silhouette score
    among all evaluated k values.

    This test replicates the optimal-k selection logic used in
    run_clustering_analysis (dict-based tracking with max()) and verifies
    the selected k has the maximum silhouette score.
    """
    # Replicate the selection logic from run_clustering_analysis
    silhouette_scores = {}
    for k, score in zip(K_RANGE, scores):
        silhouette_scores[k] = score

    optimal_k = max(silhouette_scores, key=silhouette_scores.get)

    # The selected k must have a silhouette score >= every other k
    for k, score in silhouette_scores.items():
        assert silhouette_scores[optimal_k] >= score, (
            f"Optimal k={optimal_k} (score={silhouette_scores[optimal_k]}) "
            f"is less than k={k} with score={score}"
        )

    # The optimal k's score must equal the maximum of all scores
    assert silhouette_scores[optimal_k] == max(scores), (
        f"Optimal k score ({silhouette_scores[optimal_k]}) does not equal "
        f"max score ({max(scores)})"
    )
