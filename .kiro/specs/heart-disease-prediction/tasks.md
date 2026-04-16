# Implementation Plan: HeartLink Heart Disease Prediction Pipeline

## Overview

This plan implements the HeartLink pipeline as a single Python script (`src/pipeline.py`) with supporting tests in `tests/`. Each task builds incrementally — from project scaffolding and data loading through to classification, regression, clustering, and final integration. Property-based tests (Hypothesis) and unit tests (pytest) are woven in as optional sub-tasks close to the code they validate.

## Tasks

- [x] 1. Set up project structure, dependencies, and global constants
  - Create `src/pipeline.py` with module docstring, imports (pandas, numpy, scikit-learn, matplotlib, seaborn), and global constants (`NUMERICAL_FEATURES`, `CATEGORICAL_FEATURES`, `DROP_COLUMNS`, `REGRESSION_FEATURES`, `REGRESSION_TARGET`, `CLUSTERING_EXCLUDE_COLUMNS`, `MISSING_THRESHOLD`, `PCA_VARIANCE_THRESHOLD`, `K_RANGE`)
  - Create `tests/__init__.py` and `tests/conftest.py` with shared fixtures (e.g., sample DataFrame matching the UCI schema)
  - Ensure `outputs/` directory creation logic is present
  - _Requirements: 10.1, 10.2, 10.3_

- [x] 2. Implement data loading and target variable creation
  - [x] 2.1 Implement `load_data(filepath)` function
    - Read CSV into DataFrame, validate exactly 16 columns (id, age, sex, dataset, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal, num), print shape and first 5 rows
    - Raise `FileNotFoundError` with expected path if file is missing; raise `ValueError` if columns do not match
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [x] 2.2 Implement `create_target_variable(df)` function
    - Map `num` == 0 → 0 and `num` ∈ {1,2,3,4} → 1 to create `Target_Variable`
    - Print class distribution, drop original `num` column, return (features_df, target_series)
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 2.3 Write property test for binarisation correctness (Property 1)
    - **Property 1: Binarisation correctness and completeness**
    - Use Hypothesis to generate DataFrames with `num` values in {0,1,2,3,4} and verify every `Target_Variable` value is in {0,1}, original 0 maps to 0, values 1–4 map to 1, and `num` is absent from the returned DataFrame
    - **Validates: Requirements 2.1, 2.3, 2.4**

  - [x] 2.4 Write unit tests for data loading
    - Test successful load with correct shape and columns
    - Test `FileNotFoundError` on missing file
    - Test `ValueError` on incorrect column schema
    - _Requirements: 1.1, 1.2, 1.4_

- [x] 3. Implement exploratory data analysis
  - [x] 3.1 Implement `run_eda(df, target, output_dir)` function
    - Print summary statistics for all numerical features and missing value counts per column
    - Save correlation heatmap, target distribution bar chart, distribution histograms per numerical feature, and boxplots per numerical feature categorised by sex and Target_Variable as PNGs to `outputs/`
    - Use `[EDA]` prefix for all print statements
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [x] 3.2 Write unit tests for EDA outputs
    - Verify all expected PNG files are created in the output directory
    - _Requirements: 3.3, 3.4, 3.5, 3.6_

- [x] 4. Implement classification preprocessing pipeline
  - [x] 4.1 Implement `apply_clinical_guardrails(df)` function
    - Replace zero values in `trestbps` and `chol` with NaN
    - Print count of replaced zeroes per column using `[GUARDRAIL]` prefix
    - _Requirements: 4.2, 4.3_

  - [x] 4.2 Implement `build_classification_preprocessor()` and `preprocess_classification(df)` functions
    - Drop `id` and `dataset` columns
    - Build `ColumnTransformer` with numerical pipeline (SimpleImputer median → StandardScaler) and categorical pipeline (SimpleImputer most_frequent → OneHotEncoder drop='first', handle_unknown='ignore')
    - Return transformed array and feature names
    - _Requirements: 4.1, 4.4, 4.5, 4.6, 4.7, 4.8_

  - [x] 4.3 Write property test for clinical guardrails (Property 2)
    - **Property 2: Clinical guardrails replace biologically impossible values**
    - Use Hypothesis to generate DataFrames with zero and non-zero values in `trestbps`/`chol`; verify all zeroes become NaN and non-zero values are unchanged
    - **Validates: Requirements 4.2, 8.2**

  - [x] 4.4 Write property test for imputation completeness (Property 3)
    - **Property 3: Imputation completeness**
    - Use Hypothesis to generate DataFrames with random NaN placement in numerical and categorical columns; verify no NaN values remain after imputation
    - **Validates: Requirements 4.4, 4.5, 9.4**

  - [x] 4.5 Write property test for one-hot encoding column count (Property 4)
    - **Property 4: One-hot encoding produces correct column count**
    - Use Hypothesis to generate categorical columns with varying cardinality; verify output column count equals sum of (unique categories − 1) per column
    - **Validates: Requirements 4.6, 9.5**

  - [x] 4.6 Write property test for standardisation (Property 5)
    - **Property 5: Standardisation produces zero mean and unit variance**
    - Use Hypothesis to generate numerical arrays with at least two distinct values; verify transformed mean ≈ 0 and std ≈ 1 (within 1e-7)
    - **Validates: Requirements 4.7, 9.5**

  - [x] 4.7 Write property test for column exclusion (Property 8)
    - **Property 8: Non-predictive and high-missingness columns are excluded**
    - Verify `id` and `dataset` are absent after classification preprocessing; verify `id`, `dataset`, `ca`, `thal`, `slope` are absent after clustering preprocessing
    - **Validates: Requirements 4.1, 9.1**

- [x] 5. Checkpoint — Verify data loading, EDA, and preprocessing
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement train-test split and classification model
  - [x] 6.1 Implement `split_data(X, y, test_size, random_state)` function
    - Perform stratified 80/20 split with `random_state=42`
    - Print training and testing set sizes
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 6.2 Implement `train_classifier(X_train, y_train, random_state)` function
    - Fit `RandomForestClassifier(random_state=42)` on training data
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 6.3 Implement `evaluate_classifier(model, X_test, y_test, output_dir)` function
    - Compute and print accuracy, precision, recall, F1-score, and AUC-ROC using `[CLASSIFICATION]` prefix
    - Save confusion matrix PNG, ROC curve PNG, and classification report text file to `outputs/`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 6.4 Write property test for split proportions (Property 6)
    - **Property 6: Train-test split preserves data and respects proportions**
    - Use Hypothesis to generate datasets of varying sizes (5–1000); verify training + testing sizes sum to n and testing size ≈ round(n × 0.2) within ±1
    - **Validates: Requirements 5.1, 8.5**

  - [x] 6.5 Write property test for stratified class proportions (Property 7)
    - **Property 7: Stratified split preserves class proportions**
    - Use Hypothesis to generate binary targets with varying class ratios; verify class 1 proportion in train and test sets is within 0.05 of the original proportion
    - **Validates: Requirements 5.3**

  - [x] 6.6 Write unit tests for classification evaluation
    - Verify confusion matrix, ROC curve, and classification report files are created
    - Verify metrics dictionary contains expected keys
    - _Requirements: 7.4, 7.5, 7.6_

- [x] 7. Implement regression analysis with 8 dataset variants
  - [x] 7.1 Implement `generate_regression_variants(df)` function
    - Generate 8 variants from the Cartesian product of hospital inclusion (all/Cleveland-only), imputation method (Unknown category + median / mean + mode), and outlier removal (with/without IQR-based filtering)
    - Replace biologically impossible values with NaN before each variant's imputation
    - Print variant label, sample count, and feature count using `[REGRESSION]` prefix
    - _Requirements: 8.1, 8.2, 8.3_

  - [x] 7.2 Implement `train_and_evaluate_regression(variants, output_dir)` function
    - For each variant: extract regression features (age, trestbps, chol) and target (thalch), drop NaN rows, split 80/20 (seed 42), train LinearRegression, compute MAE/RMSE/R²
    - Print comparison table, select best variant by highest R²
    - Save comparison CSV, predicted-vs-actual scatter plot, and residual plot for the best variant to `outputs/`
    - _Requirements: 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11, 8.12_

  - [x] 7.3 Write property test for best variant selection (Property 12)
    - **Property 12: Best regression variant has maximum R²**
    - Use Hypothesis to generate lists of 8 random R² values; verify the selected best variant has R² ≥ every other variant's R²
    - **Validates: Requirements 8.9**

  - [x] 7.4 Write unit tests for regression analysis
    - Verify exactly 8 variants are generated
    - Verify comparison CSV has 8 rows
    - Verify scatter and residual plot PNGs are saved
    - _Requirements: 8.1, 8.10, 8.11, 8.12_

- [x] 8. Checkpoint — Verify classification and regression
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement clustering analysis with stricter preprocessing and PCA
  - [x] 9.1 Implement `preprocess_clustering(df)` function
    - Exclude `ca`, `thal`, `slope` columns to prevent hospital bias
    - Remove rows with >30% missing values and print count of removed samples
    - Impute `chol` using sex-and-age-group medians
    - Impute remaining numerical features by column median and categorical features by column mode
    - One-hot encode categoricals, standardise all features
    - Apply PCA with `n_components=0.9`, print components retained and cumulative explained variance
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

  - [x] 9.2 Implement `run_clustering_analysis(df, output_dir)` function
    - Evaluate KMeans for k = 2 to 8 using silhouette score, select optimal k
    - Train KMeans with optimal k and `random_state=42`
    - Print silhouette score, patient count per cluster, and mean feature values per cluster using `[CLUSTERING]` prefix
    - Save elbow plot (inertia vs k) and 2D cluster scatter plot (first two PCA components) to `outputs/`
    - _Requirements: 9.8, 9.9, 9.10, 9.11, 9.12, 9.13_

  - [x] 9.3 Write property test for row missingness filter (Property 9)
    - **Property 9: Clustering row filter removes high-missingness rows**
    - Use Hypothesis to generate DataFrames with random NaN patterns; verify no remaining row has >30% missing values
    - **Validates: Requirements 9.2**

  - [x] 9.4 Write property test for cholesterol imputation (Property 10)
    - **Property 10: Cholesterol sex-and-age-group median imputation**
    - Use Hypothesis to generate DataFrames with known sex/age groups and missing `chol`; verify no NaN remains in `chol` and each imputed value equals the group median
    - **Validates: Requirements 9.3**

  - [x] 9.5 Write property test for PCA variance retention (Property 11)
    - **Property 11: PCA retains at least 90% of variance**
    - Use Hypothesis to generate random standardised matrices; verify cumulative explained variance ratio ≥ 0.90
    - **Validates: Requirements 9.6**

  - [x] 9.6 Write property test for optimal k selection (Property 13)
    - **Property 13: Optimal k has maximum silhouette score**
    - Use Hypothesis to generate lists of 7 random silhouette scores (k = 2..8); verify the selected k corresponds to the highest score
    - **Validates: Requirements 9.9**

  - [x] 9.7 Write unit tests for clustering analysis
    - Verify elbow and scatter plot PNGs are saved
    - Verify cluster summary is printed
    - _Requirements: 9.11, 9.12, 9.13_

- [x] 10. Wire together the main entry point and end-to-end integration
  - [x] 10.1 Implement `main()` function
    - Set global random seed (numpy, random module) to 42
    - Create `outputs/` directory if it does not exist
    - Orchestrate all stages in sequence: load_data → create_target_variable → run_eda → preprocess_classification → split_data → train_classifier → evaluate_classifier → run_regression_analysis → run_clustering_analysis
    - Print final summary listing all generated output files
    - Add `if __name__ == "__main__": main()` guard
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x] 10.2 Write property test for pipeline idempotence (Property 14)
    - **Property 14: Pipeline idempotence**
    - Run the full pipeline twice on the same input CSV with seed 42; verify identical classification metrics, regression metrics, and clustering results
    - **Validates: Requirements 5.2, 6.2, 10.5**

  - [x] 10.3 Write end-to-end smoke test
    - Run `python src/pipeline.py` and verify exit code 0 and all expected output files exist in `outputs/`
    - _Requirements: 10.1, 10.4_

- [x] 11. Final checkpoint — Full pipeline verification
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at natural breakpoints
- Property tests validate the 14 universal correctness properties from the design document
- Unit tests validate specific examples, edge cases, and file-output verification
- All code resides in `src/pipeline.py`; all tests reside in `tests/`
- British English is used throughout all print statements and documentation

## Visualisation Overhaul Tasks

- [x] 12. Apply global visualisation style to match reference imagery
  - [x] 12.1 Add global style constants and apply matplotlib rcParams
    - Add `PLOT_DPI`, `PRIMARY_COLOR`, `SECONDARY_COLOR`, `ACCENT_COLOR` constants
    - Apply white background, 100 DPI, and consistent font settings via `plt.rcParams` in `main()`
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x] 12.2 Restyle correlation heatmap
    - Change colormap from `coolwarm` to `YlOrRd` (warm sequential) to match reference `data/heatmap.png`
    - Resize to 8×6 inches at 100 DPI
    - _Requirements: 10.12_

  - [x] 12.3 Restyle ROC curve
    - Change line colour from blue to red (`ACCENT_COLOR`) to match reference `data/roc.png`
    - Use grey dashed diagonal reference line
    - _Requirements: 10.9_

  - [x] 12.4 Restyle clustering elbow plot
    - Change marker/line colour to `PRIMARY_COLOR` (teal/blue) to match reference `data/elbow.png`
    - Resize to 8×5 inches
    - _Requirements: 10.4_

  - [x] 12.5 Restyle clustering scatter plot
    - Use distinct high-contrast cluster colours (tab10 colormap) to match reference `data/pca.png`
    - Resize to 8×6 inches
    - _Requirements: 10.11_

  - [x] 12.6 Restyle regression scatter and residual plots
    - Use `PRIMARY_COLOR` for scatter markers, red for ideal/zero lines
    - Resize scatter to 7×7, residuals to 7×5
    - _Requirements: 10.4_

  - [x] 12.7 Restyle distribution histograms and target distribution
    - Use `PRIMARY_COLOR` for histogram fill
    - Use `[PRIMARY_COLOR, ACCENT_COLOR]` for target distribution bars
    - Resize to 8×5 inches
    - _Requirements: 10.4_

- [x] 13. Add new visualisation outputs
  - [x] 13.1 Add categorical feature distribution plots
    - Save bar charts for cp, ca, thal, slope, and dataset as `cat_distribution_{col}.png`
    - Use `PRIMARY_COLOR` bars, 8×5 inches
    - _Requirements: 10.5_

  - [x] 13.2 Add cholesterol by target distribution plot
    - Save overlapping histograms or boxplot of chol grouped by Target_Variable as `target_chol.png`
    - Use `[PRIMARY_COLOR, ACCENT_COLOR]` colours
    - _Requirements: 10.6_

  - [x] 13.3 Add feature importance bar chart
    - Extract top-10 feature importances from the trained RandomForestClassifier
    - Save horizontal bar chart as `feature_importance.png` using viridis gradient
    - _Requirements: 10.7_

  - [x] 13.4 Add classifier performance summary bar chart
    - Plot accuracy, precision, recall, F1, AUC-ROC as grouped bars in `classifier_performance.png`
    - Use `PRIMARY_COLOR` bars
    - _Requirements: 10.8_

  - [x] 13.5 Add R² comparison bar chart for regression variants
    - Plot R² scores for all 8 variants as a bar chart in `regression_r2_comparison.png`
    - Use viridis gradient, highlight best variant
    - _Requirements: 10.10_

- [x] 14. Update tests for new visualisation outputs
  - [x] 14.1 Update end-to-end smoke test expected file list
    - Add all new PNG files to the expected output list in `tests/test_e2e_smoke.py`
    - _Requirements: 11.1, 11.4_

  - [x] 14.2 Update EDA unit tests for new plots
    - Add assertions for categorical distribution PNGs and target_chol.png
    - _Requirements: 10.5, 10.6_

  - [x] 14.3 Add unit tests for new classification and regression plots
    - Verify feature_importance.png, classifier_performance.png, and regression_r2_comparison.png are created
    - _Requirements: 10.7, 10.8, 10.10_
