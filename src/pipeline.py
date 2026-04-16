"""
HeartLink — End-to-End Heart Disease Prediction Pipeline
=========================================================

A reproducible machine learning pipeline for heart disease analysis using the
UCI Heart Disease dataset. The pipeline delivers three complementary analyses:

1. **Binary classification** of heart disease presence (Random Forest)
2. **Regression** to predict maximum heart rate as a proxy for cardiac
   functional decline (Linear Regression across 8 dataset variants)
3. **Unsupervised clustering** to discover patient risk groups (KMeans with
   PCA dimensionality reduction)

Designed for NHS clinical contexts, HeartLink prioritises data integrity,
clinical guardrails, and full reproducibility over model complexity.

Usage::

    python src/pipeline.py

All outputs (plots, metrics, reports) are saved to the ``outputs/`` directory.
A global random seed of 42 ensures deterministic results across runs.
"""

import os
import random
import warnings

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Global Constants
# ---------------------------------------------------------------------------

#: Columns expected in the raw UCI Heart Disease CSV.
EXPECTED_COLUMNS: list[str] = [
    "id", "age", "sex", "dataset", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

#: Continuous / integer features used in classification preprocessing.
NUMERICAL_FEATURES: list[str] = [
    "age", "trestbps", "chol", "thalch", "oldpeak", "ca",
]

#: Discrete, non-numeric features used in classification preprocessing.
CATEGORICAL_FEATURES: list[str] = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "thal",
]

#: Non-predictive identifier columns to drop before modelling.
DROP_COLUMNS: list[str] = ["id", "dataset"]

#: Predictor features for the regression task (predicting maximum heart rate).
REGRESSION_FEATURES: list[str] = ["age", "trestbps", "chol"]

#: Target variable for the regression task.
REGRESSION_TARGET: str = "thalch"

#: High-missingness columns excluded from clustering to prevent hospital bias.
CLUSTERING_EXCLUDE_COLUMNS: list[str] = ["ca", "thal", "slope"]

#: Rows with more than this fraction of missing values are removed in clustering.
MISSING_THRESHOLD: float = 0.30

#: Minimum cumulative explained variance ratio retained by PCA in clustering.
PCA_VARIANCE_THRESHOLD: float = 0.90

#: Range of cluster counts evaluated during KMeans selection.
K_RANGE: range = range(2, 9)

#: Default path to the UCI Heart Disease CSV file.
DATA_PATH: str = os.path.join("data", "heart_disease_uci.csv")

#: Directory where all pipeline outputs (plots, reports, CSVs) are saved.
OUTPUT_DIR: str = "outputs"

#: Global random seed for reproducibility.
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Helper — ensure outputs directory exists
# ---------------------------------------------------------------------------

def ensure_output_dir(output_dir: str = OUTPUT_DIR) -> None:
    """Create the outputs directory if it does not already exist."""
    os.makedirs(output_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage 1 — Data Loading
# ---------------------------------------------------------------------------

def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """Load the UCI Heart Disease CSV and validate its schema.

    Reads the CSV file at *filepath* into a pandas DataFrame, checks that
    it contains exactly the 16 expected columns, and prints the shape and
    first five rows for quick visual verification.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (defaults to ``DATA_PATH``).

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame with all 16 columns intact.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the DataFrame does not contain exactly the expected columns.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Dataset not found at expected path: {filepath}"
        )

    df = pd.read_csv(filepath)

    # Validate column schema
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Column mismatch — expected {EXPECTED_COLUMNS}, "
            f"but got {list(df.columns)}"
        )

    print(f"Dataset shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())

    return df


# ---------------------------------------------------------------------------
# Stage 2 — Target Variable Creation
# ---------------------------------------------------------------------------

def create_target_variable(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Binarise the ``num`` column into a binary ``Target_Variable``.

    Maps ``num == 0`` to 0 (no heart disease) and ``num ∈ {1, 2, 3, 4}``
    to 1 (heart disease present). The original ``num`` column is dropped
    from the returned features DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a ``num`` column with integer values 0–4.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        A two-element tuple of (*features_df*, *target_series*) where
        *features_df* no longer contains the ``num`` column and
        *target_series* is named ``Target_Variable`` with values in {0, 1}.
    """
    target = (df["num"] > 0).astype(int).rename("Target_Variable")

    print("Class distribution (Target_Variable):")
    print(target.value_counts().sort_index())

    features_df = df.drop(columns=["num"])

    return features_df, target

# ---------------------------------------------------------------------------
# Stage 3 — Exploratory Data Analysis
# ---------------------------------------------------------------------------

def run_eda(
    df: pd.DataFrame, target: pd.Series, output_dir: str = OUTPUT_DIR
) -> None:
    """Run exploratory data analysis and save visualisations.

    Prints summary statistics for all numerical features and a count of
    missing values per column. Saves the following plots to *output_dir*:

    - Correlation heatmap of numerical features
    - Target variable distribution bar chart
    - Distribution histogram for each numerical feature
    - Boxplot for each numerical feature, categorised by sex and
      Target_Variable

    All console output is prefixed with ``[EDA]``.

    Parameters
    ----------
    df : pd.DataFrame
        Features DataFrame (``num`` column already removed).
    target : pd.Series
        Binary target series (``Target_Variable``).
    output_dir : str
        Directory where PNG files are saved (defaults to ``OUTPUT_DIR``).
    """
    ensure_output_dir(output_dir)

    # --- Summary statistics for numerical features --------------------------
    numerical_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    print("[EDA] Summary statistics for numerical features:")
    print(df[numerical_cols].describe())

    # --- Missing value counts -----------------------------------------------
    missing = df.isnull().sum()
    print("\n[EDA] Missing values per column:")
    print(missing)

    # --- Correlation heatmap ------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap — Numerical Features")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close(fig)

    # --- Target distribution bar chart --------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    target.value_counts().sort_index().plot(kind="bar", ax=ax, color=["#2196F3", "#F44336"])
    ax.set_xlabel("Target Variable")
    ax.set_ylabel("Count")
    ax.set_title("Target Variable Distribution")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Disease (0)", "Disease (1)"], rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "target_distribution.png"))
    plt.close(fig)

    # --- Distribution histograms per numerical feature ----------------------
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[col].dropna().hist(bins=30, ax=ax, edgecolor="black")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {col}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"distribution_{col}.png"))
        plt.close(fig)

    # --- Boxplots per numerical feature by sex and Target_Variable ----------
    # Temporarily attach target to the DataFrame for grouping
    plot_df = df.copy()
    plot_df["Target_Variable"] = target.values

    for col in numerical_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Boxplot by sex
        if "sex" in plot_df.columns:
            sns.boxplot(data=plot_df, x="sex", y=col, ax=axes[0])
            axes[0].set_title(f"{col} by Sex")
        else:
            axes[0].set_visible(False)

        # Boxplot by Target_Variable
        sns.boxplot(data=plot_df, x="Target_Variable", y=col, ax=axes[1])
        axes[1].set_title(f"{col} by Target Variable")

        fig.suptitle(f"Boxplots — {col}", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"boxplot_{col}.png"))
        plt.close(fig)

    print(f"\n[EDA] All EDA plots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Stage 4 — Classification Preprocessing
# ---------------------------------------------------------------------------

def apply_clinical_guardrails(df: pd.DataFrame) -> pd.DataFrame:
    """Replace biologically impossible zero values with NaN.

    Living patients cannot have a resting blood pressure or serum
    cholesterol of exactly zero. This function replaces such values with
    ``NaN`` so that downstream imputation treats them as missing data
    rather than valid measurements.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``trestbps`` and ``chol`` columns.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with zeroes in ``trestbps`` and
        ``chol`` replaced by ``NaN``.
    """
    df = df.copy()
    guardrail_columns = ["trestbps", "chol"]

    for col in guardrail_columns:
        zero_count = (df[col] == 0).sum()
        df[col] = df[col].replace(0, np.nan)
        print(f"[GUARDRAIL] Replaced {zero_count} zero value(s) in '{col}' with NaN")

    return df


def build_classification_preprocessor() -> ColumnTransformer:
    """Construct a ``ColumnTransformer`` for classification preprocessing.

    Builds two sub-pipelines:

    - **Numerical**: ``SimpleImputer(strategy='median')`` followed by
      ``StandardScaler`` — applied to ``NUMERICAL_FEATURES``.
    - **Categorical**: ``SimpleImputer(strategy='most_frequent')`` followed
      by ``OneHotEncoder(drop='first', handle_unknown='ignore')`` — applied
      to ``CATEGORICAL_FEATURES``.

    Returns
    -------
    ColumnTransformer
        An unfitted transformer ready to be fit on training data.
    """
    numerical_pipeline = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore",
                                  sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def preprocess_classification(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Preprocess features for the classification task.

    Drops non-predictive columns (``id``, ``dataset``), applies clinical
    guardrails, then fits and transforms the data using the classification
    ``ColumnTransformer``.

    Parameters
    ----------
    df : pd.DataFrame
        Features DataFrame (``num`` column already removed).

    Returns
    -------
    tuple[np.ndarray, list[str]]
        A two-element tuple of (*transformed_array*, *feature_names*) where
        *transformed_array* is the fully preprocessed NumPy array and
        *feature_names* lists the output column names.
    """
    # Drop non-predictive identifier columns
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # Apply clinical guardrails
    df = apply_clinical_guardrails(df)

    # Build and fit the preprocessor
    preprocessor = build_classification_preprocessor()
    X = preprocessor.fit_transform(df)

    # Extract feature names
    feature_names = preprocessor.get_feature_names_out().tolist()

    return X, feature_names


# ---------------------------------------------------------------------------
# Stage 5 — Train-Test Split
# ---------------------------------------------------------------------------

def split_data(
    X: np.ndarray,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Perform a stratified train-test split.

    Splits the preprocessed feature matrix and target series into training
    and testing subsets using stratified sampling to preserve class
    proportions.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed feature matrix.
    y : pd.Series
        Binary target series (``Target_Variable``).
    test_size : float
        Proportion of the dataset to include in the test split
        (defaults to 0.2).
    random_state : int
        Random seed for reproducibility (defaults to 42).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]
        ``(X_train, X_test, y_train, y_test)``
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[CLASSIFICATION] Training set size: {X_train.shape[0]} samples")
    print(f"[CLASSIFICATION] Testing set size:  {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Stage 6 — Classification Model Training
# ---------------------------------------------------------------------------

def train_classifier(
    X_train: np.ndarray,
    y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Fit a Random Forest classifier on the training data.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : pd.Series
        Training target series.
    random_state : int
        Random seed for the classifier (defaults to 42).

    Returns
    -------
    RandomForestClassifier
        The fitted classifier.
    """
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    print(f"[CLASSIFICATION] RandomForestClassifier trained on {X_train.shape[0]} samples "
          f"with {X_train.shape[1]} features")

    return model


# ---------------------------------------------------------------------------
# Stage 7 — Classification Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: pd.Series,
    output_dir: str = OUTPUT_DIR,
) -> dict[str, float]:
    """Evaluate the classifier and save diagnostic outputs.

    Computes accuracy, precision, recall, F1-score, and AUC-ROC on the
    test set. Saves a confusion matrix heatmap, ROC curve plot, and a
    full classification report to *output_dir*.

    Parameters
    ----------
    model : RandomForestClassifier
        Fitted classifier.
    X_test : np.ndarray
        Testing feature matrix.
    y_test : pd.Series
        Testing target series.
    output_dir : str
        Directory where output files are saved (defaults to ``OUTPUT_DIR``).

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``accuracy``, ``precision``, ``recall``,
        ``f1``, and ``auc_roc``.
    """
    ensure_output_dir(output_dir)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- Compute metrics ----------------------------------------------------
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba),
    }

    print("[CLASSIFICATION] Evaluation Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

    # --- Confusion matrix ---------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close(fig)

    # --- ROC curve ----------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['auc_roc']:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close(fig)

    # --- Classification report text file ------------------------------------
    report = classification_report(y_test, y_pred,
                                   target_names=["No Disease", "Disease"])
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"[CLASSIFICATION] Confusion matrix saved to {output_dir}/confusion_matrix.png")
    print(f"[CLASSIFICATION] ROC curve saved to {output_dir}/roc_curve.png")
    print(f"[CLASSIFICATION] Classification report saved to {report_path}")

    return metrics


# ---------------------------------------------------------------------------
# Stage 8 — Regression Analysis (8 Dataset Variants)
# ---------------------------------------------------------------------------

def generate_regression_variants(df: pd.DataFrame) -> list[dict]:
    """Generate 8 dataset variants for regression model comparison.

    Produces the Cartesian product of three binary preprocessing dimensions:

    - **Hospital inclusion**: all four hospitals *or* Cleveland-only
    - **Imputation method**: "Unknown" category for missing categoricals +
      median for numericals *or* mean for numericals + mode for categoricals
    - **Outlier removal**: retain all data *or* remove outliers using
      IQR-based filtering (1.5× IQR beyond Q1/Q3) on numerical features

    Before each variant's imputation, biologically impossible zero values in
    ``trestbps`` and ``chol`` are replaced with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Features DataFrame (``num`` column already removed, but ``dataset``
        column still present for hospital filtering).

    Returns
    -------
    list[dict]
        A list of 8 variant dictionaries, each containing:
        ``label``, ``hospital``, ``imputation``, ``outlier_removal``,
        ``X``, ``y``, ``sample_count``, ``feature_count``.
    """
    hospitals = ["all", "cleveland"]
    imputation_methods = ["unknown", "mean_mode"]
    outlier_options = [False, True]

    regression_cols = REGRESSION_FEATURES + [REGRESSION_TARGET]
    variants: list[dict] = []

    for hospital in hospitals:
        for imputation in imputation_methods:
            for outlier_removal in outlier_options:
                # --- Build descriptive label --------------------------------
                hosp_label = "All" if hospital == "all" else "Cleveland"
                imp_label = "Unknown" if imputation == "unknown" else "MeanMode"
                out_label = "NoOutliers" if outlier_removal else "WithOutliers"
                label = f"{hosp_label}-{imp_label}-{out_label}"

                # --- Filter by hospital -------------------------------------
                if hospital == "cleveland":
                    variant_df = df[df["dataset"] == "Cleveland"].copy()
                else:
                    variant_df = df.copy()

                # --- Select regression columns ------------------------------
                variant_df = variant_df[regression_cols].copy()

                # --- Clinical guardrails: replace impossible zeroes ---------
                for col in ["trestbps", "chol"]:
                    if col in variant_df.columns:
                        variant_df[col] = variant_df[col].replace(0, np.nan)

                # --- Apply imputation method --------------------------------
                if imputation == "unknown":
                    # Median for numerical columns
                    for col in regression_cols:
                        if variant_df[col].dtype in [np.float64, np.int64, float, int]:
                            median_val = variant_df[col].median()
                            variant_df[col] = variant_df[col].fillna(median_val)
                else:
                    # mean_mode: mean for numerical columns
                    for col in regression_cols:
                        if variant_df[col].dtype in [np.float64, np.int64, float, int]:
                            mean_val = variant_df[col].mean()
                            variant_df[col] = variant_df[col].fillna(mean_val)

                # --- Apply IQR-based outlier removal ------------------------
                if outlier_removal:
                    for col in regression_cols:
                        q1 = variant_df[col].quantile(0.25)
                        q3 = variant_df[col].quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        variant_df = variant_df[
                            (variant_df[col] >= lower) & (variant_df[col] <= upper)
                        ]

                # --- Drop any remaining NaN rows ----------------------------
                variant_df = variant_df.dropna(subset=regression_cols)

                X = variant_df[REGRESSION_FEATURES].values
                y = variant_df[REGRESSION_TARGET].values
                sample_count = X.shape[0]
                feature_count = X.shape[1]

                variant = {
                    "label": label,
                    "hospital": hospital,
                    "imputation": imputation,
                    "outlier_removal": outlier_removal,
                    "X": X,
                    "y": y,
                    "sample_count": sample_count,
                    "feature_count": feature_count,
                }
                variants.append(variant)

                print(
                    f"[REGRESSION] Variant '{label}': "
                    f"{sample_count} samples, {feature_count} features"
                )

    return variants


def train_and_evaluate_regression(
    variants: list[dict], output_dir: str = OUTPUT_DIR
) -> pd.DataFrame:
    """Train and evaluate a Linear Regression model on each dataset variant.

    For each variant the function:

    1. Splits into 80 % training / 20 % testing (seed 42).
    2. Fits a ``LinearRegression`` model on the training subset.
    3. Computes MAE, RMSE, and R² on the testing subset.

    After evaluating all variants it prints a comparison table, selects the
    best variant by highest R², and saves:

    - ``regression_variant_comparison.csv`` — full comparison table
    - ``regression_scatter.png`` — predicted-vs-actual scatter for the best
      variant
    - ``regression_residuals.png`` — residual plot for the best variant

    Parameters
    ----------
    variants : list[dict]
        List of variant dictionaries produced by
        :func:`generate_regression_variants`.
    output_dir : str
        Directory where output files are saved (defaults to ``OUTPUT_DIR``).

    Returns
    -------
    pd.DataFrame
        Comparison DataFrame with columns ``variant_label``, ``hospital``,
        ``imputation``, ``outlier_removal``, ``n_samples``, ``MAE``,
        ``RMSE``, ``R2``.
    """
    ensure_output_dir(output_dir)

    results: list[dict] = []
    best_r2 = -np.inf
    best_variant_label: str = ""
    best_y_test: np.ndarray | None = None
    best_y_pred: np.ndarray | None = None

    for variant in variants:
        label = variant["label"]
        X = variant["X"]
        y = variant["y"]

        if len(X) < 5:
            warnings.warn(
                f"[REGRESSION] Variant '{label}' has fewer than 5 samples — skipping."
            )
            continue

        # --- 80/20 split (seed 42) ------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

        # --- Train Linear Regression ----------------------------------------
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Compute metrics ------------------------------------------------
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = model.score(X_test, y_test)

        results.append({
            "variant_label": label,
            "hospital": variant["hospital"],
            "imputation": variant["imputation"],
            "outlier_removal": variant["outlier_removal"],
            "n_samples": variant["sample_count"],
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        })

        # Track best variant by R²
        if r2 > best_r2:
            best_r2 = r2
            best_variant_label = label
            best_y_test = y_test
            best_y_pred = y_pred

    # --- Build comparison DataFrame -----------------------------------------
    comparison_df = pd.DataFrame(results)

    # --- Print comparison table ---------------------------------------------
    print("\n[REGRESSION] Variant Comparison Table:")
    print(comparison_df.to_string(index=False))
    print(f"\n[REGRESSION] Best variant: '{best_variant_label}' (R² = {best_r2:.4f})")

    # --- Save comparison CSV ------------------------------------------------
    csv_path = os.path.join(output_dir, "regression_variant_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"[REGRESSION] Comparison table saved to {csv_path}")

    # --- Predicted-vs-actual scatter plot (best variant) --------------------
    if best_y_test is not None and best_y_pred is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(best_y_test, best_y_pred, alpha=0.6, edgecolors="k", linewidths=0.5)
        min_val = min(best_y_test.min(), best_y_pred.min())
        max_val = max(best_y_test.max(), best_y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
        ax.set_xlabel("Actual thalch")
        ax.set_ylabel("Predicted thalch")
        ax.set_title(f"Predicted vs Actual — {best_variant_label}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "regression_scatter.png"))
        plt.close(fig)
        print(f"[REGRESSION] Scatter plot saved to {output_dir}/regression_scatter.png")

        # --- Residual plot (best variant) -----------------------------------
        residuals = best_y_test - best_y_pred
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(best_y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.5)
        ax.axhline(y=0, color="r", linestyle="--")
        ax.set_xlabel("Predicted thalch")
        ax.set_ylabel("Residuals")
        ax.set_title(f"Residual Plot — {best_variant_label}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "regression_residuals.png"))
        plt.close(fig)
        print(f"[REGRESSION] Residual plot saved to {output_dir}/regression_residuals.png")

    return comparison_df

# ---------------------------------------------------------------------------
# Stage 9 — Clustering Analysis
# ---------------------------------------------------------------------------

def preprocess_clustering(df: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Preprocess features for unsupervised clustering with PCA reduction.

    Applies a stricter cleaning pipeline than classification preprocessing:

    1. Drops non-predictive columns (``id``, ``dataset``) and high-missingness
       columns (``ca``, ``thal``, ``slope``) to prevent hospital bias.
    2. Removes rows with more than 30 % missing values.
    3. Replaces biologically impossible zero values in ``trestbps`` and
       ``chol`` with NaN.
    4. Imputes ``chol`` using sex-and-age-group medians.
    5. Imputes remaining numerical features by column median and categorical
       features by column mode.
    6. One-hot encodes categorical features and standardises all features.
    7. Applies PCA retaining at least 90 % of cumulative explained variance.

    Parameters
    ----------
    df : pd.DataFrame
        Raw features DataFrame (``num`` column already removed but ``id``,
        ``dataset``, and other columns still present).

    Returns
    -------
    tuple[np.ndarray, int]
        A two-element tuple of (*pca_array*, *n_components*) where
        *pca_array* is the PCA-transformed feature matrix and
        *n_components* is the number of principal components retained.
    """
    df = df.copy()

    # --- Drop non-predictive and high-missingness columns -------------------
    cols_to_exclude = DROP_COLUMNS + CLUSTERING_EXCLUDE_COLUMNS
    df = df.drop(columns=[c for c in cols_to_exclude if c in df.columns])

    # --- Remove rows with >30% missing values ------------------------------
    row_missing_frac = df.isnull().mean(axis=1)
    high_missing_mask = row_missing_frac > MISSING_THRESHOLD
    n_removed = high_missing_mask.sum()
    df = df[~high_missing_mask].reset_index(drop=True)
    print(f"[CLUSTERING] Removed {n_removed} sample(s) with >{MISSING_THRESHOLD * 100:.0f}% missing values")

    # --- Replace biologically impossible zero values with NaN ---------------
    for col in ["trestbps", "chol"]:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            df[col] = df[col].replace(0, np.nan)
            if zero_count > 0:
                print(f"[CLUSTERING] Replaced {zero_count} zero value(s) in '{col}' with NaN")

    # --- Impute chol using sex-and-age-group medians ------------------------
    df["age_group"] = pd.cut(df["age"], bins=range(0, 130, 10), right=False)
    group_medians = df.groupby(["sex", "age_group"], observed=True)["chol"].median()

    chol_missing_mask = df["chol"].isnull()
    for idx in df.index[chol_missing_mask]:
        sex_val = df.loc[idx, "sex"]
        age_grp = df.loc[idx, "age_group"]
        if (sex_val, age_grp) in group_medians.index:
            df.loc[idx, "chol"] = group_medians[(sex_val, age_grp)]

    # Fallback: if any chol values are still NaN (group had no data), use
    # the overall median
    if df["chol"].isnull().any():
        df["chol"] = df["chol"].fillna(df["chol"].median())

    df = df.drop(columns=["age_group"])

    # --- Identify numerical and categorical columns for clustering ----------
    clustering_numerical = [
        c for c in NUMERICAL_FEATURES
        if c in df.columns and c not in CLUSTERING_EXCLUDE_COLUMNS
    ]
    clustering_categorical = [
        c for c in CATEGORICAL_FEATURES
        if c in df.columns and c not in CLUSTERING_EXCLUDE_COLUMNS
    ]

    # --- Impute remaining numerical features by column median ---------------
    for col in clustering_numerical:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # --- Impute remaining categorical features by column mode ---------------
    for col in clustering_categorical:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
                df[col] = df[col].infer_objects(copy=False)

    # --- One-hot encode categorical features --------------------------------
    df = pd.get_dummies(df, columns=clustering_categorical, drop_first=True)

    # --- Standardise all features -------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # --- Apply PCA with n_components=0.9 ------------------------------------
    pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    n_components = pca.n_components_
    cumulative_variance = pca.explained_variance_ratio_.cumsum()[-1]

    print(f"[CLUSTERING] PCA components retained: {n_components}")
    print(f"[CLUSTERING] Cumulative explained variance: {cumulative_variance:.4f}")

    return X_pca, n_components


def run_clustering_analysis(
    df: pd.DataFrame, output_dir: str = OUTPUT_DIR
) -> None:
    """Run unsupervised clustering analysis with KMeans and PCA.

    Preprocesses the data using :func:`preprocess_clustering`, evaluates
    KMeans for k = 2 to 8 using silhouette scores, selects the optimal k,
    and trains the final model. Saves an elbow plot and a 2D cluster scatter
    plot to *output_dir*.

    Parameters
    ----------
    df : pd.DataFrame
        Raw features DataFrame (``num`` column already removed).
    output_dir : str
        Directory where output files are saved (defaults to ``OUTPUT_DIR``).
    """
    ensure_output_dir(output_dir)

    # --- Preprocess for clustering ------------------------------------------
    X_pca, n_components = preprocess_clustering(df)

    # --- Evaluate KMeans for k = 2 to 8 ------------------------------------
    silhouette_scores: dict[int, float] = {}
    inertias: dict[int, float] = {}

    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X_pca)
        sil_score = silhouette_score(X_pca, labels)
        silhouette_scores[k] = sil_score
        inertias[k] = kmeans.inertia_

    # --- Select optimal k by highest silhouette score -----------------------
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"[CLUSTERING] Optimal k = {optimal_k} (silhouette score = {silhouette_scores[optimal_k]:.4f})")

    # --- Train final KMeans with optimal k ----------------------------------
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = final_kmeans.fit_predict(X_pca)

    # --- Print silhouette score ---------------------------------------------
    final_sil = silhouette_score(X_pca, cluster_labels)
    print(f"[CLUSTERING] Final silhouette score: {final_sil:.4f}")

    # --- Print patient count per cluster ------------------------------------
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("[CLUSTERING] Patient count per cluster:")
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} patients")

    # --- Print mean feature values per cluster ------------------------------
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f"PC{i + 1}" for i in range(X_pca.shape[1])],
    )
    pca_df["Cluster"] = cluster_labels
    cluster_means = pca_df.groupby("Cluster").mean()
    print("[CLUSTERING] Mean feature values per cluster:")
    print(cluster_means.to_string())

    # --- Save elbow plot (inertia vs k) -------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ks = sorted(inertias.keys())
    ax.plot(ks, [inertias[k] for k in ks], "bo-")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Plot — KMeans Inertia vs k")
    ax.set_xticks(ks)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "clustering_elbow.png"))
    plt.close(fig)
    print(f"[CLUSTERING] Elbow plot saved to {output_dir}/clustering_elbow.png")

    # --- Save 2D cluster scatter plot (first two PCA components) ------------
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=cluster_labels,
        cmap="viridis",
        alpha=0.6,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"KMeans Clustering (k = {optimal_k}) — First Two PCA Components")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "clustering_scatter.png"))
    plt.close(fig)
    print(f"[CLUSTERING] Scatter plot saved to {output_dir}/clustering_scatter.png")


# ---------------------------------------------------------------------------
# Stage 10 — Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the full HeartLink pipeline from data loading to evaluation.

    Sets the global random seed (NumPy, Python ``random`` module) to 42,
    ensures the ``outputs/`` directory exists, then executes every pipeline
    stage in sequence:

    1. Load data
    2. Create target variable
    3. Run exploratory data analysis
    4. Preprocess for classification
    5. Split data (train / test)
    6. Train classifier
    7. Evaluate classifier
    8. Run regression analysis
    9. Run clustering analysis

    Finally, prints a summary listing all generated output files.
    """
    # --- Set global random seed for reproducibility -------------------------
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # --- Ensure outputs directory exists ------------------------------------
    ensure_output_dir(OUTPUT_DIR)

    # --- Stage 1: Load data -------------------------------------------------
    df = load_data(DATA_PATH)

    # --- Stage 2: Create target variable ------------------------------------
    features_df, target = create_target_variable(df)

    # --- Stage 3: Exploratory data analysis ---------------------------------
    run_eda(features_df, target, OUTPUT_DIR)

    # --- Stage 4: Preprocess for classification -----------------------------
    X, feature_names = preprocess_classification(features_df)

    # --- Stage 5: Train-test split ------------------------------------------
    X_train, X_test, y_train, y_test = split_data(X, target)

    # --- Stage 6: Train classifier ------------------------------------------
    model = train_classifier(X_train, y_train)

    # --- Stage 7: Evaluate classifier ---------------------------------------
    metrics = evaluate_classifier(model, X_test, y_test, OUTPUT_DIR)

    # --- Stage 8: Regression analysis ---------------------------------------
    variants = generate_regression_variants(features_df)
    comparison_df = train_and_evaluate_regression(variants, OUTPUT_DIR)

    # --- Stage 9: Clustering analysis ---------------------------------------
    run_clustering_analysis(features_df, OUTPUT_DIR)

    # --- Final summary: list all generated output files ---------------------
    print("\n" + "=" * 60)
    print("HeartLink Pipeline — Complete")
    print("=" * 60)
    print(f"\nAll outputs saved to '{OUTPUT_DIR}/':\n")

    output_files = sorted(os.listdir(OUTPUT_DIR))
    for fname in output_files:
        print(f"  • {fname}")

    print(f"\nTotal files generated: {len(output_files)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
