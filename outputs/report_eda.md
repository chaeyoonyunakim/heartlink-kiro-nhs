# HeartLink — Exploratory Data Analysis Report

## Dataset Overview

- **Source:** UCI Heart Disease dataset
- **Patients:** 920 across 4 hospitals (Cleveland, Hungary, Switzerland, VA Long Beach)
- **Features:** 16 columns (6 numerical, 7 categorical, 2 identifiers, 1 target)

## Correlation Heatmap

**File:** `correlation_heatmap.png`

Age and maximum heart rate (thalch) show a moderate negative correlation (r ≈ −0.4). Older patients achieve lower peak heart rates during exercise testing. Most other features are weakly correlated, confirming they capture distinct clinical information and are suitable as independent predictors.

## Target Variable Distribution

**File:** `target_distribution.png`

The dataset is roughly balanced — approximately 410 patients without heart disease (target = 0) and 510 with heart disease (target = 1). This balance means accuracy is a meaningful metric and the classifier is not biased towards one class.

## Cholesterol by Target Variable

**File:** `target_chol.png`

Cholesterol distributions overlap substantially between disease and no-disease groups. Cholesterol alone is a weak discriminator — the model needs multiple features together to make accurate predictions.

## Numerical Feature Distributions

**Files:** `distribution_age.png`, `distribution_trestbps.png`, `distribution_chol.png`, `distribution_thalch.png`, `distribution_oldpeak.png`, `distribution_ca.png`

- **Age:** Roughly normally distributed around 54 years.
- **Resting BP:** Centres around 130 mm Hg with a right tail. One zero value (biologically impossible) caught by guardrails.
- **Cholesterol:** Wide spread (100–600 mg/dl) with 172 zero values flagged as biologically impossible. Right skew justifies median imputation.
- **Max heart rate:** Centres around 140 bpm. Lower values during exercise testing are strongly associated with disease.
- **Oldpeak:** Heavily right-skewed. Most patients show minimal ST depression, but high values indicate exercise-induced ischaemia.
- **Major vessels (ca):** Most patients show 0. Higher values strongly predict disease. 66% missingness in non-Cleveland sources.

## Categorical Feature Distributions

**Files:** `cat_distribution_cp.png`, `cat_distribution_dataset.png`, `cat_distribution_ca.png`, `cat_distribution_thal.png`, `cat_distribution_slope.png`

- "Asymptomatic" is the most common chest pain type and paradoxically the strongest predictor of disease.
- Cleveland contributes the most complete records.
- ca (66%), thal (53%), and slope (34%) have high missingness concentrated in non-Cleveland sources — this hospital bias drives the clustering pipeline's exclusion of these variables.

## Boxplots by Sex and Target Variable

**Files:** `boxplot_age.png`, `boxplot_trestbps.png`, `boxplot_chol.png`, `boxplot_thalch.png`, `boxplot_oldpeak.png`, `boxplot_ca.png`

- Disease patients achieve significantly lower maximum heart rates and higher oldpeak values.
- Blood pressure and cholesterol show less separation between groups — weaker individual predictors but contribute in combination.
- Age shows slight elevation in disease patients with substantial overlap.

## Data Quality Summary

| Issue | Count | Action |
|-------|-------|--------|
| Cholesterol = 0 (biologically impossible) | 172 | Replaced with NaN, median imputed |
| Resting BP = 0 (biologically impossible) | 1 | Replaced with NaN, median imputed |
| ca — missing values | 611 (66%) | Imputed for classification, excluded from clustering |
| thal — missing values | 486 (53%) | Imputed for classification, excluded from clustering |
| slope — missing values | 309 (34%) | Imputed for classification, excluded from clustering |
