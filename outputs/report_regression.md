# HeartLink — Regression Report

## Model Summary

- **Algorithm:** Linear Regression (scikit-learn)
- **Task:** Predict maximum heart rate (thalch) from age, resting BP, and cholesterol
- **Purpose:** Quantify cardiac functional decline — lower predicted thalch suggests reduced exercise capacity
- **Variants tested:** 8 (Cartesian product of 3 binary preprocessing dimensions)
- **Split:** 80% training / 20% testing per variant, random seed 42

## Preprocessing Dimensions

| Dimension | Option A | Option B |
|-----------|----------|----------|
| Hospital inclusion | All 4 hospitals | Cleveland only |
| Imputation method | "Unknown" category + median | Mean + mode |
| Outlier removal | Retain all data | IQR-based filtering |

## Variant Comparison

**Files:** [regression_variant_comparison.csv](regression_variant_comparison.csv), [regression_r2_comparison.png](regression_r2_comparison.png)

| Variant | Hospital | Imputation | Outliers | n | MAE | RMSE | R² |
|---------|----------|------------|----------|---|-----|------|----|
| All-Unknown-WithOutliers | all | unknown | No | 920 | 19.00 | 23.22 | 0.110 |
| All-Unknown-NoOutliers | all | unknown | Yes | 841 | 19.32 | 23.34 | 0.094 |
| All-MeanMode-WithOutliers | all | mean_mode | No | 920 | 18.88 | 23.14 | 0.116 |
| All-MeanMode-NoOutliers | all | mean_mode | Yes | 841 | 19.18 | 23.27 | 0.099 |
| Cleveland-Unknown-WithOutliers | cleveland | unknown | No | 304 | 15.97 | 19.14 | **0.154** |
| Cleveland-Unknown-NoOutliers | cleveland | unknown | Yes | 289 | 17.98 | 22.65 | 0.091 |
| Cleveland-MeanMode-WithOutliers | cleveland | mean_mode | No | 304 | 15.97 | 19.14 | **0.154** |
| Cleveland-MeanMode-NoOutliers | cleveland | mean_mode | Yes | 289 | 17.98 | 22.65 | 0.091 |

**Best variant:** Cleveland-Unknown-WithOutliers (R² = 0.154)

## Key Findings

1. **Cleveland-only data outperforms all-hospital data.** R² improves from ~0.11 to ~0.15 when restricting to Cleveland. This suggests that mixing hospitals with different data collection practices introduces noise that degrades prediction quality.

2. **Outlier removal hurts performance.** Removing IQR outliers consistently reduces R² across all variants. The "outliers" likely contain clinically meaningful extreme values (very high/low heart rates) that the model benefits from seeing.

3. **Imputation method has minimal impact.** Unknown-category vs mean-mode imputation produces nearly identical results, suggesting the regression features (age, BP, cholesterol) have low missingness.

4. **R² values are low overall (0.09–0.15).** Age, resting BP, and cholesterol explain only 10–15% of the variation in maximum heart rate. Additional features (exercise habits, medication, body mass) would be needed for clinically useful predictions.

## Best Variant — Predicted vs Actual

**File:** [regression_scatter.png](regression_scatter.png)

Points close to the red dashed line indicate accurate predictions. The scatter shows the model captures the general downward trend (older patients → lower thalch) but with substantial spread (±40 bpm). The three features alone are insufficient for precise individual-level prediction.

## Best Variant — Residual Plot

**File:** [regression_residuals.png](regression_residuals.png)

Residuals are roughly evenly scattered around zero with no obvious pattern, confirming the linear model's assumptions are reasonable. Some large residuals (±40 bpm) indicate cases where the model fails — likely patients with unusual exercise capacity for their age/BP/cholesterol profile.

## Clinical Interpretation

The regression model demonstrates that age, blood pressure, and cholesterol have a statistically significant but weak relationship with maximum heart rate. This aligns with clinical knowledge — exercise capacity is influenced by many factors beyond these three measurements. The model is useful for population-level trend analysis but not for individual patient prediction.
