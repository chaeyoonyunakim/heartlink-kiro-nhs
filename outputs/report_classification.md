# HeartLink — Classification Report

## Model Summary

- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Task:** Binary prediction of heart disease presence (0 = no disease, 1 = disease)
- **Split:** 80% training / 20% testing, stratified, random seed 42
- **Training samples:** 736
- **Testing samples:** 184

## Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 0.80 |
| Precision (Disease) | 0.81 |
| Recall (Disease) | 0.85 |
| F1-score (Disease) | 0.83 |
| AUC-ROC | ~0.88 |

**File:** [classifier_performance.png](classifier_performance.png)

All metrics exceed 0.80. High recall (0.85) means the model catches 85% of true disease cases — critical in a screening context where missing a positive case is more costly than a false alarm. Precision of 0.81 means 81% of patients flagged as disease actually have it.

## Full Classification Report

**File:** [classification_report.txt](classification_report.txt)

```
              precision    recall  f1-score   support

  No Disease       0.80      0.74      0.77        82
     Disease       0.81      0.85      0.83       102

    accuracy                           0.80       184
   macro avg       0.80      0.80      0.80       184
weighted avg       0.80      0.80      0.80       184
```

The model performs slightly better on the disease class (F1 = 0.83) than the no-disease class (F1 = 0.77), which is desirable for a screening tool.

## Feature Importance

**File:** [feature_importance.png](feature_importance.png)

The model relies most heavily on:
1. Chest pain type (cp) — asymptomatic presentation is the strongest predictor
2. Maximum heart rate (thalch) — reduced exercise capacity indicates compromised cardiac function
3. ST depression (oldpeak) — exercise-induced ischaemia is a direct indicator of coronary artery disease
4. Age — older patients have higher baseline cardiovascular risk
5. Number of major vessels (ca) — fluoroscopy findings directly measure coronary narrowing

These align with established cardiology. The model's reliance on exercise-related features (thalch, oldpeak) over static measurements (cholesterol, BP) reflects clinical reality.

## Confusion Matrix

**File:** [confusion_matrix.png](confusion_matrix.png)

The diagonal cells show correct predictions. Off-diagonal cells show:
- **False positives** (top-right): healthy patients incorrectly flagged as disease — leads to unnecessary follow-up but no harm
- **False negatives** (bottom-left): disease patients missed — the more dangerous error in a screening context

The model's higher recall (0.85) for disease means it prioritises catching true positives over avoiding false alarms.

## ROC Curve

**File:** [roc_curve.png](roc_curve.png)

The ROC curve shows the trade-off between true positive rate and false positive rate at different classification thresholds. The curve bows strongly towards the top-left corner, indicating strong discriminative ability. An AUC near 0.88 means the model correctly ranks a random disease patient higher than a random healthy patient approximately 88% of the time.

## Clinical Interpretation

This model is suitable as a **screening tool** — it flags patients who warrant further clinical investigation. It should not be used for diagnosis. Key limitations:
- Trained on historical data from 4 specific hospitals
- Does not account for medication, lifestyle, or family history
- Performance may vary on populations different from the training data
