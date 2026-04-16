# HeartLink — Clustering Report

## Model Summary

- **Algorithm:** KMeans (scikit-learn)
- **Task:** Unsupervised patient risk group discovery without diagnostic labels
- **Dimensionality reduction:** PCA with n_components=0.9 (retains ≥90% variance)
- **k selection:** Silhouette score evaluated for k = 2 to 8, optimal k selected
- **Random seed:** 42

## Preprocessing (Stricter Than Classification)

The clustering pipeline applies stricter data cleaning to prevent hospital bias:

1. **Excluded columns:** ca, thal, slope — missingness is concentrated in non-Cleveland sources (66%, 53%, 34% respectively), meaning their absence reflects collection practices, not clinical reality
2. **Row filter:** Removed samples with >30% missing values (~55 rows removed, 920 → ~865)
3. **Cholesterol imputation:** Sex-and-age-group medians (reflects medical trends where cholesterol varies by sex and age bracket)
4. **Remaining imputation:** Column median for numerical features, column mode for categorical features
5. **Encoding:** One-hot encoding for categorical features
6. **Scaling:** StandardScaler (zero mean, unit variance)
7. **PCA:** Retains ≥90% cumulative explained variance

## Elbow Plot

**File:** [clustering_elbow.png](clustering_elbow.png)

Inertia (within-cluster sum of squares) decreases as k increases. The "elbow" — where the rate of decrease slows — suggests the natural number of clusters. Beyond this point, adding clusters provides diminishing returns and risks overfitting to noise.

## Cluster Scatter Plot

**File:** [clustering_scatter.png](clustering_scatter.png)

Each dot represents a patient, coloured by cluster assignment, plotted on the first two principal components from PCA. Distinct colour groupings indicate the model has found meaningful patient subgroups. Overlap between clusters is expected — cardiovascular risk exists on a spectrum, not in discrete categories.

## Clinical Interpretation

The clusters may correspond to different risk profiles:

- **Low-risk cluster:** Younger patients with normal BP, lower cholesterol, higher exercise capacity
- **Moderate-risk cluster:** Middle-aged patients with borderline elevated markers
- **High-risk cluster:** Older patients with elevated BP, higher cholesterol, reduced exercise capacity

**Important:** These cluster assignments are exploratory. They reflect data patterns, not confirmed diagnostic categories. Further clinical validation is required before using clusters for patient stratification or treatment decisions.

## Limitations

- Clustering excludes ca, thal, and slope — three of the most clinically important features — to avoid hospital bias. This means the clusters are based on a reduced feature set.
- PCA transforms the original features into abstract components, making cluster interpretation less intuitive.
- KMeans assumes spherical clusters of similar size, which may not reflect the true structure of cardiovascular risk groups.
- The silhouette score provides a quantitative measure of cluster quality but does not guarantee clinical meaningfulness.
