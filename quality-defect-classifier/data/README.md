# Data

The SECOM dataset isn't bundled in this repo — keep raw data out of git.

## How to get it
1. Go to the UCI Machine Learning Repository SECOM page:
   https://archive.ics.uci.edu/dataset/179/secom
2. Click "Download" — you'll get a zip with two files:
   - `secom.data` — the 590 sensor feature values (space-delimited, no header)
   - `secom_labels.data` — pass/fail label (-1 = pass, 1 = fail) plus a timestamp column
3. Extract both files into this `data/` folder.

## Format note
- `secom.data`: 1567 rows × 590 columns, space-delimited, no header row. Contains missing values (represented as `NaN`).
- `secom_labels.data`: 1567 rows × 2 columns — label and timestamp, space-delimited.

Add a `.gitignore` entry so raw data files aren't committed to GitHub — only processed/summary outputs should be.

## Results

Class imbalance made this a non-trivial classification problem — only 6.6% of units failed. A naive model predicting "pass" every time would score 93% accuracy while catching zero defects.

| Model | Fail Precision | Fail Recall | Fail F1 |
|---|---|---|---|
| Logistic Regression (unweighted baseline) | 0.10 | 0.05 | 0.06 |
| Logistic Regression (class-weighted) | 0.88 | 0.82 | 0.85 |
| Random Forest (class-weighted) | 0.00 | 0.00 | 0.00 |
| Random Forest + SMOTE | 0.33 | 0.05 | 0.08 |

**Key findings:**
- Reduced feature space from 590 to 297 sensors by removing high-missing-value and near-zero-variance columns.
- Plain accuracy is misleading on this dataset: the unweighted baseline hit 91% accuracy while catching only 5% of actual defects.
- Class-weighted Logistic Regression was the strongest model by a wide margin, improving Fail recall from 5% to 82%.
- Random Forest, with or without SMOTE oversampling, underperformed Logistic Regression on this dataset — a reminder that more complex models don't always win, especially on small, high-dimensional, imbalanced data.

![Feature Importance](outputs/feature_importance.png)
![Confusion Matrix](outputs/confusion_matrix.png)

**Interpretation:** In a quality control context, missing a real defect (false negative) is typically far costlier than a false alarm (false positive). This project prioritized recall on the Fail class accordingly — the same logic Six Sigma practitioners apply when weighing inspection sensitivity against cost.
