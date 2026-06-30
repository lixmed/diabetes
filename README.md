# 30-Day Hospital Readmission Prediction for Diabetic Patients

Predicting whether a diabetic patient will be readmitted within 30 days of discharge using the UCI Diabetes Hospital dataset. The project compares Logistic Regression and Random Forest classifiers with hyperparameter tuning and cross-validation.

## Dataset

**UCI Diabetes Hospital Readmission Dataset** (`diabetic_data.csv`) — contains 10 years of clinical care data from 130 US hospitals (1999–2008).

- **Target**: `readmitted < 30` (binary — readmitted within 30 days or not)
- **Original features**: encounter ID, patient number, race, gender, age, admission type, discharge disposition, admission source, time in hospital, medical specialty, number of lab procedures, number of medications, diabetes medications, diagnoses, and more.
- **Size**: ~100k encounters with ~50 features after cleaning.

## Project Structure

```
diabetes/
├── data/
│   └── diabetic_data.csv       # Dataset (not tracked in git)
├── models/
│   ├── logistic_regression.pkl # Trained Logistic Regression model
│   └── random_forest_best.pkl  # Best tuned Random Forest model
├── outputs/
│   ├── cv_fold_results.csv     # Per-fold cross-validation metrics
│   ├── model_comparison.csv    # Test-set comparison table
│   ├── cv_fold_performance.png
│   ├── feature_importance.png
│   ├── missing_values.png
│   ├── model_comparison.png
│   ├── rf_confusion_matrix.png
│   ├── roc_comparison.png
│   └── target_distribution.png
├── readmission_prediction.ipynb # Main analysis notebook
├── requirements.txt
└── README.md
```

## Pipeline

1. **Data Cleaning** — Replace `?` with NaN; drop columns with >80% missing values and ID fields.
2. **Preprocessing** — Median imputation for numeric features; most-frequent imputation + one-hot encoding for categorical features.
3. **Modeling**:
   - **Logistic Regression** (balanced class weights) — baseline.
   - **Random Forest** (balanced class weights) — tuned via `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` using 5-fold CV with ROC AUC scoring.
4. **Cross-Validation** — Stratified 5-fold CV with the best hyperparameters.
5. **Evaluation** — Accuracy, Precision, Recall, F1, ROC AUC, confusion matrix, ROC curve, and feature importance.

## Results

| Metric              | Logistic Regression | Random Forest (tuned) |
|---------------------|---------------------|------------------------|
| Accuracy            | —                   | —                      |
| Precision           | —                   | —                      |
| Recall              | —                   | —                      |
| F1                  | —                   | —                      |
| **ROC AUC**         | —                   | —                      |

*Metrics are populated after running the notebook. The Random Forest model typically outperforms Logistic Regression on ROC AUC.*

## Setup

```bash
pip install -r requirements.txt
```

Place `diabetic_data.csv` from the [UCI Diabetes dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) into the `data/` directory, then run the notebook.

## Requirements

- Python 3.8+
- numpy, pandas, matplotlib, seaborn, scikit-learn, joblib

## License

MIT
