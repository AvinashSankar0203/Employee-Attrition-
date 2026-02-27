# 👥 Employee Attrition Prediction (R)

Predict employee attrition (Yes/No) using classic classification models in **R**:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)

This repo is based on the project report: **“Employee Attrition Prediction — CSP571 final report.”** 

---

## 📌 What this project does

1. Loads the IBM HR Analytics dataset (CSV)
2. Preprocesses the data
   - Drops non-informative identifiers (e.g., `EmployeeNumber`, `EmployeeCount`, etc.)
   - Converts categorical variables to factors
   - Scales numeric variables
   - Splits into train/test (default 70/30)
3. Trains and evaluates four models
4. Reports metrics: Accuracy, Precision, Recall, F1
5. Plots ROC curves and computes AUC

---

## 🧾 Dataset

Recommended dataset:
- **IBM HR Analytics Employee Attrition & Performance** (Kaggle)

Expected target column:
- `Attrition` with values `Yes` / `No`

---

## 🧰 Requirements

Install packages once:

```r
install.packages(c(
  "caret","dplyr","ggplot2","randomForest","e1071","pROC","readr"
))
```

---

## ▶️ How to run

1. Put the dataset CSV into a `data/` folder, for example:
   - `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`

2. Run the script:

```bash
Rscript attrition_model.R --data data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Optional flags:

```bash
Rscript attrition_model.R --data data/your.csv --split 0.7 --seed 123 --out outputs
```

Outputs (saved to `outputs/` by default):
- `metrics.csv`
- `roc_curves.png`

---

## 📈 Model evaluation

The script prints:
- Confusion matrix for each model
- Accuracy / Precision / Recall / F1
- ROC + AUC

It also selects the “best” model by **F1 score** (changeable in code).

---

## 📂 Repo structure

```
.
├── attrition_model.R
├── README.md
├── data/                 # (you add the CSV here)
└── outputs/              # generated results
```

---

## 🔮 Future work ideas

- Hyperparameter tuning (grid/random search)
- Handling class imbalance (SMOTE / class weights)
- Feature selection + regularization (LASSO)
- Model interpretability (SHAP-like methods for tree models)
