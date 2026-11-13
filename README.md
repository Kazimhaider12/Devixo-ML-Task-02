# Devixo Solutions — Task 02: Build and Compare Machine Learning Models

## 📌 Project Overview
This repository contains the complete implementation for **Devixo Solutions Machine Learning Internship — Task 02**. The project solves a real-world **Customer Churn Prediction** problem (>1,500 records) by constructing a leakage-free preprocessing pipeline, training six classification algorithms, comparing their performance across multiple evaluation metrics, and optimizing the best model using `GridSearchCV`.

---

## 🛠️ Technologies & Libraries Used
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Data Visualization:** `matplotlib`, `seaborn`

---

## 📂 Repository Structure
```text
├── task_02_model_comparison.py       # Main end-to-end ML pipeline script
├── customer_churn_dataset.csv        # Generated/Loaded 1,520-record dataset
├── outputs/
│   ├── confusion_matrices.png        # Visual confusion matrix grid for all models
│   ├── model_comparison_chart.png    # Grouped bar chart of Accuracy, Precision, Recall, F1
│   └── model_comparison_results.csv  # Exported metrics table
└── README.md                         # Project documentation and comparison report
```

---

## ⚙️ Workflow & Task Requirements Completed

### Part 1 — Data Preprocessing
1. **Dataset Loading:** Loaded 1,520 customer records with 7 features (`Tenure_Months`, `Monthly_Charges`, `Total_Charges`, `Support_Calls`, `Contract_Type`, `Internet_Service`, `Payment_Method`) and target `Churn`.
2. **Duplicate Removal:** Identified and dropped 20 duplicate rows (`1,500` unique records retained).
3. **Missing Value Imputation:** Imputed missing numerical values using `SimpleImputer(strategy='median')` and missing categorical values using `SimpleImputer(strategy='most_frequent')`.
4. **Categorical Encoding:** Applied `OneHotEncoder(drop='first')` inside a `ColumnTransformer`.
5. **Feature Scaling:** Standardized numerical features via `StandardScaler()`.
6. **Train-Test Split:** Split dataset 80/20 (`1,200` train / `300` test) using stratified sampling (`stratify=y`).

### Part 2 & 3 — Model Development & Evaluation
Trained and evaluated six classification models on the test set:
* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM - RBF)
* XGBoost Classifier

---

## 📊 Part 4 — Model Comparison Table

| Model Name | Training Time | Accuracy | Precision | Recall | F1 Score | Advantages | Limitations |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Random Forest (Tuned)** | 0.2410s | **81.67%** | **79.85%** | **78.68%** | **0.7926** | Highest generalization; handles non-linear interactions and reduces variance via bagging. | Slower inference than linear models; less directly interpretable. |
| **Random Forest (Base)** | 0.2185s | 80.33% | 78.52% | 77.21% | 0.7786 | Robust to outliers and unscaled features; strong out-of-the-box performance. | Larger memory footprint than single trees. |
| **XGBoost** | 0.1492s | 79.67% | 77.78% | 76.47% | 0.7712 | Sequential gradient boosting with built-in L1/L2 regularization. | Sensitive to hyperparameter choices on noisy data. |
| **SVM (RBF)** | 0.0940s | 78.33% | 76.69% | 75.00% | 0.7584 | Strong non-linear boundary separation in high-dimensional space. | Strictly requires feature scaling; slower on very large datasets. |
| **Logistic Regression** | 0.0085s | 77.00% | 75.57% | 72.79% | 0.7416 | Extremely fast training; outputs calibrated probabilities; interpretable. | Assumes linear relationship between features and log-odds. |
| **Decision Tree** | 0.0062s | 74.67% | 72.79% | 70.59% | 0.7167 | Easy to visualize and interpret; no feature scaling required. | Prone to overfitting and high variance without ensemble averaging. |
| **KNN (k=7)** | 0.0038s | 73.33% | 71.32% | 69.12% | 0.7020 | Simple non-parametric algorithm; zero training time overhead. | High prediction latency; sensitive to irrelevant or one-hot encoded features. |

---

## 🏆 Part 5 — Conclusion & Model Recommendation
**Recommended Model:** **Random Forest Classifier (optimized via `GridSearchCV`)**

**Why It Performed Better:**
1. **Captures Complex Feature Interactions:** Customer churn is strongly driven by non-linear combinations (e.g., customers on *month-to-month* contracts who also have *Fiber optic* service and high *support calls*). Tree ensembles naturally isolate these multi-variable splits.
2. **Overfitting Prevention:** While a standalone Decision Tree achieved an F1 score of `0.7167` due to variance on split boundaries, averaging across 200 trees in Random Forest boosted the F1 score to `0.7926`.
3. **Hyperparameter Optimization (Bonus):** Using 5-fold `GridSearchCV` (`max_depth=10`, `min_samples_split=5`, `n_estimators=200`) improved test Accuracy to **81.67%** and Recall to **78.68%**, ensuring the business catches the maximum number of at-risk customers.

---

## 🚀 How to Run Locally
```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn xgboost

# 2. Run the script
python task_02_model_comparison.py
```