# ==============================================================================
# DEVIXO SOLUTIONS - TASK 02: BUILD AND COMPARE MACHINE LEARNING MODELS
# Problem Domain: Customer Churn Prediction (1,500+ Records)
# Technologies: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
# ==============================================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def load_or_generate_dataset(csv_path="customer_churn_dataset.csv"):
    """
    Loads the Kaggle Customer Churn dataset if present in the directory,
    or generates and saves a realistic 1,520-record Telco Churn CSV dataset.
    """
    if os.path.exists(csv_path):
        print(f"[INFO] Loading dataset from {csv_path}...")
        return pd.read_csv(csv_path)

    print("[INFO] Generating realistic 1,520-record Customer Churn dataset...")
    np.random.seed(42)
    n_samples = 1500

    data = {
        "Tenure_Months": np.random.randint(1, 72, n_samples),
        "Monthly_Charges": np.round(np.random.uniform(20.0, 120.0, n_samples), 2),
        "Support_Calls": np.random.poisson(1.8, n_samples),
        "Contract_Type": np.random.choice(
            ["Month-to-month", "One year", "Two year"],
            n_samples,
            p=[0.55, 0.25, 0.20]
        ),
        "Internet_Service": np.random.choice(
            ["DSL", "Fiber optic", "No"],
            n_samples,
            p=[0.35, 0.45, 0.20]
        ),
        "Payment_Method": np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            n_samples
        ),
    }

    df = pd.DataFrame(data)
    df["Total_Charges"] = np.round(
        df["Tenure_Months"] * df["Monthly_Charges"] + np.random.normal(0, 45, n_samples), 2
    )

    # Realistic non-linear churn probability
    logit = (
        -1.2
        + 1.35 * (df["Contract_Type"] == "Month-to-month").astype(float)
        + 0.85 * (df["Internet_Service"] == "Fiber optic").astype(float)
        + 0.35 * (df["Support_Calls"] >= 3).astype(float)
        - 0.028 * df["Tenure_Months"]
        + 0.012 * df["Monthly_Charges"]
    )
    prob = 1 / (1 + np.exp(-logit))
    df["Churn"] = (np.random.rand(n_samples) < prob).astype(int)

    # Inject missing values & duplicate rows to demonstrate Part 1 preprocessing
    df.loc[np.random.choice(df.index, 25, replace=False), "Total_Charges"] = np.nan
    df.loc[np.random.choice(df.index, 15, replace=False), "Internet_Service"] = np.nan
    df = pd.concat([df, df.iloc[:20]], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved raw dataset to '{csv_path}' (Shape: {df.shape})")
    return df


def main():
    os.makedirs("outputs", exist_ok=True)

    # ==========================================================================
    # PART 1: DATA PREPROCESSING
    # ==========================================================================
    print("\n" + "=" * 75)
    print("PART 1: DATA PREPROCESSING")
    print("=" * 75)

    df = load_or_generate_dataset()
    print(f"1. Initial Raw Dataset Shape       : {df.shape}")
    print(f"2. Initial Missing Values Count    :\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")

    # Remove duplicates
    duplicates_count = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"3. Removed {duplicates_count} duplicate rows -> Clean Shape: {df.shape}")

    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    num_cols = ["Tenure_Months", "Monthly_Charges", "Total_Charges", "Support_Calls"]
    cat_cols = ["Contract_Type", "Internet_Service", "Payment_Method"]

    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"4. Train/Test Split Complete       : Train={X_train.shape[0]} rows, Test={X_test.shape[0]} rows")

    # Preprocessing Pipelines (Imputation + Encoding + Scaling)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"5. Feature Matrix Post-Encoding    : {X_train_processed.shape[1]} engineered features")

    # ==========================================================================
    # PART 2 & PART 3: MODEL DEVELOPMENT & EVALUATION
    # ==========================================================================
    print("\n" + "=" * 75)
    print("PART 2 & 3: MODEL TRAINING AND EVALUATION")
    print("=" * 75)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel="rbf", random_state=42),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(random_state=42, eval_metric="logloss")

    results = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        start_t = time.perf_counter()
        model.fit(X_train_processed, y_train)
        train_time = round(time.perf_counter() - start_t, 4)

        y_pred = model.predict(X_test_processed)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            "Model Name": name,
            "Training Time (s)": train_time,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "Confusion Matrix": f"[[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]"
        })

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx])
        axes[idx].set_title(f"{name}\nAcc: {acc:.3f} | F1: {f1:.3f}")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("Actual Label")

    # Hide unused subplot if XGBoost is not installed
    for j in range(len(models), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/confusion_matrices.png", dpi=300)
    plt.close()
    print("[INFO] Saved confusion matrices plot to 'outputs/confusion_matrices.png'")

    # ==========================================================================
    # PART 4: MODEL COMPARISON TABLE
    # ==========================================================================
    print("\n" + "=" * 75)
    print("PART 4: MODEL COMPARISON TABLE")
    print("=" * 75)

    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
    print(results_df.to_string(index=False))
    results_df.to_csv("outputs/model_comparison_results.csv", index=False)

    # Generate visual comparison bar chart
    melted = results_df.melt(
        id_vars="Model Name",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score"
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x="Model Name", y="Score", hue="Metric", palette="viridis")
    plt.title("Machine Learning Model Performance Comparison", fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("outputs/model_comparison_chart.png", dpi=300)
    plt.close()
    print("[INFO] Saved performance comparison chart to 'outputs/model_comparison_chart.png'")

    # ==========================================================================
    # BONUS: HYPERPARAMETER TUNING (GridSearchCV)
    # ==========================================================================
    print("\n" + "=" * 75)
    print("BONUS: HYPERPARAMETER TUNING VIA GRIDSEARCHCV")
    print("=" * 75)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    grid_search.fit(X_train_processed, y_train)

    best_rf = grid_search.best_estimator_
    y_tuned_pred = best_rf.predict(X_test_processed)

    print(f"Best Hyperparameters : {grid_search.best_params_}")
    print(f"Tuned Accuracy       : {accuracy_score(y_test, y_tuned_pred):.4f}")
    print(f"Tuned Precision      : {precision_score(y_test, y_tuned_pred):.4f}")
    print(f"Tuned Recall         : {recall_score(y_test, y_tuned_pred):.4f}")
    print(f"Tuned F1 Score       : {f1_score(y_test, y_tuned_pred):.4f}")
    print("\nDetailed Classification Report (Tuned Random Forest):")
    print(classification_report(y_test, y_tuned_pred))


if __name__ == "__main__":
    main()