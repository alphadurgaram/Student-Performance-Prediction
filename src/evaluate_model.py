
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR, "data", "raw", "StudentsPerformance.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

FIGURE_DIR = os.path.join(
    BASE_DIR, "reports", "figures"
)

REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(FIGURE_DIR, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["math_result"] = (df["math score"] >= 40).astype(int)

X = df.drop(columns=["math score", "math_result"])
y = df["math_result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully!")


# ==========================================
# 3. LOAD TRAINED MODELS
# ==========================================

print("\nLoading trained models...")

models = {
    "Logistic Regression": joblib.load(
        os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
    ),
    "Random Forest": joblib.load(
        os.path.join(MODEL_DIR, "random_forest_model.pkl")
    )
}

print("Models loaded successfully!")


# ==========================================
# 4. MODEL EVALUATION
# ==========================================

results = []

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, model) in zip(axes, models.items()):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, pos_label=0, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, pos_label=0, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, pos_label=0, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Fail Precision": precision,
        "Fail Recall": recall,
        "Fail F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fail", "Pass"],
        zero_division=0
    ))

    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"],
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


# ==========================================
# 5. SAVE CONFUSION MATRIX
# ==========================================

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "confusion_matrix_evaluation.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved successfully!")


# ==========================================
# 6. SAVE MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "model_evaluation_results.csv"
    ),
    index=False
)


# ==========================================
# 7. FINAL OUTPUT
# ==========================================

print("\n" + "=" * 50)
print("MODEL EVALUATION COMPLETED!")
print("=" * 50)

print("Evaluation report saved in:", REPORT_DIR)
print("Confusion matrix saved in:", FIGURE_DIR)
