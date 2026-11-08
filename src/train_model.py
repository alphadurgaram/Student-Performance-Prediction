
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "StudentsPerformance.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 3. CREATE TARGET VARIABLE
# ==========================================

# 40 is the project-defined pass threshold
df["math_result"] = (df["math score"] >= 40).astype(int)

X = df.drop(columns=["math score", "math_result"])
y = df["math_result"]


# ==========================================
# 4. IDENTIFY FEATURE TYPES
# ==========================================

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 6. DATA PREPROCESSING
# ==========================================

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])


# ==========================================
# 7. CREATE MODELS
# ==========================================

logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

random_forest_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ))
])


# ==========================================
# 8. TRAIN MODELS
# ==========================================

print("\nTraining Logistic Regression...")
logistic_model.fit(X_train, y_train)

print("Logistic Regression training completed!")

print("\nTraining Random Forest...")
random_forest_model.fit(X_train, y_train)

print("Random Forest training completed!")


# ==========================================
# 9. SAVE TRAINED MODELS
# ==========================================

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    logistic_model,
    os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
)

joblib.dump(
    random_forest_model,
    os.path.join(MODEL_DIR, "random_forest_model.pkl")
)

# Provisional selected model based on current
# cross-validation priorities for Fail detection
joblib.dump(
    logistic_model,
    os.path.join(MODEL_DIR, "best_model.pkl")
)


# ==========================================
# 10. FINAL OUTPUT
# ==========================================

print("\n" + "=" * 50)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 50)

print("Models saved successfully!")
print("Selected model: Logistic Regression")
print("Model directory:", MODEL_DIR)

print("\nTarget distribution:")
print(y.value_counts().rename(index={
    0: "Fail",
    1: "Pass"
}))
