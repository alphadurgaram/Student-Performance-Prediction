
import joblib
import os
import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_model.pkl"
)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

print("Loading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# 3. TAKE STUDENT INPUT
# ==========================================

print("\nEnter Student Details")
print("-" * 35)

gender = input("Gender (male/female): ").strip().lower()

race = input(
    "Race/Ethnicity (group A/B/C/D/E): "
).strip()

parent_education = input(
    "Parental Education: "
).strip()

lunch = input(
    "Lunch (standard/free-reduced): "
).strip().lower()

test_prep = input(
    "Test Preparation (completed/none): "
).strip().lower()

reading_score = float(input("Reading Score (0-100): "))
writing_score = float(input("Writing Score (0-100): "))


# ==========================================
# 4. PREPARE INPUT DATA
# ==========================================

student_data = pd.DataFrame([{
    "gender": gender,
    "race/ethnicity": race,
    "parental level of education": parent_education,
    "lunch": lunch,
    "test preparation course": test_prep,
    "reading score": reading_score,
    "writing score": writing_score
}])


# ==========================================
# 5. PREDICT RESULT
# ==========================================

prediction = model.predict(student_data)[0]

probabilities = model.predict_proba(student_data)[0]

fail_probability = probabilities[0] * 100
pass_probability = probabilities[1] * 100


# ==========================================
# 6. DISPLAY RESULT
# ==========================================

print("\n" + "=" * 40)
print("STUDENT PERFORMANCE PREDICTION")
print("=" * 40)

if prediction == 1:
    print("Predicted Result: PASS")
else:
    print("Predicted Result: FAIL")

print(f"Pass Probability: {pass_probability:.2f}%")
print(f"Fail Probability: {fail_probability:.2f}%")

print("=" * 40)
print("Prediction completed successfully!")
