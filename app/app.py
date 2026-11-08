
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_model.pkl"
)

LOGISTIC_PATH = os.path.join(
    BASE_DIR, "models", "logistic_regression_model.pkl"
)

RF_PATH = os.path.join(
    BASE_DIR, "models", "random_forest_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR, "data", "raw", "StudentsPerformance.csv"
)

REPORT_PATH = os.path.join(
    BASE_DIR, "reports", "model_evaluation_results.csv"
)


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# 3. LOAD MODELS AND DATA
# ==========================================

@st.cache_resource
def load_models():
    return {
        "Best Model": joblib.load(MODEL_PATH),
        "Logistic Regression": joblib.load(LOGISTIC_PATH),
        "Random Forest": joblib.load(RF_PATH)
    }


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


models = load_models()
df = load_data()


# ==========================================
# 4. SIDEBAR
# ==========================================

st.sidebar.title("🎓 Student Predictor")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Student Prediction",
        "Model Comparison"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Machine Learning Project\n\n"
    "Student Performance Prediction"
)


# ==========================================
# 5. DASHBOARD
# ==========================================

if page == "Dashboard":

    st.title("🎓 Student Performance Dashboard")

    st.write(
        "Explore student performance data and "
        "machine learning insights."
    )

    total_students = len(df)
    average_math = df["math score"].mean()
    average_reading = df["reading score"].mean()
    average_writing = df["writing score"].mean()

    pass_count = (df["math score"] >= 40).sum()
    fail_count = (df["math score"] < 40).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", total_students)
    col2.metric("Average Math Score", f"{average_math:.2f}")
    col3.metric("Average Reading Score", f"{average_reading:.2f}")
    col4.metric("Average Writing Score", f"{average_writing:.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pass vs Fail Distribution")

        fig, ax = plt.subplots()

        ax.pie(
            [pass_count, fail_count],
            labels=["Pass", "Fail"],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Average Scores")

        score_data = pd.DataFrame({
            "Subject": ["Math", "Reading", "Writing"],
            "Average Score": [
                average_math,
                average_reading,
                average_writing
            ]
        })

        fig, ax = plt.subplots()

        sns.barplot(
            data=score_data,
            x="Subject",
            y="Average Score",
            ax=ax
        )

        ax.set_ylim(0, 100)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)


# ==========================================
# 6. STUDENT PREDICTION
# ==========================================

elif page == "Student Prediction":

    st.title("🔍 Student Performance Prediction")

    st.write(
        "Enter student details to predict the "
        "mathematics result."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["male", "female"]
            )

            race = st.selectbox(
                "Race/Ethnicity",
                [
                    "group A",
                    "group B",
                    "group C",
                    "group D",
                    "group E"
                ]
            )

            parent_education = st.selectbox(
                "Parental Education",
                [
                    "some high school",
                    "high school",
                    "some college",
                    "associate's degree",
                    "bachelor's degree",
                    "master's degree"
                ]
            )

        with col2:

            lunch = st.selectbox(
                "Lunch",
                ["standard", "free/reduced"]
            )

            test_prep = st.selectbox(
                "Test Preparation",
                ["completed", "none"]
            )

            reading_score = st.slider(
                "Reading Score",
                0, 100, 75
            )

            writing_score = st.slider(
                "Writing Score",
                0, 100, 75
            )

        submitted = st.form_submit_button(
            "Predict Student Result",
            use_container_width=True
        )

    if submitted:

        student_data = pd.DataFrame([{
            "gender": gender,
            "race/ethnicity": race,
            "parental level of education": parent_education,
            "lunch": lunch,
            "test preparation course": test_prep,
            "reading score": reading_score,
            "writing score": writing_score
        }])

        model = models["Best Model"]

        prediction = model.predict(student_data)[0]

        probabilities = model.predict_proba(
            student_data
        )[0]

        fail_probability = probabilities[0] * 100
        pass_probability = probabilities[1] * 100

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("🎉 Predicted Result: PASS")
        else:
            st.error("⚠️ Predicted Result: FAIL")

        col1, col2 = st.columns(2)

        col1.metric(
            "Pass Probability",
            f"{pass_probability:.2f}%"
        )

        col2.metric(
            "Fail Probability",
            f"{fail_probability:.2f}%"
        )

        st.caption(
            "These are model-estimated probabilities, "
            "not guaranteed outcomes. The project uses "
            "40 as its defined mathematics threshold."
        )


# ==========================================
# 7. MODEL COMPARISON
# ==========================================

elif page == "Model Comparison":

    st.title("📊 Model Performance Comparison")

    st.write(
        "Compare Logistic Regression and Random Forest."
    )

    if os.path.exists(REPORT_PATH):

        results = pd.read_csv(REPORT_PATH)

        st.subheader("Evaluation Metrics")

        st.dataframe(
            results.round(4),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Accuracy Comparison")

        fig, ax = plt.subplots()

        sns.barplot(
            data=results,
            x="Model",
            y="Accuracy",
            ax=ax
        )

        ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Fail Detection Comparison")

        fail_metrics = results.melt(
            id_vars="Model",
            value_vars=[
                "Fail Precision",
                "Fail Recall",
                "Fail F1 Score"
            ],
            var_name="Metric",
            value_name="Score"
        )

        fig, ax = plt.subplots()

        sns.barplot(
            data=fail_metrics,
            x="Metric",
            y="Score",
            hue="Model",
            ax=ax
        )

        ax.set_ylim(0, 1)
        plt.xticks(rotation=15)
        st.pyplot(fig)
        plt.close(fig)

    else:

        st.warning(
            "Model evaluation report not found. "
            "Run python src/evaluate_model.py first."
        )

    st.info(
        "Note: The test dataset contains only 8 Fail "
        "students. Interpret Fail detection metrics "
        "with caution."
    )
