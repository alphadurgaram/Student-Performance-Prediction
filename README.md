
# Student Performance Prediction Using Machine Learning

A Machine Learning project that predicts whether a student will pass or fail based on demographic information, test preparation status, and reading and writing scores.

## Project Overview

This project uses the Students Performance dataset to explore student academic performance and build classification models.

The project includes data preprocessing, model training, evaluation, prediction, and an interactive Streamlit dashboard.

**Note:** For this project, a student is classified as Pass if their math score is greater than or equal to 40; otherwise, they are classified as Fail. This is a project-defined threshold, not an official pass criterion provided by the dataset.

## Features

- Student performance data analysis
- Data preprocessing and feature encoding
- Logistic Regression classification
- Random Forest classification
- Model evaluation and comparison
- Individual student prediction
- Interactive Streamlit dashboard
- Confusion matrix visualization

## Dataset

Dataset: StudentsPerformance.csv

- Total records: 1,000
- Total original columns: 8
- Categorical features: 5
- Numerical input features: 2

### Input Features

- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch
- Test Preparation Course
- Reading Score
- Writing Score

### Target Variable

- Pass (1): Math score >= 40
- Fail (0): Math score < 40

## Machine Learning Models

1. Logistic Regression
2. Random Forest Classifier

The models use preprocessing pipelines with missing-value imputation, feature scaling, and one-hot encoding.

The selected model is Logistic Regression, based on the cross-validation results, particularly its Fail-class recall and balanced accuracy.

## Project Structure

```text
Student-Performance-Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── StudentsPerformance.csv
│   └── processed/
│
├── models/
│   ├── best_model.pkl
│   ├── logistic_regression_model.pkl
│   └── random_forest_model.pkl
│
├── notebooks/
│   └── 01_Student_Performance_Prediction.ipynb
│
├── reports/
│   ├── cross_validation_results.csv
│   ├── model_evaluation_results.csv
│   └── figures/
│
├── src/
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Student-Performance-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

### Train Models

```bash
python src/train_model.py
```

### Evaluate Models

```bash
python src/evaluate_model.py
```

### Run Individual Prediction

```bash
python src/predict.py
```

### Launch Streamlit Dashboard

```bash
streamlit run app/app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Cross-validation

The dataset is imbalanced: 960 Pass records and 40 Fail records. The test set contains only 8 Fail records, so class-wise metrics should be interpreted cautiously.

Random Forest achieved a higher Fail-class F1 score on the test set, while Logistic Regression achieved better Fail recall and balanced accuracy in the reported cross-validation results. The selected model should therefore be treated as a provisional choice rather than a definitive winner.

## Limitations

- The dataset contains relatively few Fail examples.
- The Pass/Fail threshold of 40 is project-defined.
- Predictions depend on the quality and availability of input scores.
- Model probabilities are estimates, not guarantees.
- Reading and writing scores must be available when making a prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Author

**Durga Ram Singla**

B.Tech CSE Student | Aspiring Software Developer

## Disclaimer

This project is developed for educational and learning purposes. Predictions should not be used as the sole basis for real academic decisions.
