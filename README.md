# ❤️ Heart Disease Prediction Dashboard
### KNN · Logistic Regression · Naive Bayes
**C.V. Raman Global University — Department of Computer Science & Engineering**
Bachelor of Technology | March 2026 | Supervisor: Prof. Prabhat Dansena

---

## 👥 Team Members

| Name                  | Registration No  |
|-----------------------|------------------|
| Gaurav Kumar Nayak    | 2401020547       |
| Ankit Dash            | 2401020536       |
| Debasish Rout         | 2401020546       |
| Aryomon Mohapatra     | 2401021127       |
| Aritra Nandi          | 2401020507       |
| Akash Chandra Das     | 2401020515       |
| Rohan Sahoo           | 2401020570       |
| Subhajit Bera         | 2401020583       |

---

## 📌 Project Overview

This project presents an interactive Streamlit dashboard for predicting heart disease
using three machine learning algorithms trained on the UCI Heart Disease Dataset
(Cleveland Clinic Foundation). The dashboard includes live patient prediction,
visual data analysis, model performance evaluation, and clinical observations
derived from the dataset.

Heart disease remains the leading cause of death globally, accounting for approximately
17.9 million deaths per year (WHO). Early and accurate prediction using clinical
parameters can significantly improve patient outcomes. This project benchmarks
three ML models and presents findings in an interactive, clinical-grade dashboard.

---

## 📁 Project Structure

    your_folder/
    ├── heart_dashboard.py      # Main Streamlit application
    ├── heart.csv               # Dataset (UCI Heart Disease — 1,025 records)
    └── README.md               # This file

> ⚠️ Both heart_dashboard.py and heart.csv MUST be in the same folder.

---

## 📦 Dataset Information

- **Source:**   UCI Machine Learning Repository (Cleveland Heart Disease Database)
- **Kaggle:**   https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- **Records:**  1,025 patients
- **Features:** 13 clinical input features + 1 binary target variable
- **Target:**   0 = No Heart Disease | 1 = Heart Disease Present
- **Balance:**  51.3% positive (disease) · 48.7% negative (no disease)

### Feature Summary

| Column    | Full Name               | Type        | Range / Values                          |
|-----------|-------------------------|-------------|-----------------------------------------|
| age       | Age                     | Integer     | 29 – 77 years                           |
| sex       | Sex                     | Binary      | 0 = Female, 1 = Male                    |
| cp        | Chest Pain Type         | Categorical | 0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic |
| trestbps  | Resting Blood Pressure  | Integer     | 94 – 200 mmHg                           |
| chol      | Serum Cholesterol       | Integer     | 126 – 564 mg/dL                         |
| fbs       | Fasting Blood Sugar     | Binary      | 0 = ≤120 mg/dL, 1 = >120 mg/dL         |
| restecg   | Resting ECG             | Categorical | 0=Normal, 1=ST-T Abnormal, 2=LV Hypertrophy |
| thalach   | Max Heart Rate          | Integer     | 71 – 202 bpm                            |
| exang     | Exercise Angina         | Binary      | 0 = No, 1 = Yes                         |
| oldpeak   | ST Depression           | Float       | 0.0 – 6.2 mm                            |
| slope     | ST Slope                | Categorical | 0=Upsloping, 1=Flat, 2=Downsloping      |
| ca        | Major Vessels           | Integer     | 0 – 4                                   |
| thal      | Thalassemia             | Categorical | 0=None, 1=Normal, 2=Fixed, 3=Reversible |
| target    | Disease Label           | Binary      | 0 = No Disease, 1 = Disease Present     |

---

## ⚙️ Requirements & Installation

### Step 1 — Make sure Python is installed
    python --version
    # Should show Python 3.7 or higher

### Step 2 — Install all required libraries
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn

### Step 3 — Verify Streamlit installation
    streamlit --version
    # Should show something like: Streamlit 1.x.x

### Full dependency list

| Library       | Purpose                              |
|---------------|--------------------------------------|
| streamlit     | Web dashboard framework              |
| pandas        | Data loading and manipulation        |
| numpy         | Numerical computations               |
| matplotlib    | Chart and plot generation            |
| seaborn       | Statistical visualizations           |
| scikit-learn  | ML models, preprocessing, metrics   |

---

## ▶️ How to Run

### Step 1 — Navigate to your project folder
    cd path/to/your/folder

### Step 2 — Run the dashboard
    streamlit run heart_dashboard.py

### Step 3 — Open in browser
The dashboard will automatically open at:
    http://localhost:8501

If it does not open automatically, copy and paste that URL into your browser manually.

### To stop the server
Press Ctrl + C in the terminal.

---

## 🖥️ Dashboard Pages

### 🏠 Overview
- Total patient count, feature count, disease/no-disease split
- Model accuracy summary cards for all 3 models
- Live accuracy progress bars
- Dataset preview (first 10 rows)

### 📊 Data Analysis
**Distributions Tab**
- Target variable distribution (disease vs no disease)
- Age distribution histogram
- Cholesterol distribution histogram
- Max Heart Rate distribution histogram
- Sex vs Disease Status grouped bar chart
- Chest Pain Type vs Disease grouped bar chart
- ST Slope vs Disease grouped bar chart
- Major Vessels (ca) vs Disease grouped bar chart

**Correlation Tab**
- Full feature correlation heatmap (lower triangle)
- Feature–target correlation bar chart (positive/negative)
- Feature–Target Direction Summary table (all 13 features with strength and insight)
- ST Depression clinical thresholds table (Normal / Borderline / Significant / Severe)
- Mean oldpeak metric cards (overall / disease / no-disease)
- ST Slope clinical interpretation table

**Feature Deep Dive Tab**
- Select any feature and explore its distribution by disease group
- Histogram overlay (disease vs no disease)
- Boxplot by target group
- Descriptive statistics table

### 🤖 Model Performance
- Accuracy comparison bar chart (KNN vs LR vs Naive Bayes)
- 80% baseline reference line
- Confusion matrices for all 3 models (heatmap)
- Detailed classification reports (precision, recall, F1-score) in tabbed view

### 🔮 Live Prediction
- Input form for all 13 clinical features
- Real-time prediction from all 3 models simultaneously
- Disease probability bar per model
- Majority vote final verdict (≥2/3 models agree)
- Average disease probability across all models

### 📋 Clinical Summary
- 8 key clinical observations from dataset analysis
- Top 6 most predictive features with clinical explanations
- ST Depression (oldpeak) final interpretation note
- 5 known dataset limitations and caveats
- Academic disclaimer

---

## 🤖 Machine Learning Models

### K-Nearest Neighbors (KNN)
- Classifies a patient based on the K=5 most similar patients in training data
- Distance-based — sensitive to feature scaling (StandardScaler applied)
- Non-parametric, no assumptions about data distribution

### Logistic Regression
- Linear model that estimates the probability of disease using a sigmoid function
- max_iter=1000, random_state=42
- Interpretable coefficients — good baseline classifier

### Naive Bayes (Gaussian)
- Probabilistic classifier assuming feature independence (Bayes theorem)
- Assumes Gaussian distribution for continuous features
- Fast training, works well even with limited data

### Training Setup
- Train/Test Split: 80% train / 20% test (random_state=42)
- Preprocessing: StandardScaler (fit on train, transform on both)
- All models trained on the same split for fair comparison

---

## 🔬 Key Clinical Findings

- Dataset is near-balanced (51.3% disease) — no aggressive resampling needed
- Males = 69.6% of dataset, yet females show proportionally higher disease rate
- Asymptomatic chest pain (cp=0) at 48.5% — the silent CAD paradox
- Disease patients average 139 bpm max HR vs 158 bpm in healthy patients
- Mean ST depression: 0.57 mm (disease) vs 1.60 mm (no disease) — inverse coding
- Reversible thalassemia defect (thal=3) in 40% of patients — strongest thal predictor
- Only 14.9% of patients have elevated fasting blood sugar (>120 mg/dL)

---

## ⚠️ Disclaimer

This dashboard is developed as an academic and educational machine learning project.
It is NOT intended to be used as a clinical diagnostic tool or as a substitute for
professional medical advice. Always consult a qualified cardiologist or medical
professional for any health-related decisions.

---

## 📚 References

1. Detrano R. et al. (1989). International application of a new probability algorithm
   for diagnosis of coronary artery disease. American Journal of Cardiology, 64(5).

2. UCI ML Repository — Heart Disease Dataset:
   https://archive.ics.uci.edu/ml/datasets/Heart+Disease

3. Kaggle Dataset (johnsmith88):
   https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

4. WHO Cardiovascular Disease Fact Sheet (2021):
   https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

5. Pedregosa F. et al. (2011). Scikit-learn: Machine Learning in Python.
   Journal of Machine Learning Research, 12, 2825–2830.

6. Gibbons R.J. et al. (2002). ACC/AHA Guideline Update for Exercise Testing.
   Circulation, 106(14), 1883–1892.

---

Dataset: UCI Heart Disease (Cleveland) via Kaggle — johnsmith88/heart-disease-dataset
Generated: March 2026 | C.V. Raman Global University, Bhubaneswar, Odisha
