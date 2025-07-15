# 🩺 Health Insurance Claim Prediction using XGBoost

This project aims to predict whether a health insurance claim will be approved or not, using machine learning techniques and SHAP-based model interpretability.

## 📌 Problem Statement

Insurance companies process thousands of health insurance claims. Identifying potentially fraudulent or rejected claims early can reduce losses and improve operational efficiency.

This project predicts whether a claim will be **approved (1)** or **rejected (0)** based on various features like patient age, diagnosis code, claim amount, procedure details, etc.

## 📂 Dataset

- **File:** `enhanced_health_insurance_claims.csv`
- **Columns Used:**
  - `ClaimID`, `ClaimDate`, `ClaimAmount`
  - `PatientAge`, `PatientGender`, `DiagnosisCode`, `ProcedureCode`
  - `ClaimType`, `ClaimSubmissionMethod`, etc.
  - `ClaimApproved` (Target variable: 1 = Approved, 0 = Rejected)

---

## 🛠️ Tech Stack

| Tool            | Purpose                              |
|-----------------|--------------------------------------|
| Python          | Core Programming Language            |
| Pandas, NumPy   | Data Processing                      |
| Matplotlib, Seaborn | Data Visualization             |
| XGBoost         | ML Model (Gradient Boosted Trees)    |
| Scikit-learn    | Train-test split, metrics            |
| SHAP            | Model Interpretability               |

---

## 🧠 Workflow

1. **Load Dataset**
2. **Handle Missing Values**
3. **Convert Dates & Feature Engineering**
4. **Encode Categorical Variables**
5. **Split Dataset (Train/Test)**
6. **Train XGBoost Classifier**
7. **Model Evaluation (Accuracy, Report)**
8. **SHAP Analysis (Explainability)**

---

## 📊 Results

- **Model Used:** XGBoost Classifier
- **Accuracy:** ~`46.33%` 
- **Top Features:** `ClaimAmount`, `PatientAge`, `DiagnosisCode_X`, etc.
- **Explainability:** SHAP summary plot included

---

## 📈 SHAP Summary Plot

<img src="shap_summary_plot.png" alt="SHAP Summary Plot" width="600">

---

## 📁 Project Structure

