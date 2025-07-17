import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Health Insurance Claim Predictor", layout="centered")
st.title("🏥 Health Insurance Claim Approval Predictor")
st.markdown("Enter claim and patient details to predict if the health insurance claim will be **Approved** or **Rejected**.")

# --- Input Fields ---
age = st.slider("Patient Age", 0, 100, 30)
amount = st.number_input("Claim Amount (₹)", min_value=0, max_value=100000, value=5000, step=500)
gender = st.selectbox("Patient Gender", ["Male", "Female"])
diagnosis = st.selectbox("Diagnosis Code", ["D1", "D2", "D3"])  # match with training dummy
procedure = st.selectbox("Procedure Code", ["P1", "P2", "P3"])  # match with training dummy

# --- Feature Encoding (same as training dataset one-hot) ---
input_df = pd.DataFrame({
    'PatientAge': [age],
    'ClaimAmount': [amount],
    'PatientGender_Female': [1 if gender == "Female" else 0],
    'DiagnosisCode_D2': [1 if diagnosis == "D2" else 0],
    'DiagnosisCode_D3': [1 if diagnosis == "D3" else 0],
    'ProcedureCode_P2': [1 if procedure == "P2" else 0],
    'ProcedureCode_P3': [1 if procedure == "P3" else 0],
})

# --- Predict Button ---
if st.button("🧠 Predict"):
    prediction = model.predict(input_df)[0]
    st.success("✅ Claim Approved" if prediction == 1 else "❌ Claim Rejected")
