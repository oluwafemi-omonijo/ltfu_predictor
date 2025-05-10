import streamlit as st
import pandas as pd
import numpy as np
import joblib

#Page configuration
st.set_page_config(page_title="HIV LTFU Risk Predictor", layout="centered")

#load trained model
model = joblib.load("random_forest_model.pkl")

#App title
# App Title
st.title("🩺 HIV Loss to Follow-Up (LTFU) Risk Predictor")
st.markdown("Estimate a patient's risk of being lost to follow-up based on clinical and visit history.")

# User Inputs
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
cd4 = st.number_input("CD4 Count", min_value=0, max_value=5500, value=350)
missed_visits = st.slider("Number of Missed Clinic Visits", 0, 5, 1)
visit_gap = st.slider("Days Since Last Clinic Visit", 0, 180, 30)

# Process categorical input
sex_binary = 0 if sex == "Male" else 1

# Dynamic adherence score (computed from inputs)
adherence_score = (
    (1 - (missed_visits / 5)) * 0.4 +
    (1 - (visit_gap / 180)) * 0.4 +
    np.random.uniform(0.85, 1.0) * 0.2
)

# Feature engineering
age_x_missed = age * missed_visits
adherence_x_gap = adherence_score * visit_gap

# Create input DataFrame
input_df = pd.DataFrame([[
    age, sex_binary, adherence_score, cd4, visit_gap,
    missed_visits, age_x_missed, adherence_x_gap
]], columns=[
    'age', 'sex', 'adherence_score', 'cd4_count', 'last_visit_gap_days',
    'missed_visits', 'age_x_missed', 'adherence_x_gap'
])

# Prediction
if st.button("Predict LTFU Risk"):
    prediction_proba = model.predict_proba(input_df)[0][1]
    st.subheader(f"🔎 Estimated Risk of LTFU: **{prediction_proba:.2%}**")

    if prediction_proba > 0.5:
        st.error("⚠️ High risk: Consider follow-up interventions.")
    else:
        st.success("✅ Low risk: Patient likely to stay in care.")