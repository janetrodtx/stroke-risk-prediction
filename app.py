import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open("stroke_risk_model.pkl", "rb") as file:
    model = pickle.load(file)

# Web App Title
st.title("💖 Stroke Risk Prediction App")
st.markdown("### Simplified and Personalized Stroke Risk Assessment")

# Collect User Input
st.sidebar.header("Enter Your Details:")
age = st.sidebar.slider("Age:", 0, 100, 25)
heart_disease = st.sidebar.radio("Do you have heart disease?", ["Yes", "No"])
hypertension = st.sidebar.radio("Do you have hypertension?", ["Yes", "No"])
avg_glucose_level = st.sidebar.slider("Average Glucose Level:", 0.0, 300.0, 100.0)
bmi = st.sidebar.slider("BMI:", 0.0, 60.0, 25.0)
smoking_status = st.sidebar.selectbox("Smoking Status:", ["never smoked", "formerly smoked", "smokes"])
ever_married = st.sidebar.radio("Have you ever been married?", ["Yes", "No"])
previous_stroke = st.sidebar.radio("Have you had a stroke before?", ["Yes", "No"])
work_type = st.sidebar.selectbox("Work Type:", ["Never_worked", "Private", "Self-employed", "children"])

# Encode user input
def encode_input():
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0
    hypertension_encoded = 1 if hypertension == "Yes" else 0
    ever_married_encoded = 1 if ever_married == "Yes" else 0
    previous_stroke_encoded = 1 if previous_stroke == "Yes" else 0

    # One-hot encoding for smoking_status
    smoking_status_encoded = [0, 0, 0]
    if smoking_status == "formerly smoked":
        smoking_status_encoded[0] = 1
    elif smoking_status == "never smoked":
        smoking_status_encoded[1] = 1
    elif smoking_status == "smokes":
        smoking_status_encoded[2] = 1

    # One-hot encoding for work_type
    work_type_encoded = [0, 0, 0, 0]
    if work_type == "Never_worked":
        work_type_encoded[0] = 1
    elif work_type == "Private":
        work_type_encoded[1] = 1
    elif work_type == "Self-employed":
        work_type_encoded[2] = 1
    elif work_type == "children":
        work_type_encoded[3] = 1

    # Combine all inputs
    features = [
        age, heart_disease_encoded, hypertension_encoded,
        avg_glucose_level, bmi, ever_married_encoded, previous_stroke_encoded
    ] + smoking_status_encoded + work_type_encoded

    return np.array(features).reshape(1, -1)

# Encode and predict
user_input = encode_input()
risk_score = model.predict_proba(user_input)[0][1]

# Display risk score as a percentage
risk_percentage = risk_score * 100
st.write(f"### 🩺 Your Risk Score: **{risk_percentage:.2f}%**")

# Progress bar based on risk score
st.progress(risk_score)

# Determine risk category
if risk_score < 0.3:
    st.success("You are at **Low Risk** for stroke. Keep up the healthy habits!")
    st.write("✅ **Recommendations:**")
    st.write("- Maintain regular physical activity (150 mins/week).")
    st.write("- Follow a balanced diet (low sodium, rich in fruits and vegetables).")
    st.write("- Keep regular health checkups for blood pressure and cholesterol.")
elif risk_score < 0.6:
    st.warning("You are at **Medium Risk** for stroke. Consider making lifestyle improvements.")
    st.write("⚠️ **Recommendations:**")
    st.write("- Increase physical activity: Aim for 30 minutes of exercise, 5 days a week.")
    st.write("- Quit smoking: Seek support or resources.")
    st.write("- Monitor cholesterol and blood pressure regularly.")
else:
    st.error("You are at **High Risk** for stroke. Take immediate action.")
    st.write("🚨 **Recommendations:**")
    st.write("- Consult a healthcare provider immediately.")
    st.write("- Implement lifestyle changes: Quit smoking, reduce salt intake.")
    st.write("- Increase physical activity to manage weight and cardiovascular health.")
    st.write("- Consider medication for blood pressure and cholesterol if advised.")

# Footer
st.markdown("---")
st.markdown("📋 **Note:** This prediction is based on the data provided and is not a substitute for professional medical advice.")

<<<<<<< HEAD

=======
>>>>>>> 4b2c18753fa624828bfade9c8e30147335a1efc5

