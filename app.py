import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler  # Import StandardScaler

# Load the trained model
with open("stroke_risk_model.pkl", "rb") as file:
    model = pickle.load(file)

# Web App Title
st.title("Simplified Stroke Risk Prediction")

# Collect User Input
st.header("Enter Your Details:")
age = st.slider("Age:", 0, 100, 25)
heart_disease = st.radio("Do you have heart disease?", ["Yes", "No"])
work_status = st.radio("Do you work?", ["Yes", "No"])
hypertension = st.radio("Do you have hypertension?", ["Yes", "No"])
avg_glucose_level = st.slider("Average Glucose Level:", 0.0, 300.0, 100.0)
bmi = st.slider("BMI:", 0.0, 60.0, 25.0)
smoking_status = st.selectbox("Smoking Status:", ["never smoked", "formerly smoked", "smokes"])
ever_married = st.radio("Have you ever been married?", ["Yes", "No"])
previous_stroke = st.radio("Have you had a stroke before?", ["Yes", "No"])

# Encode User Input
def encode_input():
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0
    work_status_encoded = 1 if work_status == "Yes" else 0
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

    # Combine all inputs
    features = [
        age, heart_disease_encoded, work_status_encoded, hypertension_encoded, 
        avg_glucose_level, bmi, ever_married_encoded, previous_stroke_encoded
    ] + smoking_status_encoded

    return np.array(features).reshape(1, -1)

# Predict Stroke Risk
if st.button("Predict Stroke Risk"):
    user_input = encode_input()

    # Ensure input is scaled correctly
    scaler = StandardScaler()
    user_input[:, [0, 4, 5]] = scaler.fit_transform(user_input[:, [0, 4, 5]])

    # Debug: Print encoded input
    st.write("Encoded User Input:", user_input)

    # Make predictions
    risk_score = model.predict_proba(user_input)[0][1]

    # Display Risk Score
    st.write(f"Risk Score: {risk_score:.2f}")

    # Display Risk Category
    if risk_score < 0.3:
        st.success("Low Risk")
    elif risk_score < 0.6:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")

