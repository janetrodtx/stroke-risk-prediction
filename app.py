import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler  # Added this line


# Load the trained model
with open("stroke_risk_model.pkl", "rb") as file:
    model = pickle.load(file)

# Web App Title
st.title("Stroke Risk Prediction")

# Collect User Input
st.header("Enter Your Details:")
gender = st.selectbox("Gender:", ["Male", "Female"])
age = st.slider("Age:", 0, 100, 25)
hypertension = st.radio("Do you have hypertension?", ["Yes", "No"])
heart_disease = st.radio("Do you have heart disease?", ["Yes", "No"])
ever_married = st.radio("Have you ever been married?", ["Yes", "No"])
work_type = st.selectbox("Work Type:", ["Private", "Self-employed", "Never_worked", "children"])
Residence_type = st.selectbox("Residence Type:", ["Urban", "Rural"])
avg_glucose_level = st.slider("Average Glucose Level:", 0.0, 300.0, 100.0)
bmi = st.slider("BMI:", 0.0, 60.0, 25.0)
smoking_status = st.selectbox("Smoking Status:", ["never smoked", "formerly smoked", "smokes"])

# Encode User Input
def encode_input():
    gender_encoded = 1 if gender == "Male" else 0
    hypertension_encoded = 1 if hypertension == "Yes" else 0
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0
    ever_married_encoded = 1 if ever_married == "Yes" else 0
    Residence_type_encoded = 1 if Residence_type == "Urban" else 0

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
        gender_encoded, age, hypertension_encoded, heart_disease_encoded, 
        ever_married_encoded, Residence_type_encoded, avg_glucose_level, bmi
    ] + work_type_encoded + smoking_status_encoded

    return np.array(features).reshape(1, -1)

# Predict Stroke Risk
# Ensure input is scaled correctly
scaler = StandardScaler()
numerical_features = ['age', 'avg_glucose_level', 'bmi']
user_input[:, [1, 6, 7]] = scaler.fit_transform(user_input[:, [1, 6, 7]])

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

