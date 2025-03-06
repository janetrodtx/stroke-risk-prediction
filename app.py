import streamlit as st
import pickle
import numpy as np
import pandas as pd
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

# Define the encode_input function
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

# Encode User Input
user_input = encode_input()  # Define user_input here

# Create a DataFrame to align features
feature_columns = [
    'age', 'heart_disease', 'work_status', 'hypertension', 
    'avg_glucose_level', 'bmi', 'ever_married', 'previous_stroke',
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 
    'smoking_status_smokes', 'work_type_Never_worked', 'work_type_Private', 
    'work_type_Self-employed', 'work_type_children'
]

user_input_df = pd.DataFrame(user_input, columns=[
    'age', 'heart_disease', 'work_status', 'hypertension', 
    'avg_glucose_level', 'bmi', 'ever_married', 'previous_stroke',
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 
    'smoking_status_smokes'
])

# Add missing columns with 0 values
for col in feature_columns:
    if col not in user_input_df.columns:
        user_input_df[col] = 0

# Ensure the columns are in the correct order
user_input_df = user_input_df[feature_columns]

# Convert to numpy array for prediction
user_input = user_input_df.values

# Ensure input is scaled correctly
# Make predictions
risk_score = model.predict_proba(user_input)[0][1]  # Define risk_score here

# Display Risk Score
st.write(f"Risk Score: {risk_score:.2f}")

# Adjusted Threshold for Risk Levels
threshold = 0.3  # Adjust this value as needed
if risk_score > threshold:
    st.error("High Risk")
elif risk_score > 0.15:
    st.warning("Medium Risk")
else:
    st.success("Low Risk")

