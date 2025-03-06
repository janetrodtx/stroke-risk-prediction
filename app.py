import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler  # Import StandardScaler

# Load the trained model
with open("stroke_risk_model.pkl", "rb") as file:
    model = pickle.load(file)

# Web App Title
st.title("Simplified Stroke Risk Prediction")

# Encode User Input
user_input = encode_input()  # Define user_input here

# Create a DataFrame to align features
import pandas as pd

# Create a DataFrame for user input with the original 15 columns
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
scaler = StandardScaler()
user_input[:, [0, 4, 5]] = scaler.fit_transform(user_input[:, [0, 4, 5]])

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
