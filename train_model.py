import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Handle missing values
df['bmi'].fillna(df['bmi'].mean(), inplace=True)
df['avg_glucose_level'].fillna(df['avg_glucose_level'].mean(), inplace=True)

# Drop rows with missing target values (if any)
df = df.dropna(subset=['stroke'])

# Encode categorical features
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})

# Encode previous stroke feature
df['previous_stroke'] = df['stroke'].apply(lambda x: 1 if x > 0 else 0)

# One-hot encoding for smoking status and work type
df = pd.get_dummies(df, columns=['smoking_status', 'work_type'], drop_first=True)

# Define feature columns
feature_columns = [
    "gender", "age", "hypertension", "heart_disease", "previous_stroke", "ever_married", "Residence_type",
    "avg_glucose_level", "bmi",
    "smoking_status_formerly smoked", "smoking_status_never smoked", "smoking_status_smokes",
    "work_type_Never_worked", "work_type_Private", "work_type_Self-employed", "work_type_children"
]

# Prepare data
X = df[feature_columns]  # Features
y = df['stroke']          # Target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train StandardScaler on training data
scaler = StandardScaler()
scaler.fit(X_train)

# Save the updated scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# Standardize training data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the trained model
with open("stroke_risk_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model and Scaler trained and saved successfully!")
