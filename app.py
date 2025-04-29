import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
with open("feature_columns.txt", "r") as f:
    feature_names = f.read().split(",")

st.title("💓 Heart Disease Prediction App")
st.markdown("Provide the following details to check your heart disease risk:")

# Form Inputs
age = st.slider("Age", 18, 100, 45)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
bp = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Rest ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
thalach = st.slider("Max Heart Rate Achieved", 60, 210, 150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST", ["Upsloping", "Flat", "Downsloping"])

# Basic encodings
data = {
    "Age": age,
    "RestingBP": bp,
    "Cholesterol": chol,
    "FastingBS": 1 if fbs == "Yes" else 0,
    "MaxHR": thalach,
    "Oldpeak": oldpeak,
}

# One-hot encodings
data[f"Sex_{sex}"] = 1
data[f"ChestPainType_{chest_pain}"] = 1
data[f"RestingECG_{restecg}"] = 1
data[f"ExerciseAngina_{exang}"] = 1
data[f"Slope_{slope}"] = 1

# Prepare final input array
final_input = np.zeros(len(feature_names))
for idx, col in enumerate(feature_names):
    final_input[idx] = data.get(col, 0)

# Scale and Predict
input_scaled = scaler.transform([final_input])
prediction = model.predict(input_scaled)[0]
confidence = model.predict_proba(input_scaled)[0][1]

# Output
if prediction == 1:
    st.error(f"⚠️ High Risk of Heart Disease (Confidence: {confidence:.2f})")
else:
    st.success(f"✅ Low Risk of Heart Disease (Confidence: {1-confidence:.2f})")
