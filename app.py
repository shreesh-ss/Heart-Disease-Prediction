import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

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

# Encoding
sex = 1 if sex == "Male" else 0
chest_pain_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
fbs = 1 if fbs == "Yes" else 0
restecg_map = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
exang = 1 if exang == "Yes" else 0
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

input_data = np.array([[age, sex, chest_pain_map[chest_pain], bp, chol, fbs,
                        restecg_map[restecg], thalach, exang, oldpeak, slope_map[slope]]])

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]
confidence = model.predict_proba(input_scaled)[0][1]

# Output
if prediction == 1:
    st.error(f"⚠️ High Risk of Heart Disease (Confidence: {confidence:.2f})")
else:
    st.success(f"✅ Low Risk of Heart Disease (Confidence: {1-confidence:.2f})")
