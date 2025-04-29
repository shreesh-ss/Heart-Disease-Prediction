# 💓 Heart Disease Prediction App

A high-accuracy machine learning web app that predicts the risk of heart disease using clinical inputs.  
Built with **XGBoost**, **Scikit-learn**, and deployed using **Streamlit**.

## 🚀 Features
- Predict heart disease with high confidence
- Uses XGBoost model with GridSearchCV tuning
- Interactive UI with sliders and dropdowns
- Live deployed using Streamlit Cloud

## 📂 How to Run

1. Clone the repo
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the app:
```bash
streamlit run streamlit_app.py
```

## 📦 Files Included
- `streamlit_app.py` – Streamlit frontend
- `requirements.txt` – Required libraries
- `heart_model.pkl` – Trained XGBoost model *(add this)*
- `scaler.pkl` – StandardScaler object *(add this)*

## 📸 Thumbnail
![App Thumbnail](thumbnail.png)

## 📜 License
This project is licensed under the MIT License.
