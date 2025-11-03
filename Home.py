# C:\Users\hp\Documents\DS_internship\emi_prediction_app\Home.py
import streamlit as st
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import sklearn.compose._column_transformer



st.set_page_config(
    page_title="EMIPredict AI - Financial Risk Assessment",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 EMIPredict AI: Intelligent Financial Risk Assessment Platform")

st.markdown("""
Welcome to the **EMIPredict AI Platform**. This multi-page application hosts dual machine learning models 
for comprehensive financial risk assessment:
""")

st.subheader("Project Goals & Components:")
st.markdown("""
1.  **Classification Model (EMI Eligibility):** Predicts the loan applicant's risk level (`Eligible`, `High_Risk`, or `Not_Eligible`).
2.  **Regression Model (Max Monthly EMI):** Predicts the maximum monthly EMI the applicant can realistically afford.
3.  **MLflow Integration:** All models are tracked and versioned using MLflow for monitoring.
4.  **Streamlit UI:** Provides real-time prediction, data exploration, and model monitoring dashboards.
""")

st.subheader("Navigate the Application:")
st.info("""
👈 Use the sidebar to switch between pages:
- **Prediction:** Enter customer data for real-time risk assessment.
- **Data Insights:** Explore the underlying financial dataset.
- **MLflow Monitor:** View model performance tracking and version control.
- **Admin Panel:** Mock interface for data management operations (CRUD).
""")

st.subheader("Best Performing Models (Selected from Step 5)")
st.markdown("""
| Model Type | Best Model | Key Metric | Score |
|:-----------|:-----------|:-----------|:------|
| Classification | **XGBoost Classifier** | F1 Score | 0.9678 |
| Regression | **XGBoost Regressor** | R-squared | 0.9763 |
""")