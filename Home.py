# C:\Users\hp\Documents\DS_internship\emi_prediction_app\Home.py
import streamlit as st
import pandas as pd
import os
import joblib # Ensure joblib is imported at the top

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import sklearn.compose._column_transformer # Import to ensure it's available if needed elsewhere


# --- Caching Functions (Critical for Streamlit Cloud) ---

@st.cache_resource
def load_models(model_name):
    # This assumes your models are named best_emi_classifier_pipeline.pkl and best_emi_regressor_pipeline.pkl
    try:
        model_path = os.path.join(os.path.dirname(__file__), model_name)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None

# Load the models using caching
CLASSIFIER = load_models('best_emi_classifier_pipeline.pkl')
REGRESSOR = load_models('best_emi_regressor_pipeline.pkl')

# --- CRITICAL FIX: SHARE MODELS VIA SESSION STATE ---
if CLASSIFIER is not None:
    st.session_state['CLASSIFIER'] = CLASSIFIER
if REGRESSOR is not None:
    st.session_state['REGRESSOR'] = REGRESSOR
# ----------------------------------------------------

if CLASSIFIER is None or REGRESSOR is None:
    st.error("Application cannot run because one or both required model files failed to load. Please check file paths.")
    st.stop()
    
# --- Streamlit UI Start ---

st.set_page_config(
    page_title="EMIPredict AI - Financial Risk Assessment",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 EMIPredict AI: Intelligent Financial Risk Assessment Platform")


# --- Main Content ---

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