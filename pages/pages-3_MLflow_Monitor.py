import streamlit as st
import os

st.set_page_config(layout="wide")
st.title("👁️ MLflow Experiment Monitoring Dashboard")
st.subheader("Model Performance Comparison and Version Tracking")

# --- MLflow Configuration ---
mlflow_url = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

st.markdown("""
This page provides a centralized view of all model training runs and final model versions. 
All hyperparameters, metrics, and artifacts are logged here.
""")

st.header("⚠️ Prerequisite: Run the MLflow Tracking Server")

st.markdown("""
To view the live dashboard, you **must** run the MLflow server in a separate terminal window.
""")

st.subheader("1. Start the MLflow Server")
st.code("mlflow ui")

# Using a RAW STRING (r"""...""") here to prevent the unicode escape error
st.markdown(r"""
**Execution Steps:**
1.  **Open your terminal** and navigate to your project root:
    ```bash
    cd C:\Users\hp\Documents\DS_internship\emi_prediction_app 
    ```
    *(The 'r' prefix above ensures the backslashes in the path are read correctly.)*
2.  **Execute the command** `mlflow ui` in that terminal.
3.  Once running, access the dashboard via the link below.
""")

st.markdown(f"**MLflow Dashboard Link (Open in Browser):** **[{mlflow_url}]({mlflow_url})**")

st.write("---")

st.header("Key Features in the MLflow UI")

st.info("""
**In the MLflow UI, you can perform the following critical MLOps tasks:**

1.  **Experiment Comparison:** Compare the performance of all six models side-by-side using metrics like **F1 Score** and **R2/RMSE**.
2.  **Model Registry:** View the best models as registered artifacts, moving them between stages like **Staging** and **Production** for formal deployment.
""")

st.write("---")

# --- Embedded View (for local demo) ---
st.subheader("Embedded MLflow View (Local Demonstration)")
st.caption("If running locally, this iframe provides a direct view of the MLflow UI. (Requires MLflow server to be running)")

try:
    st.components.v1.iframe(mlflow_url, height=750)
except Exception:
    st.warning("Could not embed MLflow UI. Please open the link above directly in your browser.")