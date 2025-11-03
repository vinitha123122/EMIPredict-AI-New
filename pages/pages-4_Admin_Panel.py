import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("⚙️ Data Management / Admin Panel")
st.subheader("Simulated CRUD Operations for Financial Data")

# --- 1. Data Loading ---
FILE_NAME = 'emi_prediction_dataset.csv'
try:
    # FIX: Added low_memory=False to suppress the DtypeWarning
    df = pd.read_csv(FILE_NAME, low_memory=False) 
    df.index.name = 'Record_ID'
    df.reset_index(inplace=True) 
    
    def clean_and_convert(series):
        return pd.to_numeric(series.astype(str).str.replace(',', '').str.replace('$', '').str.strip(), errors='coerce')
    
    df['monthly_salary'] = clean_and_convert(df['monthly_salary'])
    df['max_monthly_emi'] = clean_and_convert(df['max_monthly_emi'])

except FileNotFoundError:
    st.error("Error: 'emi_prediction_dataset.csv' not found. Cannot load data for the Admin Panel.")
    df = pd.DataFrame()
except Exception as e:
    st.error(f"Error loading data: {e}")
    df = pd.DataFrame()

if not df.empty:
    
    st.info("""
    This interface simulates basic **C**reate, **R**ead, **U**pdate, and **D**elete (**CRUD**) operations. 
    In a real system, these actions would interact with a secure database, ensuring data quality for continuous model retraining.
    """)
    
    st.markdown("---")

    # --- 2. READ Operation ---
    st.header("1. Read (View) Operations")
    st.dataframe(df.head(200), width='stretch')
    st.write(f"**Total Records in Dataset:** {len(df)}")
    
    st.markdown("---")

    # --- 3. CREATE/UPDATE Operation (Simulated Form) ---
    st.header("2. Create/Update Record (Simulated)")

    with st.form("admin_form"):
        # Max ID for validation/defaulting input
        max_id = df['Record_ID'].max() if 'Record_ID' in df.columns else 0
        
        # User input for which record to interact with
        record_id = st.number_input(
            "Enter **Record ID** to Update, or **0** for a New Record (CREATE)", 
            0, max_id, 0
        )
        
        # Determine if we are loading existing data for UPDATE
        initial_data = df[df['Record_ID'] == record_id].iloc[0] if record_id != 0 and record_id in df['Record_ID'].values else {}
        
        st.subheader("Key Financial Fields")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 18, 90, int(initial_data.get('age', 30)), key="admin_age")
            monthly_salary = st.number_input("Monthly Salary ($)", 5000.0, 500000.0, float(initial_data.get('monthly_salary', 50000.0)))
        with col2:
            eligibility_options = ["Eligible", "High_Risk", "Not_Eligible"]
            initial_eligibility = initial_data.get('emi_eligibility', "Eligible").replace(' ', '_')
            emi_eligibility = st.selectbox("EMI Eligibility", eligibility_options, index=eligibility_options.index(initial_eligibility) if initial_eligibility in eligibility_options else 0, key="admin_eligibility")
            max_monthly_emi = st.number_input("Max Monthly EMI ($)", 500.0, 50000.0, float(initial_data.get('max_monthly_emi', 5000.0)))

        submitted = st.form_submit_button("Simulate Save/Update to Database")

        if submitted:
            if record_id == 0:
                # CREATE Simulation
                new_id = max_id + 1
                st.success(f"✅ **CREATE Operation Simulated (Record ID: {new_id}):** New record would be added to the database. This new data is crucial for future model retraining.")
            else:
                # UPDATE Simulation
                if record_id in df['Record_ID'].values:
                    st.warning(f"🔄 **UPDATE Operation Simulated (Record ID: {record_id}):** Record data has been revised (e.g., Eligibility changed to {emi_eligibility}). Database record would be updated.")
                else:
                    st.error(f"Record ID {record_id} not found.")
    
    st.markdown("---")

    # --- 4. DELETE Operation (Simulated) ---
    st.header("3. Delete Record (Simulated)")
    
    col_del, col_btn = st.columns([2, 1])
    with col_del:
        delete_id = st.number_input("Enter **Record ID** to Delete", 0, max_id, 0, key="delete_input")
    with col_btn:
        # Add a small buffer space to align the button
        st.text("")
        st.text("")
        delete_button = st.button("Simulate Delete Record")

    if delete_button and delete_id != 0:
        if delete_id in df['Record_ID'].values:
            st.error(f"🗑️ **DELETE Operation Simulated (Record ID: {delete_id}):** The record would be permanently removed from the database.")
        else:
            st.error(f"Record ID {delete_id} not found.")