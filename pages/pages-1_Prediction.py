import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost # Required to load XGBoost models

# --- FIX for AttributeError: Can't get attribute '_RemainderColsList' ---
# Explicitly import all components needed by the saved pipeline structure.
# This forces the Python interpreter to recognize internal classes used 
# during serialization (e.g., ColumnTransformer components).
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
# The specific module that contains the hidden class:
import sklearn.compose._column_transformer 
# ---------------------------------------------------------------------

# --- 1. Load Models and Setup ---
try:
    CLS_PIPELINE = joblib.load('best_emi_classifier_pipeline.pkl')
    REG_PIPELINE = joblib.load('best_emi_regressor_pipeline.pkl')
    st.sidebar.success("Models loaded successfully.")
except FileNotFoundError:
    st.error("Error: Model files not found. Ensure .pkl files are in the root directory.")
    CLS_PIPELINE = None
    REG_PIPELINE = None
except Exception as e:
    # Catch any remaining load error and display for debugging
    st.error(f"Failed to load model files despite fix. Error: {e}")
    CLS_PIPELINE = None
    REG_PIPELINE = None


# --- 2. Feature Engineering Logic (MUST match Step 4) ---
def apply_feature_engineering(df):
    """Applies the exact feature engineering steps used during model training."""
    epsilon = 1e-6

    # Base expense columns
    expense_only_cols = ['monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 
                         'groceries_utilities', 'other_monthly_expenses']
    
    # Calculate Engineered Features
    total_debt_and_emi = df[expense_only_cols].sum(axis=1) + df['current_emi_amount']
    df['DTI_ratio'] = total_debt_and_emi / (df['monthly_salary'] + epsilon)
    df['LTI_ratio'] = df['requested_amount'] / (df['monthly_salary'] + epsilon)
    df['ETI_ratio'] = df[expense_only_cols].sum(axis=1) / (df['monthly_salary'] + epsilon)
    df['liquidity_to_loan_ratio'] = df['bank_balance'] / (df['requested_amount'] + epsilon)
    df['employment_stability_score'] = df['years_of_employment'] / (df['age'] + epsilon)
    df['bank_health_score'] = df['bank_balance'] / (df['monthly_salary'] + epsilon)

    # Interaction features
    df['capped_DTI_safety'] = 1 - np.minimum(df['DTI_ratio'], 1.0)
    df['credit_debt_interaction'] = df['credit_score'] * df['capped_DTI_safety']
    df['salary_balance_interaction'] = df['monthly_salary'] * df['bank_balance']
    df.drop(columns=['capped_DTI_safety'], inplace=True, errors='ignore')
    
    # Define the final feature set expected by the pipeline (X)
    X_final = df.drop(columns=[
        'emi_scenario', 'requested_amount', 'current_emi_amount', 
        'monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 
        'groceries_utilities', 'other_monthly_expenses'
    ])
    
    return X_final

# --- 3. Streamlit UI and Prediction Logic ---
st.title("📈 Real-time Risk Prediction")
st.subheader("Input Customer Financial Details")

if CLS_PIPELINE and REG_PIPELINE:
    with st.form("customer_input_form"):
        # ... [Input fields below] ...
        st.write("---")
        
        # --- Demographic Inputs (Row 1) ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("Age", 18, 90, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
            education = st.selectbox("Education", ["Professional", "Graduate", "High School"])
        with col3:
            family_size = st.number_input("Family Size", 1, 10, 4)
            dependents = st.number_input("Dependents", 0, 5, 2)
        with col4:
            house_type = st.selectbox("House Type", ["Owned", "Rented", "Family"])
            existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
            
        st.write("---")
        
        # --- Income & Loan Request Inputs (Row 2) ---
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            monthly_salary = st.number_input("Monthly Salary ($)", 1000.0, 500000.0, 75000.0, step=1000.0)
            requested_amount = st.number_input("Requested Loan Amount ($)", 1000.0, 1000000.0, 350000.0, step=5000.0)
        with col6:
            employment_type = st.selectbox("Employment Type", ["Private", "Government", "Business"])
            years_of_employment = st.number_input("Years of Employment", 0.0, 40.0, 5.0, step=0.1)
        with col7:
            credit_score = st.number_input("Credit Score", 500, 900, 720)
            bank_balance = st.number_input("Bank Balance ($)", 0.0, 1000000.0, 250000.0, step=1000.0)
        with col8:
            emergency_fund = st.number_input("Emergency Fund ($)", 0.0, 500000.0, 50000.0, step=1000.0)
            requested_tenure = st.number_input("Requested Tenure (Months)", 6, 120, 36)
            company_type = st.selectbox("Company Type", ["MNC", "Mid-size", "Large Indian", "Startup"])

        st.write("---")
        st.subheader("Monthly Expenses and Liabilities")

        # --- Expense Inputs (Row 3) ---
        col9, col10, col11 = st.columns(3)
        with col9:
            monthly_rent = st.number_input("Monthly Rent ($)", 0.0, 50000.0, 5000.0)
            school_fees = st.number_input("School Fees ($)", 0.0, 50000.0, 0.0)
            college_fees = st.number_input("College Fees ($)", 0.0, 50000.0, 0.0)
        with col10:
            travel_expenses = st.number_input("Travel Expenses ($)", 0.0, 10000.0, 2500.0)
            groceries_utilities = st.number_input("Groceries & Utilities ($)", 0.0, 30000.0, 8000.0)
            other_monthly_expenses = st.number_input("Other Expenses ($)", 0.0, 20000.0, 5000.0)
        with col11:
            current_emi_amount = st.number_input("Current EMI Amount ($)", 0.0, 50000.0, 10000.0)
            emi_scenario = st.selectbox("EMI Scenario", ["Personal Loan EMI", "Car Loan EMI", "Home Loan EMI"])

        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        st.subheader("Prediction Results")
        
        # 4. Compile Data
        raw_data = {
            'age': age, 'gender': gender, 'marital_status': marital_status, 'education': education, 
            'monthly_salary': monthly_salary, 'employment_type': employment_type, 
            'years_of_employment': years_of_employment, 'company_type': company_type, 
            'house_type': house_type, 'monthly_rent': monthly_rent, 'family_size': family_size, 
            'dependents': dependents, 'school_fees': school_fees, 'college_fees': college_fees, 
            'travel_expenses': travel_expenses, 'groceries_utilities': groceries_utilities, 
            'other_monthly_expenses': other_monthly_expenses, 'existing_loans': existing_loans, 
            'current_emi_amount': current_emi_amount, 'credit_score': credit_score, 
            'bank_balance': bank_balance, 'emergency_fund': emergency_fund, 'emi_scenario': emi_scenario, 
            'requested_amount': requested_amount, 'requested_tenure': requested_tenure
        }
        
        input_df = pd.DataFrame([raw_data])
        
        # 5. Apply Feature Engineering
        X_pred = apply_feature_engineering(input_df.copy())
        
        # 6. Make Predictions
        cls_pred = CLS_PIPELINE.predict(X_pred)[0]
        cls_proba = CLS_PIPELINE.predict_proba(X_pred)[0]
        predicted_proba = np.max(cls_proba)
        
        log_reg_pred = REG_PIPELINE.predict(X_pred)[0]
        max_emi_pred = np.expm1(log_reg_pred)
        
        # 7. Display Results
        col_cls, col_reg = st.columns(2)
        
        with col_cls:
            if cls_pred == 'Eligible':
                st.success("✅ Loan Eligibility: **ELIGIBLE**", icon="💰")
            elif cls_pred == 'High_Risk':
                st.warning("⚠️ Loan Eligibility: **HIGH RISK**", icon="📉")
            else:
                st.error("❌ Loan Eligibility: **NOT ELIGIBLE**", icon="🛑")
            st.metric(label="Prediction Confidence", value=f"{predicted_proba:.2%}")

        with col_reg:
            st.info("🎯 Maximum Recommended Monthly EMI")
            st.metric(label="Affordable EMI", value=f"${max_emi_pred:,.2f}")

        st.markdown("---")
        st.subheader("Business Recommendation")
        
        if cls_pred == 'Eligible':
            st.markdown(f"**Recommendation:** **Approve** the loan. The customer is low-risk and can afford an EMI of up to **${max_emi_pred:,.2f}**.")
        elif cls_pred == 'High_Risk':
            st.markdown(f"**Recommendation:** **Conditional Approval**. The customer is moderate-risk. Offer a loan with a higher interest rate or shorter tenure, ensuring the EMI is strictly below **${max_emi_pred:,.2f}**.")
        else:
            st.markdown(f"**Recommendation:** **Reject** the application. The customer poses a high risk of default.")