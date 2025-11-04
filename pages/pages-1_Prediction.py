import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib # Critical import for this file
from math import log1p

# --- Streamlit Page Setup ---
st.title("📈 Real-time Risk Prediction")
st.markdown("---")

# --- 1. Access Models from Session State (Robust Method) ---
# The models MUST be loaded and stored in st.session_state in Home.py first.
CLASSIFIER = st.session_state.get('CLASSIFIER')
REGRESSOR = st.session_state.get('REGRESSOR')

if CLASSIFIER is None or REGRESSOR is None:
    st.error("❌ Error: Models were not properly loaded in Home.py. Cannot run prediction.")
    st.markdown("Please go back to the **Home** page to ensure models are loaded correctly.")
    st.stop()
    
# --- Input Fields Start ---
st.subheader("Input Customer Financial Details")

with st.form("prediction_form"):
    
    # --- Row 1: Demographics ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed"])
    
    with col2:
        education = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD", "Other"])
        employment_type = st.selectbox("Employment Type", ["Salaried", "Business Owner", "Self-Employed", "Retired", "Student"])
        years_of_employment = st.number_input("Years of Employment", min_value=0, max_value=60, value=5)
        
    with col3:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=750)
        num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
        
    st.markdown("---")
    st.subheader("Financial Details (USD)")

    # --- Row 2: Income and Assets ---
    col4, col5, col6 = st.columns(3)
    
    with col4:
        monthly_salary = st.number_input("Monthly Salary", min_value=0.0, value=5000.0, step=100.0)
        requested_amount = st.number_input("Requested Loan Amount", min_value=100.0, value=50000.0, step=1000.0)

    with col5:
        bank_balance = st.number_input("Bank Balance (Current Liquidity)", min_value=0.0, value=15000.0, step=100.0)
        current_emi_amount = st.number_input("Existing EMI Amount (if any)", min_value=0.0, value=300.0, step=50.0)
        
    with col6:
        # Placeholder for future features
        pass 

    st.markdown("---")
    st.subheader("Monthly Expenses (USD)")

    # --- Row 3: Expenses ---
    col7, col8, col9 = st.columns(3)
    
    with col7:
        monthly_rent = st.number_input("Monthly Rent/Mortgage", min_value=0.0, value=800.0, step=50.0)
        groceries_utilities = st.number_input("Groceries & Utilities", min_value=0.0, value=500.0, step=50.0)
    
    with col8:
        school_fees = st.number_input("School Fees", min_value=0.0, value=200.0, step=50.0)
        college_fees = st.number_input("College Fees", min_value=0.0, value=0.0, step=50.0)

    with col9:
        travel_expenses = st.number_input("Travel Expenses", min_value=0.0, value=150.0, step=10.0)
        other_monthly_expenses = st.number_input("Other Monthly Expenses", min_value=0.0, value=100.0, step=10.0)

    # --- Submit Button ---
    submitted = st.form_submit_button("Get EMI Prediction")

# --- 3. Prediction Logic ---

if submitted:
    
    # --- 4. Feature Engineering (Must match the training script!) ---
    epsilon = 1e-6 # To prevent division by zero
    
    # Total monthly expenses (excluding current EMI)
    total_non_emi_expenses = monthly_rent + school_fees + college_fees + travel_expenses + groceries_utilities + other_monthly_expenses
    
    # Total debt (including current EMI)
    total_debt = total_non_emi_expenses + current_emi_amount

    # Calculated Features (Ensure denominator uses the raw value)
    DTI_ratio = total_debt / (monthly_salary + epsilon) # Debt-to-Income Ratio
    LTI_ratio = requested_amount / (monthly_salary + epsilon) # Loan-to-Income Ratio
    ETI_ratio = total_non_emi_expenses / (monthly_salary + epsilon) # Expense-to-Income Ratio
    liquidity_to_loan_ratio = bank_balance / (requested_amount + epsilon) # Liquidity to Loan Ratio
    employment_stability_score = years_of_employment / (age + epsilon) # Employment Stability
    bank_health_score = bank_balance / (monthly_salary + epsilon) # Bank Balance to Monthly Salary
    
    # Interaction Features
    capped_DTI_safety = 1 - np.minimum(DTI_ratio, 1.0)
    credit_debt_interaction = credit_score * capped_DTI_safety
    salary_balance_interaction = monthly_salary * bank_balance
    
    # --- 5. Create DataFrame for Prediction ---
    input_data = {
        # Demographics
        'age': age,
        'gender': gender,
        'marital_status': marital_status,
        'education_level': education,
        'employment_type': employment_type,
        'years_of_employment': years_of_employment,
        'num_dependents': num_dependents,
        'credit_score': credit_score,
        
        # Financials (Raw inputs used by the model)
        'monthly_salary': monthly_salary,
        'bank_balance': bank_balance,
        'requested_amount': requested_amount,
        'current_emi_amount': current_emi_amount,

        # Expenses (Raw inputs used by the model)
        'monthly_rent': monthly_rent,
        'school_fees': school_fees,
        'college_fees': college_fees,
        'travel_expenses': travel_expenses,
        'groceries_utilities': groceries_utilities,
        'other_monthly_expenses': other_monthly_expenses,

        # Engineered Features (Must match the exact list used for training)
        'DTI_ratio': DTI_ratio,
        'LTI_ratio': LTI_ratio,
        'ETI_ratio': ETI_ratio,
        'liquidity_to_loan_ratio': liquidity_to_loan_ratio,
        'employment_stability_score': employment_stability_score,
        'bank_health_score': bank_health_score,
        'credit_debt_interaction': credit_debt_interaction,
        'salary_balance_interaction': salary_balance_interaction,
    }
    
    X_pred = pd.DataFrame([input_data])
    
    # --- 6. Make Predictions ---
    
    # Classification Prediction (Eligibility)
    cls_pred = CLASSIFIER.predict(X_pred)[0]
    
    # Classification Probability (Confidence)
    cls_proba = CLASSIFIER.predict_proba(X_pred)[0]
    predicted_proba = np.max(cls_proba)
    
    # Regression Prediction (Max EMI)
    log_reg_pred = REGRESSOR.predict(X_pred)[0]
    max_emi_pred = np.expm1(log_reg_pred) # Inverse transformation: expm1(log_reg_pred)
    
    # --- 7. Display Results ---
    st.markdown("## 📊 Prediction Results")
    col_cls, col_reg = st.columns(2)
    
    with col_cls:
        st.subheader("Loan Eligibility Assessment")
        if cls_pred == 'Eligible':
            st.success("✅ Loan Eligibility: **ELIGIBLE**", icon="💰")
        elif cls_pred == 'High_Risk':
            st.warning("⚠️ Loan Eligibility: **HIGH RISK**", icon="📉")
        else:
            st.error("❌ Loan Eligibility: **NOT ELIGIBLE**", icon="🛑")
        
        st.metric(label="Prediction Confidence", value=f"{predicted_proba:.2%}")

    with col_reg:
        st.subheader("Affordable EMI Calculation")
        st.info("🎯 Maximum Recommended Monthly EMI")
        st.metric(label="Affordable EMI", value=f"${max_emi_pred:,.2f}")

    st.markdown("---")
    st.subheader("Business Recommendation")
    
    if cls_pred == 'Eligible':
        st.markdown(f"**Recommendation:** **Approve** the loan. The customer is low-risk and can afford an EMI of up to **${max_emi_pred:,.2f}**.")
    elif cls_pred == 'High_Risk':
        st.markdown(f"**Recommendation:** **Conditional Approval**. The customer is moderate-risk. Offer a reduced loan amount or a strict EMI of no more than **${max_emi_pred:,.2f}**.")
    else:
        st.markdown(f"**Recommendation:** **Decline** the loan request. The customer's financial profile indicates a high risk of default, as they can only afford an EMI of **${max_emi_pred:,.2f}**.")