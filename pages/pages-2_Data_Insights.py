import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.title("📊 Data Exploration and Visualization")
st.subheader("Interactive Insights from EMI Prediction Dataset")

# --- 1. Data Loading ---
try:
    # FIX: Added low_memory=False to suppress the DtypeWarning
    df = pd.read_csv('emi_prediction_dataset.csv', low_memory=False) 
    
    # Simple cleaning for display consistency
    df['emi_eligibility'] = df['emi_eligibility'].str.replace('_', ' ')
    
    # Clean up numerical columns
    def clean_and_convert(series):
        return pd.to_numeric(series.astype(str).str.replace(',', '').str.replace('$', '').str.strip(), errors='coerce')
    
    df['monthly_salary'] = clean_and_convert(df['monthly_salary'])
    df['max_monthly_emi'] = clean_and_convert(df['max_monthly_emi'])
    df['credit_score'] = clean_and_convert(df['credit_score'])
    
    df.dropna(subset=['emi_eligibility', 'monthly_salary', 'credit_score', 'max_monthly_emi'], inplace=True)

except FileNotFoundError:
    st.error("Error: 'emi_prediction_dataset.csv' not found. Please place it in the root directory.")
    df = pd.DataFrame()
except Exception as e:
    st.error(f"Error loading or processing data: {e}")
    df = pd.DataFrame()


if not df.empty:
    st.markdown("---")
    st.header("1. Target Distribution and Summary")
    
    col_pie, col_sum = st.columns([1, 1])
    
    with col_pie:
        st.subheader("EMI Eligibility Distribution")
        eligibility_counts = df['emi_eligibility'].value_counts().reset_index()
        eligibility_counts.columns = ['Eligibility', 'Count']
        
        fig_pie = px.pie(eligibility_counts, names='Eligibility', values='Count', 
                         title='Distribution of Loan Eligibility Status', hole=0.4)
        # DEPRECATION FIX: use_container_width=True changed to width='stretch'
        st.plotly_chart(fig_pie, width='stretch') 

    with col_sum:
        st.subheader("Dataset Summary")
        # st.dataframe uses use_container_width=True by default when no width is set
        st.dataframe(df.describe().T, height=350) 
        st.write(f"**Total Records:** {len(df)}")
        
    st.markdown("---")

    # --- 2. Income and Eligibility Analysis ---
    st.header("2. Income vs. Eligibility")
    
    st.subheader("Monthly Salary vs. Eligibility Status")
    salary_eligibility = df.groupby('emi_eligibility')['monthly_salary'].agg(['mean', 'median']).reset_index()
    salary_eligibility = salary_eligibility.melt(id_vars='emi_eligibility', var_name='Metric', value_name='Salary')
    
    fig_bar = px.bar(salary_eligibility, x='emi_eligibility', y='Salary', color='Metric', 
                     barmode='group', title='Average and Median Salary by Eligibility Status',
                     labels={'Salary': 'Monthly Salary ($)', 'emi_eligibility': 'Eligibility Status'})
    # DEPRECATION FIX: use_container_width=True changed to width='stretch'
    st.plotly_chart(fig_bar, width='stretch')
    
    st.markdown("---")

    # --- 3. Risk and Affordability Analysis ---
    st.header("3. Risk Metrics & Affordability")

    col_credit, col_emi = st.columns(2)
    
    with col_credit:
        st.subheader("Credit Score Distribution by Eligibility")
        fig_box_credit = px.box(df, x='emi_eligibility', y='credit_score', 
                                title='Credit Score Distribution by Eligibility', 
                                color='emi_eligibility', 
                                labels={'credit_score': 'Credit Score'})
        # DEPRECATION FIX: use_container_width=True changed to width='stretch'
        st.plotly_chart(fig_box_credit, width='stretch') 

    with col_emi:
        st.subheader("Max Affordable EMI by Eligibility")
        df_emi = df[df['max_monthly_emi'] > 500].copy() 
        
        fig_box_emi = px.box(df_emi, x='emi_eligibility', y='max_monthly_emi', 
                             title='Distribution of Max Affordable EMI', 
                             color='emi_eligibility', 
                             labels={'max_monthly_emi': 'Max Monthly EMI ($)'})
        # DEPRECATION FIX: use_container_width=True changed to width='stretch'
        st.plotly_chart(fig_box_emi, width='stretch')