import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

print("--- Starting FINAL MODEL RE-SAVING FIX ---")

# --- 1. Data Preparation (Must match Step 4) ---
df = pd.read_csv('emi_prediction_dataset.csv')
epsilon = 1e-6

# Cleaning & Type Conversion
df.dropna(subset=['emi_eligibility', 'max_monthly_emi'], inplace=True)
def clean_and_convert(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '').str.replace('$', '').str.strip(), errors='coerce')
df['age'], df['monthly_salary'], df['bank_balance'] = clean_and_convert(df['age']), clean_and_convert(df['monthly_salary']), clean_and_convert(df['bank_balance'])

# Feature Engineering
expense_only_cols = ['monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
expense_cols = expense_only_cols + ['current_emi_amount']
df['DTI_ratio'] = df[expense_cols].sum(axis=1) / (df['monthly_salary'] + epsilon)
df['LTI_ratio'] = df['requested_amount'] / (df['monthly_salary'] + epsilon)
df['ETI_ratio'] = df[expense_only_cols].sum(axis=1) / (df['monthly_salary'] + epsilon)
df['liquidity_to_loan_ratio'] = df['bank_balance'] / (df['requested_amount'] + epsilon)
df['employment_stability_score'] = df['years_of_employment'] / (df['age'] + epsilon)
df['bank_health_score'] = df['bank_balance'] / (df['monthly_salary'] + epsilon)
df['capped_DTI_safety'] = 1 - np.minimum(df['DTI_ratio'], 1.0)
df['credit_debt_interaction'] = df['credit_score'] * df['capped_DTI_safety']
df['salary_balance_interaction'] = df['monthly_salary'] * df['bank_balance']
df.drop(columns=['capped_DTI_safety'], inplace=True)

# Define X, Y targets and Preprocessor
X = df.drop(columns=['emi_eligibility', 'max_monthly_emi', 'emi_scenario', 'requested_amount', 'current_emi_amount'] + expense_only_cols)
Y_class = df['emi_eligibility']
Y_reg = df['max_monthly_emi']

numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ], remainder='drop'
)

# Data Split and Encoding
X_train_cls, _, Y_train_cls, _ = train_test_split(X, Y_class, test_size=0.2, random_state=42, stratify=Y_class)
X_train_reg, _, Y_train_reg, _ = train_test_split(X, Y_reg, test_size=0.2, random_state=42)
Y_train_reg_log = np.log1p(Y_train_reg)

le = LabelEncoder()
Y_train_cls_encoded = le.fit_transform(Y_train_cls)


# --- 2. Build and Save XGBoost Pipelines ---

# CLASSIFICATION PIPELINE (XGBoost)
xgb_cls = xgb.XGBClassifier(objective='multi:softprob', n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
final_cls_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb_cls)])
final_cls_pipeline.fit(X_train_cls, Y_train_cls_encoded)
joblib.dump(final_cls_pipeline, 'best_emi_classifier_pipeline.pkl')
print("✅ Classification Model (best_emi_classifier_pipeline.pkl) successfully rebuilt and saved.")


# REGRESSION PIPELINE (XGBoost)
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
final_reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', xgb_reg)])
final_reg_pipeline.fit(X_train_reg, Y_train_reg_log)
joblib.dump(final_reg_pipeline, 'best_emi_regressor_pipeline.pkl')
print("✅ Regression Model (best_emi_regressor_pipeline.pkl) successfully rebuilt and saved.")

print("--- FIX COMPLETE. You can now relaunch Streamlit. ---")