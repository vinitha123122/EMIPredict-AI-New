*EMIPredict AI: Intelligent Financial Risk Assessment Platform
*Project Overview: EMIPredict AI is an end-to-end MLOps solution designed to assess the credit risk of loan applicants and determine their maximum affordable Equated Monthly Installment (EMI). The platform uses dual, high-performance XGBoost machine learning pipelines and is deployed as a multi-page web application on Streamlit Cloud.
This project demonstrates a full MLOps lifecycle, including robust data processing (400,000 records), MLflow experiment tracking, and CI/CD deployment via GitHub.

*Live Application
The application is hosted and publicly accessible via Streamlit Cloud.

Live URL: [EMIPredict AI Live App](https://emipredict-ai-new-e2hovjyvdtjcncgweabhc2.streamlit.app/)

*Best Performing Models: Model Type                Best Model       Key Metric   Score
                        Classification (Risk)	 XGBoost Classifier	 F1 Score	   0.9678
                        Regression (Max EMI)   XGBoost Regressor   R-squared   0.9763

*How to Run Locally
To set up and run this project on your local machine, follow these steps:
Python 3.8+
Git
1.clone
2.python -m venv venv
  source venv/bin/activate  # On Windows, use: venv\Scripts\activate
  pip install -r requirements.txt
3. Obtain Model Files
4. Run the Streamlit Application
  streamlit run Home.py
  
