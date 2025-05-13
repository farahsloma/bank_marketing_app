import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


df = pd.read_csv("data/bank-additional-full.csv" ,sep=';')
model = joblib.load('best_model.pkl')
le_job = joblib.load('job_encode.pkl')
le_marital = joblib.load('marital.pkl')
le_education = joblib.load('education.pkl')
le_housing = joblib.load('housing.pkl')
le_loan = joblib.load('loan.pkl')
le_contact = joblib.load('contact.pkl')
le_month = joblib.load('month.pkl')
le_day = joblib.load('day_of_week.pkl')
le_poutcome = joblib.load('poutcome.pkl')
le_campaign = joblib.load('campaign.pkl')
le_loan_combo = joblib.load('loan_combo.pkl')


def model(): 
    st.title("Bank Marketing Subscription Prediction")

    
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", le_job.classes_)
    marital = st.selectbox("Marital Status", le_marital.classes_)
    education = st.selectbox("Education", le_education.classes_)
    housing = st.selectbox("Housing Loan", le_housing.classes_)
    loan = st.selectbox("Personal Loan", le_loan.classes_)
    contact = st.selectbox("Contact Communication Type", le_contact.classes_)
    month = st.selectbox("Last Contact Month", le_month.classes_)
    day_of_week = st.selectbox("Day of Week", le_day.classes_)
    campaign = st.number_input("Number of Contacts During Campaign", min_value=1, value=1)
    poutcome = st.selectbox("Previous Outcome", le_poutcome.classes_)

    emp_var_rate = st.number_input("Employment Variation Rate", value=1.1)
    cons_price_idx = st.number_input("Consumer Price Index", value=93.2)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-40.0)
    euribor3m = st.number_input("Euribor 3 Month Rate", value=4.86)
    nr_employed = st.number_input("Number of Employees", value=5191.0)
    duration = st.number_input("Call Duration (seconds)", min_value=1, value=180)
    duration_mean = 234.9973171797611 
    duration_ratio = duration / duration_mean
    loan_combo =housing + '_' + loan

    input_data = pd.DataFrame({
        'age': [age],
        'job': le_job.transform([job]),
        'marital': le_marital.transform([marital]),
        'education': le_education.transform([education]),
        'contact': le_contact.transform([contact]),
        'month': le_month.transform([month]),
        'day_of_week': le_day.transform([day_of_week]),
        'campaign': le_campaign.transform([campaign]),
        'poutcome': le_poutcome.transform([poutcome]),
        'emp.var.rate': [emp_var_rate],
        'cons.price.idx': [cons_price_idx],
        'cons.conf.idx': [cons_conf_idx],
        'euribor3m': [euribor3m],
        'nr.employed': [nr_employed],
        'loan_combo' : le_loan_combo.transform([loan_combo]),
        'duration_ratio': [duration_ratio],

    })
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        if prediction == 'yes':
            st.success("The customer is likely to **subscribe**.")
        else:
            st.warning("The customer is **not likely to subscribe**.")
            
def plots ():
    st.subheader("Distribution of Campaign Response (Yes/No)")
    fig1, ax1 = plt.subplots()
    df['y'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'salmon'], ax=ax1)
    ax1.set_ylabel('')
    ax1.set_title('Campaign Response')
    st.pyplot(fig1)
    
    
    st.subheader("Campaign Response by Job Title")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    job_response = df.groupby(['job', 'y']).size().unstack().fillna(0)
    job_response.plot(kind='bar', stacked=True, ax=ax4, colormap='Set3')
    ax4.set_title("Response by Job")
    ax4.set_ylabel("Count")
    ax4.set_xlabel("Job")
    plt.xticks(rotation=45)
    st.pyplot(fig4)

def about():
 

    st.title("📊 Bank Marketing Prediction App")

    with st.expander("ℹ️ About the Project"):
        st.markdown("""
        **Project Goal:**  
        This project is based on marketing campaign data conducted by a European financial institution.  
        The main goal is to predict whether a client will subscribe to a term deposit using a machine learning model.

        **What this app offers:**
        - Display and explore client data
        - Train and evaluate a GradientBoosting model
        - Predict client response to the marketing campaign
        - Visualize key insights and feature relationships

        **Key Features:**
        - Used `SMOTE` to handle class imbalance
        - Applied `LabelEncoder` with saved encoders for each categorical feature
        - Engineered features like `duration_ratio` and `has_loan`
        - Built an interactive UI using Streamlit

        ---

        **🧠 Developer:** Farah  
        **📅 Year:** 2025  
        """)

page = st.sidebar.selectbox("Select page", ["Predict",'plots','About'])
if page == 'Predict' :
    model()
elif page == 'plots':
    plots()
elif page == 'About' :
    about()
