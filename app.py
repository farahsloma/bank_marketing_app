import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import json



df = pd.read_csv("data/bank-additional-full.csv" ,sep=';')
model = joblib.load('best_model.pkl')
def load_mapping(file):
    with open(file, 'r') as f:
        return json.load(f)

job_mapping = load_mapping('job_mapping.json')
marital_mapping = load_mapping('marital_mapping.json')
education_mapping = load_mapping('education_mapping.json')
housing_mapping = load_mapping('housing_mapping.json')
loan_mapping = load_mapping('loan_mapping.json')
contact_mapping = load_mapping('contact_mapping.json')
month_mapping = load_mapping('month_mapping.json')
day_mapping = load_mapping('day_of_week_mapping.json')
poutcome_mapping = load_mapping('poutcome_mapping.json')
loan_combo_mapping = load_mapping('loan_combo_mapping.json')

def model(): 
    st.title("Bank Marketing Subscription Prediction")

    
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", list(job_mapping.keys()))
    marital = st.selectbox("Marital Status", list(marital_mapping.keys()))
    education = st.selectbox("Education", list(education_mapping.keys()))
    housing = st.selectbox("Housing Loan", list(housing_mapping.keys()))
    loan = st.selectbox("Personal Loan", list(loan_mapping.keys()))
    contact = st.selectbox("Contact Communication Type", list(contact_mapping.keys()))
    month = st.selectbox("Last Contact Month", list(month_mapping.keys()))
    day_of_week = st.selectbox("Day of Week", list(day_mapping.keys()))
    campaign = st.number_input("Number of Contacts During Campaign", min_value=1, value=1)
    poutcome = st.selectbox("Previous Outcome", list(poutcome_mapping.keys()))

    emp_var_rate = st.number_input("Employment Variation Rate", value=1.1)
    cons_price_idx = st.number_input("Consumer Price Index", value=93.2)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-40.0)
    euribor3m = st.number_input("Euribor 3 Month Rate", value=4.86)
    nr_employed = st.number_input("Number of Employees", value=5191.0)
    duration = st.number_input("Call Duration (seconds)", min_value=1, value=180)
    duration_mean = 234.9973171797611 
    duration_ratio = duration / duration_mean
    loan_combo =housing + '_' + loan


    input_data = pd.DataFrame([{
        'age': age,
        'job': job_mapping[job],
        'marital': marital_mapping[marital],
        'education': education_mapping[education],
        'contact': contact_mapping[contact],
        'month': month_mapping[month],
        'day_of_week': day_mapping[day_of_week],
        'campaign': campaign,
        'poutcome': poutcome_mapping[poutcome],
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed,
        'loan_combo': loan_combo_mapping[loan_combo],
        'duration_ratio': duration_ratio
    }])
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
