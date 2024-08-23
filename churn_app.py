import streamlit as st
import pandas as pd 
import numpy as np 
import pickle 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title='Churn Predictor App',
    page_icon='C:/Users/sogor/OneDrive/Documents/DataScientist_practice/python/customer_churn_app/tinified/app_logo.png',
    layout='wide'
) 

data = pd.read_csv('C:/Users/sogor/OneDrive/Documents/DataScientist_practice/python/customer_churn_app/Customer_Churn.csv')

with open('C:/Users/sogor/OneDrive/Documents/DataScientist_practice/python/customer_churn_app/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('C:/Users/sogor/OneDrive/Documents/DataScientist_practice/python/customer_churn_app/churn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.title("Customer Churn Prediction")

tab1, tab2, tab3 = st.tabs(["Home", "Visualization", "Prediction"])

with tab1:
    st.header("Introduction: few words about the project :smile:")
    st.write("This project predicts whether a customer will churn (cancel their subscription) or remain subscribed based on various features of their account and behavior.")
    st.write("I have built a Neural Network for this project using Tensorflow library from Python.")
    st.write("Here are the first 5 rows of the dataset:")
    st.dataframe(data.head())

with tab2:
    st.header("Visualizations")
    selection = st.radio(label='Select Visualization Type', options=['Numerical Features', 'Dichotomous Features', 'Categorical Features'])
    
    if selection == 'Numerical Features':
        st.image(r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\monthly_charges_by_churn.jpg')
        st.image(r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\total_charges_by_churn.jpg')
        st.image(r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\tenure_by_churn.jpg')
    
    elif selection == 'Categorical Features':
        st.image(r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\service_by_churn.jpg')
    
    else:
        images = [
            r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\dependents_by_churn.jpg', 
            r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\device_protection_by_churn.jpg',
            r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\gender_by_churn.jpg',
            r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\multiplelines_by_churn.jpg',
            r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\online_back_up_by_churn.jpg',
            r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\papers_billing_by_churn.jpg',
            r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\partner_by_churn.jpg',
            r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\online_security_by_churn.jpg',
            r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\phone_service_by_churn.jpg',
            r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\streaming_movies_by_churn.jpg',
            r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\streaming_tv_by_churn.jpg',
            r'C:\Users\sogor\OneDrive\Documents\DataScientist_practice\python\customer_churn_app\tinified\tech_support_by_churn.jpg'
        ]

        rows = 4
        cols = 3
        for i in range(0, len(images), cols):
            cols_list = st.columns(cols)
            for j, img_path in enumerate(images[i:i+cols]):
                with cols_list[j]:
                    st.image(img_path)

with tab3:
    st.header("Prediction")
    
    with st.form(key='prediction_form'):
        st.write("#### *Customer Details for Prediction*")
        
        gender = st.selectbox("What's your gender? (1 for Male, 0 for Female)", options=[1, 0])  
        seniorcitizen = st.selectbox("Are you a senior citizen? (1 for Yes, 0 for No)", options=[1, 0])
        partner = st.selectbox("Do you have a partner? (1 for Yes, 0 for No)", options=[1, 0])
        dependents = st.selectbox("Do you have dependents? (1 for Yes, 0 for No)", options=[1, 0])
        tenure = st.slider("How many months have you been with the company?", min_value=0, max_value=100)
        phone_service = st.selectbox("Do you have a phone service? (1 for Yes, 0 for No)", options=[1, 0])
        multiple_lines = st.selectbox("Do you have multiple phone lines? (1 for Yes, 0 for No)", options=[1, 0])
        internet_service = st.selectbox("What type of internet service do you have? (0 for No Service, 1 for DSL, 2 for Fiber Optic)", options=[0, 1, 2])
        online_security = st.selectbox("Do you have online security? (1 for Yes, 0 for No)", options=[1, 0])
        online_backup = st.selectbox("Do you have online backup? (1 for Yes, 0 for No)", options=[1, 0])
        device_protection = st.selectbox("Do you have device protection? (1 for Yes, 0 for No)", options=[1, 0])
        tech_support = st.selectbox("Do you have tech support? (1 for Yes, 0 for No)", options=[1, 0])
        streaming_tv = st.selectbox("Do you use streaming TV services? (1 for Yes, 0 for No)", options=[1, 0])
        streaming_movies = st.selectbox("Do you use streaming movie services? (1 for Yes, 0 for No)", options=[1, 0])
        contract = st.selectbox("What type of contract do you have? (0 for Month-to-month, 1 for One year, 2 for Two year)", options=[0, 1, 2])
        paperless_billing = st.selectbox("Do you use paperless billing? (1 for Yes, 0 for No)", options=[1, 0])
        bank_transfer = st.selectbox("Do you pay via bank transfer? (1 for Yes, 0 for No)", options=[1, 0])
        credit_card = st.selectbox("Do you pay via credit card? (1 for Yes, 0 for No)", options=[1, 0])
        electronic_check = st.selectbox("Do you pay via electronic check? (1 for Yes, 0 for No)", options=[1, 0])
        mailed_check = st.selectbox("Do you pay via mailed check? (1 for Yes, 0 for No)", options=[1, 0])
        monthly_charges = st.number_input("What are your monthly charges?", min_value=0.0)
        total_charges = st.number_input("What are your total charges?", min_value=0.0)

        # Submit button
        submit_button = st.form_submit_button("Predict")

        if submit_button:
            user_inputs = np.array([gender, seniorcitizen, partner, dependents, tenure, phone_service, multiple_lines, internet_service, 
                                    online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract,
                                    paperless_billing, bank_transfer, credit_card, electronic_check, mailed_check, monthly_charges, total_charges]).reshape(1, -1)

            # Scale the input data
            lst_scaled = scaler.transform(user_inputs)

            # Predict churn
            prediction = model.predict(lst_scaled)
            if prediction[0][0] != 0.0:  
                st.write('The customer will cancel their subscription.') 
            else:
                st.write('The customer remains subscribed.')
