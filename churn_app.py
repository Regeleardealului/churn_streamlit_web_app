import streamlit as st
import pandas as pd 
import numpy as np 
import pickle 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns 

st.set_page_config(
    page_title='Churn Predictor App',
    page_icon='C:/Users/sogor/OneDrive/Documents/DataScientist_practice/python/customer_churn_app/app_logo.png',
    layout='wide'
) 

data = pd.read_csv('C:/Users/sogor/OneDrive/Documents/DataScientist_practice/python/customer_churn_app/cleaned_churn.csv')

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
        plt.figure(figsize=(6, 4))  
        plt.hist([data[data['Churn'] == 'Yes']['tenure'], data[data['Churn'] == 'No']['tenure']], bins=10, color=['#FF0000', '#387F39'], label=['Canceled their subscription', 'Remains subscribed'])
        plt.title('Distribution of Tenure by Customer Churn')
        plt.xlabel('Tenure')
        plt.ylabel('Frequency')
        plt.legend()
        for rect in plt.gca().patches:
            height = rect.get_height()
            plt.gca().text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
        st.pyplot(plt.gcf())
        plt.clf()  

        plt.figure(figsize=(6, 4))
        sns.boxplot(x='Churn', y='MonthlyCharges', data=data)
        plt.title('Distribution of Monthly Charges by Churn Status')
        plt.xlabel('')
        plt.ylabel('Monthly Charges')
        plt.xticks(ticks=[0, 1], labels=['Remains subscribed', 'Canceled their subscription'])
        st.pyplot(plt.gcf())
        plt.clf() 

        plt.figure(figsize=(6, 4)) 
        sns.kdeplot(data=data, x="TotalCharges", hue="Churn", fill=True, alpha=0.5)
        plt.title('Density Plot of Total Charges by Churn Status')
        plt.xlabel('Total Charges')
        plt.ylabel('Density')
        st.pyplot(plt.gcf())
        plt.clf()

    elif selection == 'Categorical Features':
        categorical_features = {
            'InternetService': data['InternetService'],
            'PaymentMethod': data['PaymentMethod'],
            'Contract': data['Contract']
        }

        num_features = len(categorical_features)
        fig, axes = plt.subplots(1, num_features, figsize=(12, 8), constrained_layout=True)

        if num_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, (feature, values) in zip(axes, categorical_features.items()):
            feature_churn_counts = data.groupby([feature, 'Churn']).size().unstack(fill_value=0)
            feature_churn_counts.plot(kind='bar', stacked=True, ax=ax, color=['green', 'red'])
            
            ax.set_title(feature)
            ax.set_xlabel('')
            ax.set_ylabel('Count')
            ax.legend(title='Churn Status', labels=['Remains subscribed', 'Canceled their subscription'])
            
            ax.set_xticks(range(len(feature_churn_counts.index)))
            ax.set_xticklabels(feature_churn_counts.index, rotation=45, ha='right')

        st.pyplot(fig)
        plt.clf()

    else:
        dichotomous_features = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
            'StreamingTV', 'StreamingMovies', 'PaperlessBilling'
        ]

        num_features = len(dichotomous_features)
        num_rows = 4
        num_cols = 3
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12), constrained_layout=True)

        for i, feature in enumerate(dichotomous_features):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col]
            
            sns.countplot(data=data, x=feature, hue='Churn', ax=ax)
            ax.set_title(feature)
            ax.set_xlabel('')
            ax.set_ylabel('Count')
            ax.legend(title='Churn Status', labels=['Remains subscribed', 'Canceled their subscription'])

        if num_features < num_rows * num_cols:
            for j in range(num_features, num_rows * num_cols):
                fig.delaxes(axes.flatten()[j])

        st.pyplot(fig)
        plt.clf()
    
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
