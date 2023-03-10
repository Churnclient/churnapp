from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import requests
import joblib

# Define the URL of the raw model file on GitHub
model_url = 'https://github.com/Churnclient/churnapp/raw/main/model22.h5'
scaler_url = 'https://github.com/Churnclient/churnapp/raw/main/scaler.pkl'

# Define a function to download the model file from GitHub
def download_model(url):
    response = requests.get(url)
    with open('model22.h5', 'wb') as file:
        file.write(response.content)

# Define a function to download the scaler file from GitHub
def download_scaler(url):
    response = requests.get(url)
    with open('scaler.pkl', 'wb') as file:
        file.write(response.content)

# Download the model and scaler files from GitHub
download_model(model_url)
download_scaler(scaler_url)

# Load the model and scaler
model2 = load_model('model22.h5')
scaler = joblib.load('scaler.pkl')

# Define the features
features = ["CreditScore", "Geography_France", "Geography_Spain", "Geography_Germany", "Gender_Male", "Gender_Female", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]

# Define the input fields
st.write("# Client Churn Prediction")
credit_score = st.number_input("Credit Score", value=650)
geography = st.radio("Geography", options=["France", "Spain", "Germany"])
gender = st.radio("Gender", options=["Male", "Female"])
age = st.number_input("Age", value=35)
tenure = st.number_input("Tenure", value=5)
balance = st.number_input("Balance", value=100000)
num_of_products = st.number_input("Number of Products", value=2)
has_cr_card = st.radio("Has Credit Card", options=["Yes", "No"])
is_active_member = st.radio("Is Active Member", options=["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", value=100000)


if st.button("OK"):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography_France": [1 if geography == "France" else 0],
        "Geography_Spain": [1 if geography == "Spain" else 0],
        "Geography_Germany": [1 if geography == "Germany" else 0],
        "Gender_Male": [1 if gender == "Male" else 0],
        "Gender_Female": [1 if gender == "Female" else 0],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [1 if has_cr_card == "Yes" else 0],
        "IsActiveMember": [1 if is_active_member == "Yes" else 0],
        "EstimatedSalary": [estimated_salary]
    })

    # Scale the input data
    input_data = scaler.transform(input_data)

    # Make a prediction
    prediction = model2.predict(input_data)

    # Display the prediction
    st.write("## Prediction")
    if prediction[0][0] < 0.5:
        st.write("<h1 style='color: green;'>The client is predicted to stay</h1>", unsafe_allow_html=True)
        st.balloons()
    else:
        st.write("<h1 style='color: red;'>The client is predicted to churn</h1>", unsafe_allow_html=True)
        st.snow()
