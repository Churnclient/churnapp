from joblib import load
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import requests


# Define the URL of the raw model file on GitHub
model_url = 'https://github.com/Churnclient/churnapp/blob/main/model.pkl'

# Define a function to download the model file from GitHub
@st.cache(allow_output_mutation=True)
def download_model(url):
    response = requests.get(url)
    with open('model.plk', 'wb') as file:
        file.write(response.content)

# Download the model file from GitHub
model = download_model(model_url)


# Define the features
features = ["CreditScore", "Geography_France", "Geography_Spain", "Geography_Germany", "Gender_Male", "Gender_Female", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]

# Define the StandardScaler
scaler = StandardScaler()

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

# Prepare the input data for prediction
input_data = np.array([[credit_score]])

# Encode the categorical variables
if geography == "France":
    input_data = np.append(input_data, [1, 0, 0])
elif geography == "Spain":
    input_data = np.append(input_data, [0, 1, 0])
else:
    input_data = np.append(input_data, [0, 0, 1])

if gender == "Male":
    input_data = np.append(input_data, [1, 0])
else:
    input_data = np.append(input_data, [0, 1])

input_data = np.append(input_data, [age, tenure, balance, num_of_products])

if has_cr_card == "Yes":
    input_data = np.append(input_data, [1])
else:
    input_data = np.append(input_data, [0])

if is_active_member == "Yes":
    input_data = np.append(input_data, [1])
else:
    input_data = np.append(input_data, [0])

input_data = np.append(input_data, [estimated_salary])

# Scale the input data
input_data = scaler.fit_transform(input_data.reshape(1, -1))

# Make a prediction
prediction = model.predict(input_data)

# Display the prediction
st.write("## Prediction")
if prediction == 0:
    st.write("The client is predicted to stay.")
    st.balloons()
else:
    st.write("The client is predicted to churn.")
    st.snow()
st.stop()
