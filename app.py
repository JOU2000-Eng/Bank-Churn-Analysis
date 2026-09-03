import streamlit as st
import joblib

# Load the trained model
model = joblib.load("bank_churn_model.pkl")
st.title("🏦 Bank Customer Churn Prediction")

st.write(
    "Predict whether a bank customer is likely to leave the bank "
    "using a machine learning model."
)
st.subheader("Customer Information")



credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)

country = st.selectbox(
    "Country",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.number_input(
    "Tenure (Years)",
    min_value=0,
    max_value=10,
    value=5
)
balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

products_number = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

credit_card = st.selectbox(
    "Has Credit Card?",
    [0, 1]
)

active_member = st.selectbox(
    "Active Member?",
    [0, 1]
)
estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)
import pandas as pd

input_data = pd.DataFrame({
    'credit_score': [credit_score],
    'country': [country],
    'gender': [gender],
    'age': [age],
    'tenure': [tenure],
    'balance': [balance],
    'products_number': [products_number],
    'credit_card': [credit_card],
    'active_member': [active_member],
    'estimated_salary': [estimated_salary]
})
if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")

    st.write(f"Churn Probability: {probability:.1%}")
