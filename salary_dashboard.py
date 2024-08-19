import streamlit as st
import pandas as pd
import joblib

# Load the trained best model
best_model = joblib.load('linear_regression_model.pkl')

# Create the Streamlit app
st.title('HR Salary Prediction Dashboard')

age = st.slider('Age', 22, 60)
experience = st.slider('Years of Experience', 1, 38)
current_salary = st.number_input('Current Salary', min_value=30000, max_value=200000, step=1000)

if st.button('Predict Salary'):
    input_data = pd.DataFrame({
        'Age': [age],
        'Years_of_Experience': [experience],
        'Current_Salary': [current_salary]
    })
    predicted_salary = best_model.predict(input_data)[0]
    st.write(f'The predicted salary for switching jobs is: ${predicted_salary:.2f}')