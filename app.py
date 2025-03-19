import streamlit as st
import pandas as pd
import pickle
import os

# Define the paths to the preprocessor and model files in the artifacts directory
preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
model_path = os.path.join('artifacts', 'model.pkl')

# Load preprocessor and model
try:
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
except FileNotFoundError:
    st.error("Preprocessor file not found. Please check the file path.")

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")

# Title of the app
st.title('Heart Disease Prediction')

# User inputs
st.header('Enter Patient Details')

# Numerical features
age = st.number_input('Age', value=30)
blood_pressure = st.number_input('Blood Pressure (mmHg)', value=120)
cholesterol_level = st.number_input('Cholesterol Level (mg/dL)', value=200)
bmi = st.number_input('BMI', value=25.0)
sleep_hours = st.number_input('Sleep Hours', value=7.0)
triglyceride_level = st.number_input('Triglyceride Level (mg/dL)', value=150)
fasting_blood_sugar = st.number_input('Fasting Blood Sugar (mg/dL)', value=90)
crp_level = st.number_input('CRP Level (mg/L)', value=2.0)
homocysteine_level = st.number_input('Homocysteine Level (µmol/L)', value=12.0)

# Categorical features with options
gender = st.selectbox('Gender', ['Female', 'Male'])
exercise_habits = st.selectbox('Exercise Habits', ['High', 'Medium', 'Low'])
smoking = st.selectbox('Smoking', ['Yes', 'No'])
family_heart_disease = st.selectbox('Family Heart Disease', ['Yes', 'No'])
diabetes = st.selectbox('Diabetes', ['Yes', 'No'])
high_bp = st.selectbox('High Blood Pressure', ['Yes', 'No'])
low_hdl = st.selectbox('Low HDL Cholesterol', ['Yes', 'No'])
high_ldl = st.selectbox('High LDL Cholesterol', ['Yes', 'No'])
alcohol_consumption = st.selectbox('Alcohol Consumption', ['High', 'Medium', 'Low'])
stress_level = st.selectbox('Stress Level', ['High', 'Medium', 'Low'])
sugar_consumption = st.selectbox('Sugar Consumption', ['High', 'Medium', 'Low'])

# Predict button
if st.button('Predict Heart Disease Risk'):
    try:
        # Collect input features
        input_features = pd.DataFrame({
            'Age': [age],
            'Blood Pressure': [blood_pressure],
            'Cholesterol Level': [cholesterol_level],
            'BMI': [bmi],
            'Sleep Hours': [sleep_hours],
            'Triglyceride Level': [triglyceride_level],
            'Fasting Blood Sugar': [fasting_blood_sugar],
            'CRP Level': [crp_level],
            'Homocysteine Level': [homocysteine_level],
            'Gender': [gender],
            'Exercise Habits': [exercise_habits],
            'Smoking': [smoking],
            'Family Heart Disease': [family_heart_disease],
            'Diabetes': [diabetes],
            'High Blood Pressure': [high_bp],
            'Low HDL Cholesterol': [low_hdl],
            'High LDL Cholesterol': [high_ldl],
            'Alcohol Consumption': [alcohol_consumption],
            'Stress Level': [stress_level],
            'Sugar Consumption': [sugar_consumption]
        })

        # Preprocess input
        preprocessed_features = preprocessor.transform(input_features)

        # Predict
        prediction = model.predict(preprocessed_features)
        result = 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease Detected'

        # Display result
        st.success(f'Prediction: {result}')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
