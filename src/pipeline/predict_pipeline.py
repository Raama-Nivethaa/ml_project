import os
import sys
import pandas as pd
from src.exception.exception import customexception
from src.logger.log_helper import logging
from src.utils.utils import load_object


class PredictPipeline:
    
    def __init__(self):
        print("Initializing Prediction Pipeline...")

    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Preprocess input features
            scaled_features = preprocessor.transform(features)
            prediction = model.predict(scaled_features)

            return prediction

        except Exception as e:
            raise customexception(e, sys)


class CustomData:
    def __init__(self,
                 age: float,
                 blood_pressure: float,
                 cholesterol_level: float,
                 bmi: float,
                 sleep_hours: float,
                 triglyceride_level: float,
                 fasting_blood_sugar: float,
                 crp_level: float,
                 homocysteine_level: float,
                 gender: str,
                 exercise_habits: str,
                 smoking: str,
                 family_heart_disease: str,
                 diabetes: str,
                 high_blood_pressure: str,
                 low_hdl_cholesterol: str,
                 high_ldl_cholesterol: str,
                 alcohol_consumption: str,
                 stress_level: str,
                 sugar_consumption: str):

        self.age = age
        self.blood_pressure = blood_pressure
        self.cholesterol_level = cholesterol_level
        self.bmi = bmi
        self.sleep_hours = sleep_hours
        self.triglyceride_level = triglyceride_level
        self.fasting_blood_sugar = fasting_blood_sugar
        self.crp_level = crp_level
        self.homocysteine_level = homocysteine_level
        self.gender = gender
        self.exercise_habits = exercise_habits
        self.smoking = smoking
        self.family_heart_disease = family_heart_disease
        self.diabetes = diabetes
        self.high_blood_pressure = high_blood_pressure
        self.low_hdl_cholesterol = low_hdl_cholesterol
        self.high_ldl_cholesterol = high_ldl_cholesterol
        self.alcohol_consumption = alcohol_consumption
        self.stress_level = stress_level
        self.sugar_consumption = sugar_consumption

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                'Age': [self.age],
                'Blood Pressure': [self.blood_pressure],
                'Cholesterol Level': [self.cholesterol_level],
                'BMI': [self.bmi],
                'Sleep Hours': [self.sleep_hours],
                'Triglyceride Level': [self.triglyceride_level],
                'Fasting Blood Sugar': [self.fasting_blood_sugar],
                'CRP Level': [self.crp_level],
                'Homocysteine Level': [self.homocysteine_level],
                'Gender': [self.gender],
                'Exercise Habits': [self.exercise_habits],
                'Smoking': [self.smoking],
                'Family Heart Disease': [self.family_heart_disease],
                'Diabetes': [self.diabetes],
                'High Blood Pressure': [self.high_blood_pressure],
                'Low HDL Cholesterol': [self.low_hdl_cholesterol],
                'High LDL Cholesterol': [self.high_ldl_cholesterol],
                'Alcohol Consumption': [self.alcohol_consumption],
                'Stress Level': [self.stress_level],
                'Sugar Consumption': [self.sugar_consumption]
            }

            df = pd.DataFrame(data_dict)
            logging.info('Dataframe created successfully.')
            return df
        except Exception as e:
            logging.error('Exception occurred in data frame creation.')
            raise customexception(e, sys)
