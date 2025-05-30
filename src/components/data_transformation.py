import pandas as pd
import numpy as np
from src.logger.log_helper import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from src.utils.utils import save_object


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.preprocessing import LabelEncoder



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformation(self):
        
        try:
            logging.info('Data Transformation initiated')
            
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease',
              'Diabetes', 'High Blood Pressure', 'Low HDL Cholesterol','High LDL Cholesterol', 'Alcohol Consumption', 'Stress Level','Sugar Consumption']
            numerical_cols = [ 'Age', 'Blood Pressure', 'Cholesterol Level', 'BMI',
              'Sleep Hours', 'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level','Homocysteine Level']
            
            # Define the custom ranking for each ordinal variable
            Gender_categories=['Female','Male']
            Exercise_Habits_categories=['High','Medium','Low']
            Smoking_categories=['Yes','No']
            Family_Heart_Disease_categories=['Yes','No']
            Diabetes_categories=['Yes','No']
            High_Blood_Pressure_categories=['Yes','No']
            Low_HDL_Cholesterol_categories=['Yes','No']
            High_LDL_Cholesterol_categories=['Yes','No']
            Alcohol_Consumption_categories=['High','Medium','Low']
            Stress_Level_categories=['High','Medium','Low']
            Sugar_Consumption_categories=['High','Medium','Low']
            
            logging.info('Pipeline Initiated')
            
            num_pipeline = Pipeline(
                steps = [
                ("imputer",SimpleImputer(strategy="mean")),
                ("sclaer",StandardScaler())

                ]
            )
            
            
            cat_pipeline = Pipeline(
                steps = [
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ("ordinal encoder",OrdinalEncoder(categories=[Gender_categories,Exercise_Habits_categories,Smoking_categories,Family_Heart_Disease_categories,Diabetes_categories,High_Blood_Pressure_categories,Low_HDL_Cholesterol_categories,High_LDL_Cholesterol_categories,Alcohol_Consumption_categories,Stress_Level_categories,Sugar_Consumption_categories]))

                ]
            )
            preprocessor =  ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols)
                ]
            )
            return preprocessor
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)
        
    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            preprocessing_obj = self.get_data_transformation()
            
            target_column_name = 'Heart Disease Status'
            
            
            input_feature_train_df = train_df.drop(columns= target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            
            input_feature_test_df=test_df.drop(columns= target_column_name,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info("Applying Label Encoding to the target column")
            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = label_encoder.transform(target_feature_test_df)
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)
                