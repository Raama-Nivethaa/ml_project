import os
import sys
from src.logger.log_helper import logging
from src.exception.exception import customexception
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


def main():
    try:
        logging.info("Starting Heart Disease Prediction Pipeline")

        # Data Ingestion
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Train Path: {train_data_path}, Test Path: {test_data_path}")

        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)
        logging.info("Data Transformation completed.")

        # Model Training
        model_trainer_obj = ModelTrainer()
        model_trainer_obj.initiate_model_training(train_arr, test_arr)
        logging.info("Model Training completed.")

        # Model Evaluation
        model_eval_obj = ModelEvaluation()
        model_eval_obj.initiate_model_evaluation(train_arr, test_arr)
        logging.info("Model Evaluation completed.")

    except Exception as e:
        logging.error(f"Exception occurred during the pipeline execution: {e}")
        raise customexception(e, sys)


if __name__ == "__main__":
    main()
