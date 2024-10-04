import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def start_pipeline(self):
        try:
            # Step 1: Data Ingestion
            logging.info("Starting Data Ingestion")
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            logging.info("Starting Data Transformation")
            train_array, test_array, preprocessor_path = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)

            # Step 3: Model Training
            logging.info("Starting Model Training")
            r2_score = self.model_trainer.initiate_model_trainer(train_array, test_array)
            logging.info(f"Model Training Completed with R2 Score: {r2_score}")

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    pipeline = TrainPipeline()
    pipeline.start_pipeline()