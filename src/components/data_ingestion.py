import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','data.csv')

'''
1. Takes Data from Source
2. Creates artifact directory if missing
3. Splits Data to train and test
4. Creates corresponding CSV files and exports them to the artifact directory
5. Returns Tuple(train_path,test_path)
'''
class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion Component")
        try:
            df = pd.read_csv("notebook/data/retail_sales_data.csv")
            #Data can be read from any sources like csv,database(mongodb,mysql)
            # Handling Some Features
            df = df.drop(columns=['Item_Identifier'])
            df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
                'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'
            })
            logging.info("Successfully read dataset as df")
            
            #create artifact directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            #export raw data to artifact/data.csv
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train Test Split Initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path ,index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data Ingestion Completed")

            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)

        except Exception as e:
            CustomException(e,sys)


if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)

    modelTrainer = ModelTrainer()
    r2_Score = modelTrainer.initiate_model_trainer(train_arr,test_arr)
    print("Your Model R2 score :",r2_Score)

