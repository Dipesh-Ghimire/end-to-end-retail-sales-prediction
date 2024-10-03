import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self)->None:
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            numerical_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']
            categorical_columns = ['Item_Fat_Content', 'Item_Type', 
                                   'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="mean")),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("Numerical Columns scaling Completed")
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore"))
                ]
            )
            logging.info("Categorical Columns encoding Completed")

            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read Train and Test Data Successfully")

            logging.info("Obtaining Pre-processing Object")
            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'Item_Outlet_Sales'
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_features_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_features_train_dense = input_features_train_arr.toarray()
            
            input_features_test_arr = preprocessor_obj.transform(input_feature_test_df)
            input_features_test_dense = input_features_test_arr.toarray()
            logging.info("Input_features train array completed")
            train_arr = np.c_[input_features_train_dense,np.array(target_feature_train_df).reshape(-1, 1)]
            logging.info("train array completed")
            test_arr = np.c_[input_features_test_dense,np.array(target_feature_test_df).reshape(-1, 1)]
            logging.info("Preprocessing Object Saving Initiated")

            #saving pickle file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            logging.info("Preprocessing Object Saving Completed")
            return(train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)