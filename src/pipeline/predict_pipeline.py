import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def  __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 item_weight:float,
                 item_visibility:float,
                 item_mrp:float,
                 item_fat_content:str,
                 item_type:str,
                 outlet_identifier:str,
                 outlet_establishment_year:int,
                 outlet_type:str,
                 outlet_size:str,
                 outlet_location_type:str
                 ):
        self.item_weight = item_weight
        self.item_visibility = item_visibility
        self.item_mrp = item_mrp
        self.item_fat_content = item_fat_content
        self.item_type = item_type
        self.outlet_identifier = outlet_identifier
        self.outlet_establishment_year = outlet_establishment_year
        self.outlet_type = outlet_type
        self.outlet_size = outlet_size
        self.outlet_location_type = outlet_location_type
    
    def get_data_as_data_frame(self):
        try:
            cutom_dta_input_dict = {
                "Item_Weight" : [self.item_weight],
                "Item_Visibility" : [self.item_visibility],
                "Item_MRP" : [self.item_mrp],
                "Item_Type" : [self.item_type],
                "Item_Fat_Content" : [self.item_fat_content],
                "Outlet_Identifier" : [self.outlet_identifier],
                "Outlet_Establishment_Year" : [self.outlet_establishment_year],
                "Outlet_Type" : [self.outlet_type],
                "Outlet_Size" : [self.outlet_size],
                "Outlet_Location_Type" : [self.outlet_location_type]
            }
            return pd.DataFrame(cutom_dta_input_dict)
        except Exception as e:
            raise CustomException(e,sys)