import os
import sys
import numpy as np
import pandas as pd

from payment_fraud_detection.exception import CustomException
from payment_fraud_detection.logger import logging
from payment_fraud_detection.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                AccountAgeDays: int,
                NumItems: int,
                localTime: int,
                PaymentMethod: str,
                PaymentMethodAgeDays: int
                ):
        self.AccountAgeDays = AccountAgeDays
        self.NumItems = NumItems
        self.localTime = localTime
        self.PaymentMethod = PaymentMethod
        self.PaymentMethodAgeDays = PaymentMethodAgeDays
            
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "AccountAgeDays": [self.AccountAgeDays],
                "NumItems": [self.NumItems],
                "localTime": [self.localTime],
                "PaymentMethod": [self.PaymentMethod],
                "PaymentMethodAgeDays": [self.PaymentMethodAgeDays]
            }
            
            return pd.DataFrame(custom_data_input_dict)                
                
        except Exception as e:
            raise CustomException(e,sys)