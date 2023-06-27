import os
import sys

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from payment_fraud_detection.exception import CustomException
from payment_fraud_detection.logger import logging
from payment_fraud_detection.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['AccountAgeDays', 'NumItems', 'localTime', 'PaymentMethodAgeDays']
            categorical_columns = ['PaymentMethod']
            
            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )
            
            
            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Categorical Columns : {categorical_columns}")
            logging.info(f"Numerical Columns : {numerical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
    
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading training and test datasets")
            logging.info("Obtaining the preprocessor object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = 'Label'
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying Label Encoder to Target Data")
            
            le = LabelEncoder()            
            target_train_arr = le.fit_transform(target_feature_train_df)
            target_test_arr = le.fit_transform(target_feature_test_df)
            
            
            train_arr = np.c_[
                input_feature_train_arr, target_train_arr
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, target_test_arr
            ]
            
            logging.info("Saved Preprocessing Object")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )            
            
        except Exception as e:
            raise CustomException(e,sys)
        
        