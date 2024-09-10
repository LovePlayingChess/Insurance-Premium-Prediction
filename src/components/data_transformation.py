import os
import pandas as pd
import sys
import numpy as np

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        '''
        This function helps transform data depending on whether
        features are numerical or categorical, applying imputation,
        scaling, and encoding techniques
        '''
        try:
            # Define our numerical and categorical columns
            numerical_columns = ['age', 'bmi', 'children']
            categorical_columns = ['sex', 'smoker', 'region']
            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("Scaler", StandardScaler())
                ]
            )
            logging.info("Numerical columns scaled properly")

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder", OneHotEncoder()),
                    ("Scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical columns scaled properly")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns),
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
           
    def Initiate_Data_Transformation(self, train_path, test_path):
        try:
            # read the paths created from data injestion
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = "expenses"

            train_df_without_target = train_df.drop(columns = [target_column_name], axis = 1) # holds all the other features from train data set
            train_df_target_column = train_df[target_column_name] # holds all the expenses from train data set

            test_df_without_target = test_df.drop(columns = [target_column_name], axis = 1) # holds all the other features from test data set
            test_df_target_column = test_df[target_column_name] # holds all the expenses from train data set

            logging.info("Apply preprocessing")

            preprocessed_train_arr = preprocessing_obj.fit_transform(train_df_without_target)
            preprocessed_test_arr = preprocessing_obj.fit_transform(test_df_without_target)

            # now stack the preprocessed arrays with the respective target columns

            train_arr = np.c_[
                preprocessed_train_arr, np.array(train_df_target_column)
            ]
            test_arr = np.c_[
                preprocessed_test_arr, np.array(test_df_target_column)
            ]

            # Conclusion: We have scaled all the input features properly and joined the y-variable column (expenses)
            logging.info("Our train and test arrays have been properly built")

            save_object(
                filepath = self.data_transformation_config.preprocessor_obj_file_path, 
                object = preprocessing_obj
            )

            logging.info("Object has been correctly saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path # preprocessor.pkl file
            )

        except Exception as e:
            raise CustomException(e, sys)