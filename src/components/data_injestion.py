import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformation
from src.components.training_model import ModelTrainer


@dataclass
# This class holds paths to files for train data, test data and raw data
class DataInjestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv') # Train data is assigned a path 'artifacts/train.csv'
    test_data_path = os.path.join('artifacts', 'test.csv') # Test data is assigned a path 'artifacts/test.csv'
    raw_data_path = os.path.join('artifacts', 'rawdata.csv') # Raw data is assigned a path 'artifacts/rawdata.csv'

# This class will actually do the data injestion
class DataInjestion:
    def __init__(self):
        self.injestion_config = DataInjestionConfig() # what does this look like
    
    # This function returns 2 things: train data, test data paths
    def initiate_data_injestion(self):
        logging.info("Data Injestion started")
        try:
            df = pd.read_csv("notebook/data/insurance.csv")
            logging.info("read the dataframe")
            # This creates the leaf directory where training data will be saved and technically all the other files - creates 'artifacts/'
            os.makedirs(
                name=os.path.dirname(self.injestion_config.train_data_path),
                exist_ok=True
            )
            logging.info("artifacts created")
            # saves the dataframe to 'artifacts/rawdata.csv'
            df.to_csv(self.injestion_config.raw_data_path, index=False, header = True)
            logging.info("creates artifacts/rawdata.csv")
            # now we split the dataframe into train and test data
            train_data, test_data = train_test_split(df, test_size = 0.2, random_state=42)
            logging.info("split data")
            # saves train_data df to 'artifacts/train.csv'
            train_data.to_csv(self.injestion_config.train_data_path, index=False, header = True)
            logging.info("creates artifacts/train.csv")
            # saves test_data df to 'artifacts/test.csv'
            test_data.to_csv(self.injestion_config.test_data_path, index=False, header = True)
            logging.info("creates artifacts/test.csv")

            return (
                self.injestion_config.train_data_path, # returns 'artifacts/train.csv'
                self.injestion_config.test_data_path # returns 'artifacts/test.csv'
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataInjestion()
    train_data_path, test_data_path = obj.initiate_data_injestion()

    obj2 = DataTransformation()
    train_array, test_array, pkl_file_path = obj2.Initiate_Data_Transformation(train_data_path, test_data_path)

    obj3 = ModelTrainer()
    score = obj3.initiate_model_trainer(train_array, test_array)

    print(score)