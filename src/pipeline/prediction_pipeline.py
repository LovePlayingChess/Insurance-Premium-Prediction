import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            scaled_data = preprocessor.transform(features)
            prediction = model.predict(scaled_data)
            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self, sex: str, smoker: str,region: str, 
                age: float, bmi: float, children: float):
        self.sex = sex
        self.smoker = smoker
        self.region = region
        self.age = age
        self.bmi = bmi
        self.children = children
    
    def make_data_frame(self):
        try:
            d = {
                "sex": [self.sex],
                "smoker": [self.smoker],
                "region": [self.region],
                "age": [self.age],
                "bmi": [self.bmi],
                "children": [self.children],
            }
            return pd.DataFrame(d)
        except Exception as e:
            raise CustomException(e, sys)