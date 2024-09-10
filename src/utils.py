import pandas as pd
import sys
import os
import dill

from src.exception import CustomException
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

def save_object(filepath, object):
    try:
        directory_name = os.path.dirname(filepath)
        os.makedirs(directory_name, exist_ok=True)
        with open(filepath, "wb") as f:
            dill.dump(object, f)

    except Exception as e:
        raise CustomException(e, sys)
    
'''
We pass in X_train, y_train, X_test, y_test data + models dictionary + params for hyperparameter tuning
'''
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            chosen_param = params[list(models.keys())[i]]
            gs = GridSearchCV(model, chosen_param, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            model_test_r2 = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = model_test_r2
        
        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(filepath):
    try:
        with open(filepath, "rb") as f:
            return dill.load(f)
        
    except Exception as e:
        raise CustomException(e, sys)

    