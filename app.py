from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys
import os

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.exception import CustomException

from src.logger import logging

application = Flask(__name__)

app = application

# Create Route for Home Page

@app.route('/')
def index():
    logging.info('checkpoint')
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        logging.info('Home')
        return render_template('home.html')
    else:
        try: 
            data=CustomData(
            sex=request.form.get('sex'),
            smoker=request.form.get('smoker'),
            region=request.form.get('region'),
            age=float(request.form.get('age')),
            bmi=float(request.form.get('bmi')),
            children=float(request.form.get('children'))
            )
            logging.info('Data obtained')
            prediction_df = data.make_data_frame()
            print(prediction_df)
            logging.info('prediction made')

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(prediction_df)
            logging.info('results obtained')

            return render_template('home.html', results = results[0])
        
        except Exception as e:
            raise CustomException(e, sys)
            
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.get('/shutdown')
def shutdown():
    shutdown_server()
    logging.info('shutting down')
    return 'Server shutting down...'

if __name__ == "__main__":
    app.run(host = "0.0.0.0")
