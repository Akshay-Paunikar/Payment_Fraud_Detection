import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

from payment_fraud_detection.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            AccountAgeDays = float(request.form.get("AccountAgeDays")),
            NumItems = float(request.form.get("NumItems")),
            localTime = float(request.form.get("localTime")),
            PaymentMethod = request.form.get("PaymentMethod"),
            PaymentMethodAgeDays = float(request.form.get("PaymentMethodAgeDays"))
        )
        
        pred_df = data.get_data_as_data_frame()
        
        preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        if results == 1:
            payment = "Fraud"
        else:
            payment = "Good"
        
        return render_template('index.html', results=payment)
    
if __name__ == "__main__":
    app.run(debug=True)