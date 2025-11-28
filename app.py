from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            senior_citizen=int(request.form.get('senior_citizen')),
            partner=request.form.get('partner'),
            dependents=request.form.get('dependents'),
            tenure=int(request.form.get('tenure')),
            phone_service=request.form.get('phone_service'),
            multiple_lines=request.form.get('multiple_lines'),
            internet_service=request.form.get('internet_service'),
            online_security=request.form.get('online_security'),
            online_backup=request.form.get('online_backup'),
            device_protection=request.form.get('device_protection'),
            tech_support=request.form.get('tech_support'),
            streaming_tv=request.form.get('streaming_tv'),
            streaming_movies=request.form.get('streaming_movies'),
            contract=request.form.get('contract'),
            paperless_billing=request.form.get('paperless_billing'),
            payment_method=request.form.get('payment_method'),
            monthly_charges=float(request.form.get('monthly_charges')),
            total_charges=float(request.form.get('total_charges'))
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        
        prediction_text = "Churn" if results[0][0] > 0.5 else "No Churn"
        
        return render_template('home.html', results=prediction_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
