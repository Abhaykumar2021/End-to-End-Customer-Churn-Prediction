import os
import sys
from flask import Flask, request, render_template, jsonify
from churn_prediction.logger import logger
from churn_prediction.exception import ChurnPredictionException
from churn_prediction.pipeline.prediction_pipeline import PredictionPipeline, CustomData

app = Flask(__name__)


@app.route('/')
def index():
    """
    Home page
    """
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Prediction endpoint
    """
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        try:
            # Get form data
            data = CustomData(
                gender=request.form.get('gender'),
                SeniorCitizen=int(request.form.get('SeniorCitizen')),
                Partner=request.form.get('Partner'),
                Dependents=request.form.get('Dependents'),
                tenure=int(request.form.get('tenure')),
                PhoneService=request.form.get('PhoneService'),
                MultipleLines=request.form.get('MultipleLines'),
                InternetService=request.form.get('InternetService'),
                OnlineSecurity=request.form.get('OnlineSecurity'),
                OnlineBackup=request.form.get('OnlineBackup'),
                DeviceProtection=request.form.get('DeviceProtection'),
                TechSupport=request.form.get('TechSupport'),
                StreamingTV=request.form.get('StreamingTV'),
                StreamingMovies=request.form.get('StreamingMovies'),
                Contract=request.form.get('Contract'),
                PaperlessBilling=request.form.get('PaperlessBilling'),
                PaymentMethod=request.form.get('PaymentMethod'),
                MonthlyCharges=float(request.form.get('MonthlyCharges')),
                TotalCharges=request.form.get('TotalCharges')
            )
            
            # Convert to DataFrame
            pred_df = data.get_data_as_dataframe()
            logger.info(f"Input data: {pred_df.to_dict()}")
            
            # Make prediction
            predict_pipeline = PredictionPipeline()
            predictions, prediction_probs = predict_pipeline.predict(pred_df)
            
            # Format result
            result = "Yes" if predictions[0] == 1 else "No"
            confidence = prediction_probs[0][predictions[0]] * 100
            
            logger.info(f"Prediction: {result}, Confidence: {confidence:.2f}%")
            
            return render_template(
                'predict.html',
                prediction_text=f"Customer Churn Prediction: {result}",
                confidence_text=f"Confidence: {confidence:.2f}%"
            )
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return render_template(
                'predict.html',
                prediction_text="Error occurred during prediction. Please check your inputs and try again."
            )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions
    """
    try:
        # Get JSON data
        json_data = request.get_json()
        
        # Create CustomData object
        data = CustomData(
            gender=json_data.get('gender'),
            SeniorCitizen=int(json_data.get('SeniorCitizen')),
            Partner=json_data.get('Partner'),
            Dependents=json_data.get('Dependents'),
            tenure=int(json_data.get('tenure')),
            PhoneService=json_data.get('PhoneService'),
            MultipleLines=json_data.get('MultipleLines'),
            InternetService=json_data.get('InternetService'),
            OnlineSecurity=json_data.get('OnlineSecurity'),
            OnlineBackup=json_data.get('OnlineBackup'),
            DeviceProtection=json_data.get('DeviceProtection'),
            TechSupport=json_data.get('TechSupport'),
            StreamingTV=json_data.get('StreamingTV'),
            StreamingMovies=json_data.get('StreamingMovies'),
            Contract=json_data.get('Contract'),
            PaperlessBilling=json_data.get('PaperlessBilling'),
            PaymentMethod=json_data.get('PaymentMethod'),
            MonthlyCharges=float(json_data.get('MonthlyCharges')),
            TotalCharges=json_data.get('TotalCharges')
        )
        
        # Convert to DataFrame
        pred_df = data.get_data_as_dataframe()
        
        # Make prediction
        predict_pipeline = PredictionPipeline()
        predictions, prediction_probs = predict_pipeline.predict(pred_df)
        
        # Format response
        result = {
            'churn': bool(predictions[0] == 1),
            'confidence': float(prediction_probs[0][predictions[0]]),
            'probabilities': {
                'no_churn': float(prediction_probs[0][0]),
                'churn': float(prediction_probs[0][1])
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction. Please check your input data.'}), 400


if __name__ == '__main__':
    import os
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
