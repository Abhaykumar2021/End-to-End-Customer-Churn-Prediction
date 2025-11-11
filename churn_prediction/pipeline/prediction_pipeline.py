import sys
import pandas as pd
import numpy as np
from churn_prediction.logger import logger
from churn_prediction.exception import ChurnPredictionException
from churn_prediction.utils import load_object
from churn_prediction.constants import MODEL_PUSHER_SAVED_MODEL_DIR, MODEL_FILE_NAME


class PredictionPipeline:
    """
    Prediction Pipeline
    Handles predictions for new data
    """
    
    def __init__(self):
        """Initialize Prediction Pipeline"""
        pass
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on input features
        
        Args:
            features: Input features as DataFrame
        
        Returns:
            np.ndarray: Predictions
        """
        try:
            logger.info("Starting prediction")
            
            # Load model and preprocessor
            import os
            model_path = os.path.join(MODEL_PUSHER_SAVED_MODEL_DIR, MODEL_FILE_NAME)
            preprocessor_path = model_path.replace("model.pkl", "preprocessor.pkl")
            
            if not os.path.exists(model_path):
                raise Exception(f"Model not found at {model_path}. Please train the model first.")
            
            logger.info(f"Loading model from {model_path}")
            model = load_object(model_path)
            
            logger.info(f"Loading preprocessor from {preprocessor_path}")
            preprocessor = load_object(preprocessor_path)
            
            # Preprocess features
            logger.info("Preprocessing features")
            
            # Handle TotalCharges column if present
            if 'TotalCharges' in features.columns:
                features['TotalCharges'] = pd.to_numeric(features['TotalCharges'], errors='coerce')
            
            # Remove customerID if exists
            if 'customerID' in features.columns:
                features = features.drop(columns=['customerID'])
            
            transformed_features = preprocessor.transform(features)
            
            # Make predictions
            logger.info("Making predictions")
            predictions = model.predict(transformed_features)
            
            # Get prediction probabilities
            prediction_probs = model.predict_proba(transformed_features)
            
            logger.info("Prediction completed successfully")
            
            return predictions, prediction_probs
        
        except Exception as e:
            raise ChurnPredictionException(e, sys)


class CustomData:
    """
    Custom Data class for handling input data
    """
    
    def __init__(
        self,
        gender: str,
        SeniorCitizen: int,
        Partner: str,
        Dependents: str,
        tenure: int,
        PhoneService: str,
        MultipleLines: str,
        InternetService: str,
        OnlineSecurity: str,
        OnlineBackup: str,
        DeviceProtection: str,
        TechSupport: str,
        StreamingTV: str,
        StreamingMovies: str,
        Contract: str,
        PaperlessBilling: str,
        PaymentMethod: str,
        MonthlyCharges: float,
        TotalCharges: str
    ):
        """
        Initialize CustomData with customer information
        """
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges
    
    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Convert custom data to DataFrame
        
        Returns:
            pd.DataFrame: Input data as DataFrame
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise ChurnPredictionException(e, sys)
