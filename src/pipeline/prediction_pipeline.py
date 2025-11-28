import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from tensorflow.keras.models import load_model

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.h5")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            
            print("Loading model...")
            model=load_model(model_path)
            
            print("Loading preprocessor...")
            preprocessor=load_object(file_path=preprocessor_path)
            
            # Ensure numerical columns are numeric
            features['Tenure Months'] = pd.to_numeric(features['Tenure Months'])
            features['Monthly Charges'] = pd.to_numeric(features['Monthly Charges'])
            features['Total Charges'] = pd.to_numeric(features['Total Charges'])

            # Ensure categorical columns are strings
            cat_cols = [
                'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 
                'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup', 
                'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies', 
                'Contract', 'Paperless Billing', 'Payment Method'
            ]
            for col in cat_cols:
                features[col] = features[col].astype(str)
            
            data_scaled=preprocessor.transform(features)
            
            print("Making prediction...")
            preds=model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        gender: str,
        senior_citizen: int,
        partner: str,
        dependents: str,
        tenure: int,
        phone_service: str,
        multiple_lines: str,
        internet_service: str,
        online_security: str,
        online_backup: str,
        device_protection: str,
        tech_support: str,
        streaming_tv: str,
        streaming_movies: str,
        contract: str,
        paperless_billing: str,
        payment_method: str,
        monthly_charges: float,
        total_charges: float):

        self.gender = gender
        self.senior_citizen = senior_citizen
        self.partner = partner
        self.dependents = dependents
        self.tenure = tenure
        self.phone_service = phone_service
        self.multiple_lines = multiple_lines
        self.internet_service = internet_service
        self.online_security = online_security
        self.online_backup = online_backup
        self.device_protection = device_protection
        self.tech_support = tech_support
        self.streaming_tv = streaming_tv
        self.streaming_movies = streaming_movies
        self.contract = contract
        self.paperless_billing = paperless_billing
        self.payment_method = payment_method
        self.monthly_charges = monthly_charges
        self.total_charges = total_charges

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.gender],
                "Senior Citizen": [self.senior_citizen],
                "Partner": [self.partner],
                "Dependents": [self.dependents],
                "Tenure Months": [self.tenure],
                "Phone Service": [self.phone_service],
                "Multiple Lines": [self.multiple_lines],
                "Internet Service": [self.internet_service],
                "Online Security": [self.online_security],
                "Online Backup": [self.online_backup],
                "Device Protection": [self.device_protection],
                "Tech Support": [self.tech_support],
                "Streaming TV": [self.streaming_tv],
                "Streaming Movies": [self.streaming_movies],
                "Contract": [self.contract],
                "Paperless Billing": [self.paperless_billing],
                "Payment Method": [self.payment_method],
                "Monthly Charges": [self.monthly_charges],
                "Total Charges": [self.total_charges],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
