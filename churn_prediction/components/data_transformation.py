import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from churn_prediction.logger import logger
from churn_prediction.exception import ChurnPredictionException
from churn_prediction.entity import DataTransformationConfig
from churn_prediction.utils import save_object, save_numpy_array_data
from churn_prediction.constants import TARGET_COLUMN, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS


class DataTransformation:
    """
    Data Transformation Component
    Responsible for transforming and preprocessing data
    """
    
    def __init__(self, data_transformation_config: DataTransformationConfig):
        """
        Initialize Data Transformation
        
        Args:
            data_transformation_config: Data transformation configuration
        """
        try:
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise ChurnPredictionException(e, sys)
    
    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Create and return data transformation pipeline
        
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        try:
            logger.info("Creating data transformation pipeline")
            
            # Numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Categorical pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
                ]
            )
            
            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numerical_pipeline, NUMERICAL_COLUMNS),
                    ("cat_pipeline", categorical_pipeline, CATEGORICAL_COLUMNS)
                ]
            )
            
            logger.info("Data transformation pipeline created successfully")
            
            return preprocessor
        
        except Exception as e:
            raise ChurnPredictionException(e, sys)
    
    def initiate_data_transformation(self, train_file_path: str, test_file_path: str) -> tuple:
        """
        Initiate data transformation process
        
        Args:
            train_file_path: Path to training data
            test_file_path: Path to testing data
        
        Returns:
            tuple: (train_array, test_array, preprocessor_path)
        """
        try:
            logger.info("Starting data transformation")
            
            # Read train and test data
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            logger.info(f"Train data shape: {train_df.shape}")
            logger.info(f"Test data shape: {test_df.shape}")
            
            # Handle TotalCharges column (convert to numeric)
            if 'TotalCharges' in train_df.columns:
                train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'], errors='coerce')
                test_df['TotalCharges'] = pd.to_numeric(test_df['TotalCharges'], errors='coerce')
            
            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()
            
            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            
            # Remove customerID if exists
            if 'customerID' in input_feature_train_df.columns:
                input_feature_train_df = input_feature_train_df.drop(columns=['customerID'])
                input_feature_test_df = input_feature_test_df.drop(columns=['customerID'])
            
            logger.info("Applying preprocessing object on training and testing datasets")
            
            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Encode target variable
            label_encoder = LabelEncoder()
            target_feature_train_arr = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)
            
            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]
            
            logger.info(f"Transformed train data shape: {train_arr.shape}")
            logger.info(f"Transformed test data shape: {test_arr.shape}")
            
            # Save transformed data
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )
            
            # Save preprocessing object
            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessing_obj
            )
            
            logger.info("Data transformation completed successfully")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise ChurnPredictionException(e, sys)
