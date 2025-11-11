import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from churn_prediction.logger import logger
from churn_prediction.exception import ChurnPredictionException
from churn_prediction.entity import DataIngestionConfig


class DataIngestion:
    """
    Data Ingestion Component
    Responsible for ingesting data and splitting into train and test sets
    """
    
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initialize Data Ingestion
        
        Args:
            data_ingestion_config: Data ingestion configuration
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise ChurnPredictionException(e, sys)
    
    def initiate_data_ingestion(self, data_path: str = None) -> tuple:
        """
        Initiate data ingestion process
        
        Args:
            data_path: Path to the data file (optional, for custom data)
        
        Returns:
            tuple: (train_file_path, test_file_path)
        """
        try:
            logger.info("Starting data ingestion")
            
            # Read data
            if data_path and os.path.exists(data_path):
                logger.info(f"Reading data from {data_path}")
                df = pd.read_csv(data_path)
            else:
                # For demo purposes, create a sample dataset structure
                logger.info("No data path provided. Please provide a dataset.")
                raise Exception("Data file not found. Please provide a valid data path.")
            
            # Create feature store directory
            os.makedirs(
                os.path.dirname(self.data_ingestion_config.feature_store_file_path),
                exist_ok=True
            )
            
            # Save to feature store
            df.to_csv(
                self.data_ingestion_config.feature_store_file_path,
                index=False,
                header=True
            )
            logger.info(f"Data saved to feature store: {self.data_ingestion_config.feature_store_file_path}")
            
            # Split data into train and test
            train_set, test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )
            
            logger.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")
            
            # Create ingested directory
            os.makedirs(
                os.path.dirname(self.data_ingestion_config.training_file_path),
                exist_ok=True
            )
            
            # Save train and test sets
            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True
            )
            
            logger.info("Data ingestion completed successfully")
            
            return (
                self.data_ingestion_config.training_file_path,
                self.data_ingestion_config.testing_file_path
            )
        
        except Exception as e:
            raise ChurnPredictionException(e, sys)
