import sys
from churn_prediction.logger import logger
from churn_prediction.exception import ChurnPredictionException
from churn_prediction.config import Configuration
from churn_prediction.components.data_ingestion import DataIngestion
from churn_prediction.components.data_validation import DataValidation
from churn_prediction.components.data_transformation import DataTransformation
from churn_prediction.components.model_trainer import ModelTrainer


class TrainingPipeline:
    """
    Training Pipeline
    Orchestrates the complete training workflow
    """
    
    def __init__(self):
        """Initialize Training Pipeline"""
        try:
            self.config = Configuration()
        except Exception as e:
            raise ChurnPredictionException(e, sys)
    
    def start_data_ingestion(self, data_path: str = None) -> tuple:
        """
        Start data ingestion stage
        
        Args:
            data_path: Path to the data file
        
        Returns:
            tuple: (train_file_path, test_file_path)
        """
        try:
            logger.info("=" * 80)
            logger.info("Starting Data Ingestion Stage")
            logger.info("=" * 80)
            
            data_ingestion_config = self.config.get_data_ingestion_config()
            data_ingestion = DataIngestion(data_ingestion_config)
            train_file_path, test_file_path = data_ingestion.initiate_data_ingestion(data_path)
            
            logger.info("Data Ingestion Stage Completed")
            logger.info("=" * 80)
            
            return train_file_path, test_file_path
        
        except Exception as e:
            raise ChurnPredictionException(e, sys)
    
    def start_data_validation(self, train_file_path: str, test_file_path: str) -> bool:
        """
        Start data validation stage
        
        Args:
            train_file_path: Path to training data
            test_file_path: Path to testing data
        
        Returns:
            bool: Validation status
        """
        try:
            logger.info("=" * 80)
            logger.info("Starting Data Validation Stage")
            logger.info("=" * 80)
            
            data_validation_config = self.config.get_data_validation_config()
            data_validation = DataValidation(data_validation_config)
            validation_status = data_validation.initiate_data_validation(
                train_file_path, test_file_path
            )
            
            logger.info("Data Validation Stage Completed")
            logger.info("=" * 80)
            
            return validation_status
        
        except Exception as e:
            raise ChurnPredictionException(e, sys)
    
    def start_data_transformation(self, train_file_path: str, test_file_path: str) -> tuple:
        """
        Start data transformation stage
        
        Args:
            train_file_path: Path to training data
            test_file_path: Path to testing data
        
        Returns:
            tuple: (train_array, test_array, preprocessor_path)
        """
        try:
            logger.info("=" * 80)
            logger.info("Starting Data Transformation Stage")
            logger.info("=" * 80)
            
            data_transformation_config = self.config.get_data_transformation_config()
            data_transformation = DataTransformation(data_transformation_config)
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_file_path, test_file_path
            )
            
            logger.info("Data Transformation Stage Completed")
            logger.info("=" * 80)
            
            return train_arr, test_arr, preprocessor_path
        
        except Exception as e:
            raise ChurnPredictionException(e, sys)
    
    def start_model_training(self, train_array, test_array) -> str:
        """
        Start model training stage
        
        Args:
            train_array: Training data array
            test_array: Testing data array
        
        Returns:
            str: Path to trained model
        """
        try:
            logger.info("=" * 80)
            logger.info("Starting Model Training Stage")
            logger.info("=" * 80)
            
            model_trainer_config = self.config.get_model_trainer_config()
            model_trainer = ModelTrainer(model_trainer_config)
            model_path = model_trainer.initiate_model_training(train_array, test_array)
            
            logger.info("Model Training Stage Completed")
            logger.info("=" * 80)
            
            return model_path
        
        except Exception as e:
            raise ChurnPredictionException(e, sys)
    
    def run_pipeline(self, data_path: str = None):
        """
        Run the complete training pipeline
        
        Args:
            data_path: Path to the data file
        """
        try:
            logger.info("*" * 80)
            logger.info("STARTING TRAINING PIPELINE")
            logger.info("*" * 80)
            
            # Data Ingestion
            train_file_path, test_file_path = self.start_data_ingestion(data_path)
            
            # Data Validation
            validation_status = self.start_data_validation(train_file_path, test_file_path)
            
            if not validation_status:
                logger.warning("Data validation failed. Continuing with training anyway...")
            
            # Data Transformation
            train_arr, test_arr, preprocessor_path = self.start_data_transformation(
                train_file_path, test_file_path
            )
            
            # Model Training
            model_path = self.start_model_training(train_arr, test_arr)
            
            logger.info("*" * 80)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Model saved at: {model_path}")
            logger.info(f"Preprocessor saved at: {preprocessor_path}")
            logger.info("*" * 80)
        
        except Exception as e:
            logger.error("Training pipeline failed")
            raise ChurnPredictionException(e, sys)
