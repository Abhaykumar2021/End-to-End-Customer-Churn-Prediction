import os
from churn_prediction.constants import (
    ARTIFACTS_DIR,
    DATA_INGESTION_DIR_NAME,
    DATA_INGESTION_FEATURE_STORE_DIR,
    DATA_INGESTION_INGESTED_DIR,
    DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO,
    DATA_VALIDATION_DIR_NAME,
    DATA_VALIDATION_DRIFT_REPORT_DIR,
    DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
    DATA_TRANSFORMATION_DIR_NAME,
    DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    PREPROCESSOR_OBJ_FILE_NAME,
    MODEL_TRAINER_DIR_NAME,
    MODEL_TRAINER_TRAINED_MODEL_DIR,
    MODEL_TRAINER_EXPECTED_SCORE,
    MODEL_TRAINER_MODEL_CONFIG_FILE_PATH,
    MODEL_FILE_NAME,
    MODEL_EVALUATION_DIR_NAME,
    MODEL_EVALUATION_REPORT_NAME,
    MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE,
    MODEL_PUSHER_DIR_NAME,
    MODEL_PUSHER_SAVED_MODEL_DIR,
)
from churn_prediction.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    TrainingPipelineConfig,
)


class Configuration:
    """
    Configuration class for the training pipeline
    """
    
    def __init__(self):
        self.training_pipeline_config = self.get_training_pipeline_config()
    
    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        """Get training pipeline configuration"""
        return TrainingPipelineConfig(artifact_dir=ARTIFACTS_DIR)
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get data ingestion configuration"""
        artifact_dir = self.training_pipeline_config.artifact_dir
        
        data_ingestion_dir = os.path.join(artifact_dir, DATA_INGESTION_DIR_NAME)
        
        feature_store_file_path = os.path.join(
            data_ingestion_dir,
            DATA_INGESTION_FEATURE_STORE_DIR,
            "data.csv"
        )
        
        training_file_path = os.path.join(
            data_ingestion_dir,
            DATA_INGESTION_INGESTED_DIR,
            "train.csv"
        )
        
        testing_file_path = os.path.join(
            data_ingestion_dir,
            DATA_INGESTION_INGESTED_DIR,
            "test.csv"
        )
        
        return DataIngestionConfig(
            feature_store_file_path=feature_store_file_path,
            training_file_path=training_file_path,
            testing_file_path=testing_file_path,
            train_test_split_ratio=DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        )
    
    def get_data_validation_config(self) -> DataValidationConfig:
        """Get data validation configuration"""
        artifact_dir = self.training_pipeline_config.artifact_dir
        
        data_validation_dir = os.path.join(artifact_dir, DATA_VALIDATION_DIR_NAME)
        
        drift_report_file_path = os.path.join(
            data_validation_dir,
            DATA_VALIDATION_DRIFT_REPORT_DIR,
            DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )
        
        return DataValidationConfig(
            data_validation_dir=data_validation_dir,
            drift_report_file_path=drift_report_file_path
        )
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """Get data transformation configuration"""
        artifact_dir = self.training_pipeline_config.artifact_dir
        
        data_transformation_dir = os.path.join(
            artifact_dir,
            DATA_TRANSFORMATION_DIR_NAME
        )
        
        transformed_train_file_path = os.path.join(
            data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            "train.npy"
        )
        
        transformed_test_file_path = os.path.join(
            data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            "test.npy"
        )
        
        preprocessor_obj_file_path = os.path.join(
            data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            PREPROCESSOR_OBJ_FILE_NAME
        )
        
        return DataTransformationConfig(
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessor_obj_file_path=preprocessor_obj_file_path
        )
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """Get model trainer configuration"""
        artifact_dir = self.training_pipeline_config.artifact_dir
        
        model_trainer_dir = os.path.join(artifact_dir, MODEL_TRAINER_DIR_NAME)
        
        trained_model_file_path = os.path.join(
            model_trainer_dir,
            MODEL_TRAINER_TRAINED_MODEL_DIR,
            MODEL_FILE_NAME
        )
        
        return ModelTrainerConfig(
            trained_model_file_path=trained_model_file_path,
            expected_accuracy=MODEL_TRAINER_EXPECTED_SCORE,
            model_config_file_path=MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
        )
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """Get model evaluation configuration"""
        artifact_dir = self.training_pipeline_config.artifact_dir
        
        model_evaluation_dir = os.path.join(artifact_dir, MODEL_EVALUATION_DIR_NAME)
        
        model_evaluation_report_file_path = os.path.join(
            model_evaluation_dir,
            MODEL_EVALUATION_REPORT_NAME
        )
        
        return ModelEvaluationConfig(
            changed_threshold_score=MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE,
            model_evaluation_report_file_path=model_evaluation_report_file_path
        )
    
    def get_model_pusher_config(self) -> ModelPusherConfig:
        """Get model pusher configuration"""
        model_pusher_dir = os.path.join(ARTIFACTS_DIR, MODEL_PUSHER_DIR_NAME)
        
        saved_model_path = os.path.join(
            MODEL_PUSHER_SAVED_MODEL_DIR,
            MODEL_FILE_NAME
        )
        
        return ModelPusherConfig(
            model_pusher_dir=model_pusher_dir,
            saved_model_path=saved_model_path
        )
