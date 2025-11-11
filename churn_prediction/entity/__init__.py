from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    """Data Ingestion Configuration"""
    feature_store_file_path: str
    training_file_path: str
    testing_file_path: str
    train_test_split_ratio: float


@dataclass
class DataValidationConfig:
    """Data Validation Configuration"""
    data_validation_dir: str
    drift_report_file_path: str
    

@dataclass
class DataTransformationConfig:
    """Data Transformation Configuration"""
    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessor_obj_file_path: str


@dataclass
class ModelTrainerConfig:
    """Model Trainer Configuration"""
    trained_model_file_path: str
    expected_accuracy: float
    model_config_file_path: str


@dataclass
class ModelEvaluationConfig:
    """Model Evaluation Configuration"""
    changed_threshold_score: float
    model_evaluation_report_file_path: str


@dataclass
class ModelPusherConfig:
    """Model Pusher Configuration"""
    model_pusher_dir: str
    saved_model_path: str
    

@dataclass
class TrainingPipelineConfig:
    """Training Pipeline Configuration"""
    artifact_dir: str
