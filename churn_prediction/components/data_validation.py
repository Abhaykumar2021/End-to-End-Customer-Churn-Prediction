import os
import sys
import pandas as pd
from scipy import stats

from churn_prediction.logger import logger
from churn_prediction.exception import ChurnPredictionException
from churn_prediction.entity import DataValidationConfig
from churn_prediction.utils import read_yaml_file, write_yaml_file
from churn_prediction.constants import SCHEMA_FILE_PATH


class DataValidation:
    """
    Data Validation Component
    Responsible for validating data schema and detecting data drift
    """
    
    def __init__(self, data_validation_config: DataValidationConfig):
        """
        Initialize Data Validation
        
        Args:
            data_validation_config: Data validation configuration
        """
        try:
            self.data_validation_config = data_validation_config
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise ChurnPredictionException(e, sys)
    
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate number of columns in dataframe
        
        Args:
            dataframe: Input dataframe
        
        Returns:
            bool: True if validation passes
        """
        try:
            expected_columns = len(self.schema['columns'])
            actual_columns = len(dataframe.columns)
            
            if expected_columns == actual_columns:
                logger.info(f"Column count validation passed: {actual_columns} columns")
                return True
            else:
                logger.warning(f"Column count mismatch: Expected {expected_columns}, Got {actual_columns}")
                return False
        except Exception as e:
            raise ChurnPredictionException(e, sys)
    
    def validate_column_names(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate column names in dataframe
        
        Args:
            dataframe: Input dataframe
        
        Returns:
            bool: True if validation passes
        """
        try:
            expected_columns = set(self.schema['columns'].keys())
            actual_columns = set(dataframe.columns)
            
            missing_columns = expected_columns - actual_columns
            extra_columns = actual_columns - expected_columns
            
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
                return False
            
            if extra_columns:
                logger.warning(f"Extra columns: {extra_columns}")
                return False
            
            logger.info("Column names validation passed")
            return True
        except Exception as e:
            raise ChurnPredictionException(e, sys)
    
    def detect_dataset_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        """
        Detect data drift between reference and current datasets using statistical tests
        
        Args:
            reference_df: Reference dataframe (train data)
            current_df: Current dataframe (test data)
        
        Returns:
            bool: True if no drift detected
        """
        try:
            logger.info("Starting data drift detection")
            
            # Get numerical columns for drift detection
            numerical_cols = reference_df.select_dtypes(include=['int64', 'float64']).columns
            
            drift_detected = False
            drift_by_columns = {}
            
            # Use Kolmogorov-Smirnov test for numerical columns
            for col in numerical_cols:
                if col in current_df.columns:
                    # Remove NaN values
                    ref_col = reference_df[col].dropna()
                    curr_col = current_df[col].dropna()
                    
                    if len(ref_col) > 0 and len(curr_col) > 0:
                        # Perform KS test
                        statistic, p_value = stats.ks_2samp(ref_col, curr_col)
                        
                        # Drift detected if p-value < 0.05
                        col_drift = p_value < 0.05
                        drift_by_columns[col] = {
                            'drift_detected': bool(col_drift),
                            'p_value': float(p_value),
                            'statistic': float(statistic)
                        }
                        
                        if col_drift:
                            drift_detected = True
                            logger.warning(f"Drift detected in column '{col}' (p-value: {p_value:.4f})")
            
            # Save report
            os.makedirs(
                os.path.dirname(self.data_validation_config.drift_report_file_path),
                exist_ok=True
            )
            
            # Save to YAML
            drift_info = {
                'dataset_drift': drift_detected,
                'drift_by_columns': drift_by_columns
            }
            
            write_yaml_file(self.data_validation_config.drift_report_file_path, drift_info)
            
            logger.info(f"Data drift report saved to {self.data_validation_config.drift_report_file_path}")
            logger.info(f"Dataset drift detected: {drift_detected}")
            
            return not drift_detected
        
        except Exception as e:
            raise ChurnPredictionException(e, sys)
    
    def initiate_data_validation(self, train_file_path: str, test_file_path: str) -> bool:
        """
        Initiate data validation process
        
        Args:
            train_file_path: Path to training data
            test_file_path: Path to testing data
        
        Returns:
            bool: True if validation passes
        """
        try:
            logger.info("Starting data validation")
            
            # Read data
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            # Validate train data
            train_col_count_valid = self.validate_number_of_columns(train_df)
            train_col_names_valid = self.validate_column_names(train_df)
            
            # Validate test data
            test_col_count_valid = self.validate_number_of_columns(test_df)
            test_col_names_valid = self.validate_column_names(test_df)
            
            # Check for data drift
            drift_status = self.detect_dataset_drift(train_df, test_df)
            
            # Overall validation status
            validation_status = all([
                train_col_count_valid,
                train_col_names_valid,
                test_col_count_valid,
                test_col_names_valid,
                drift_status
            ])
            
            logger.info(f"Data validation completed. Status: {validation_status}")
            
            return validation_status
        
        except Exception as e:
            raise ChurnPredictionException(e, sys)
