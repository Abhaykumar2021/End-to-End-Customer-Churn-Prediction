import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from churn_prediction.logger import logger
from churn_prediction.exception import ChurnPredictionException
from churn_prediction.entity import ModelTrainerConfig
from churn_prediction.utils import save_object, evaluate_models, get_best_model, read_yaml_file


class ModelTrainer:
    """
    Model Trainer Component
    Responsible for training and evaluating models
    """
    
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        """
        Initialize Model Trainer
        
        Args:
            model_trainer_config: Model trainer configuration
        """
        try:
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise ChurnPredictionException(e, sys)
    
    def initiate_model_training(self, train_array, test_array) -> str:
        """
        Initiate model training process
        
        Args:
            train_array: Training data array
            test_array: Testing data array
        
        Returns:
            str: Path to trained model
        """
        try:
            logger.info("Starting model training")
            
            # Split features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            logger.info(f"Train set - Features: {X_train.shape}, Target: {y_train.shape}")
            logger.info(f"Test set - Features: {X_test.shape}, Target: {y_test.shape}")
            
            # Define models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
            }
            
            # Define hyperparameters for tuning
            params = {
                "Logistic Regression": {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2']
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                },
                "XGBoost": {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
            }
            
            # Evaluate models
            logger.info("Evaluating models with hyperparameter tuning")
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )
            
            # Get best model
            best_model_name, best_model, best_model_score = get_best_model(
                model_report=model_report,
                threshold=self.model_trainer_config.expected_accuracy
            )
            
            logger.info(f"Best model: {best_model_name}")
            logger.info(f"Best model accuracy: {best_model_score:.4f}")
            
            # Save the best model
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True
            )
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logger.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")
            logger.info("Model training completed successfully")
            
            return self.model_trainer_config.trained_model_file_path
        
        except Exception as e:
            raise ChurnPredictionException(e, sys)
